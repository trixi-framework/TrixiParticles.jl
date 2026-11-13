"""
    merge_particles!(semi, v_ode, u_ode; before_splitting=false)

Merge pairs of particles of fluid systems with particle refinement (Alg. 3 in Haftu et al.).
A particle merges with its nearest neighbor if both particles choose each other, if their
distance is smaller than `(h_a + h_b) / 2` and if the merged mass is smaller than
`max_mass_factor` times the smaller reference mass of both particles.

As in the code of Haftu et al., merging is done in two phases:
- Before splitting (`before_splitting=true`), all particles below `max_mass_factor` times
  their reference mass can merge. This is done in one round.
- After splitting, only particles below `min_mass_factor` times their reference mass
  can merge, so that new particles of a split are only merged back if they are
  too light for their target spacing. This is done in up to three rounds.

The particle with the smaller index takes the merged properties, the other particle
is deactivated, so that its slot in the `SystemBuffer` can be reused.
"""
function merge_particles!(semi, v_ode, u_ode; before_splitting=false)
    n_merged = 0

    foreach_system(semi) do system
        n_merged += merge_particles!(system, v_ode, u_ode, semi; before_splitting)
    end

    # The neighborhood searches must not find the deactivated particles
    n_merged > 0 && update_nhs!(semi, u_ode)

    return semi
end

# Return the number of merged pairs
@inline merge_particles!(system, v_ode, u_ode, semi; before_splitting) = 0

@inline function merge_particles!(system::AbstractFluidSystem, v_ode, u_ode, semi;
                                  before_splitting)
    return merge_particles!(system, get_refinement(system), v_ode, u_ode, semi;
                            before_splitting)
end

@inline merge_particles!(system, ::Nothing, v_ode, u_ode, semi; before_splitting) = 0

function merge_particles!(system, refinement, v_ode, u_ode, semi; before_splitting)
    (; max_mass_factor, min_mass_factor) = refinement

    if before_splitting
        mass_limit_factor = max_mass_factor
        n_rounds = 1
    else
        mass_limit_factor = min_mass_factor
        n_rounds = 3
    end

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)
    neighborhood_search = get_neighborhood_search(system, semi)

    n_merged = 0
    for _ in 1:n_rounds
        collect_merge_partners!(system, u, semi, mass_limit_factor)

        n_pairs = count_merge_pairs(system)
        # Without new pairs, the following rounds would find the same partners
        n_pairs == 0 && break

        if !(system.buffer isa SystemBuffer)
            throw(ArgumentError("merging particles requires a `SystemBuffer`. " *
                                "Pass `buffer_size` to the fluid system."))
        end

        apply_merging!(system, v, u, semi)
        n_merged += n_pairs

        # The next round must not find the deactivated particles
        update_system_buffer!(system.buffer)
        update_nhs!(neighborhood_search, system, system, u, u, semi)
    end

    return n_merged
end

# Find the merge partner of each particle. A particle without a partner gets the partner 0.
function collect_merge_partners!(system, u, semi, mass_limit_factor)
    (; merge_partner) = system.cache
    (; max_mass_factor) = get_refinement(system)
    ELTYPE = eltype(system)

    set_zero!(merge_partner)

    system_coords = current_coordinates(u, system)
    neighborhood_search = get_neighborhood_search(system, semi)

    # The nearest neighbor and its distance.
    # With equal distances, take the smaller index to be independent of the loop order.
    init = (0, typemax(ELTYPE))
    nearest(a, b) = (a[2], a[1]) <= (b[2], b[1]) ? a : b

    @threaded semi for particle in each_integrated_particle(system)
        m_a = hydrodynamic_mass(system, particle)
        m_ref_a = reference_mass(system, particle)
        m_a < mass_limit_factor * m_ref_a || return

        h_a = smoothing_length(system, particle)

        partner,
        _ = mapreduce_neighbor(nearest, system_coords, system_coords,
                               neighborhood_search, semi.parallelization_backend,
                               particle; init) do particle, neighbor, _, distance
            neighbor == particle && return init

            h_b = smoothing_length(system, neighbor)
            distance < (h_a + h_b) / 2 || return init

            m_b = hydrodynamic_mass(system, neighbor)
            m_ref_b = reference_mass(system, neighbor)
            m_b < mass_limit_factor * m_ref_b || return init

            # The merged particle must not be split again
            m_a + m_b < max_mass_factor * min(m_ref_a, m_ref_b) || return init

            return (neighbor, distance)
        end

        merge_partner[particle] = partner
    end

    return system
end

# A pair is merged if both particles chose each other
@inline function is_merge_pair(merge_partner, particle)
    partner = merge_partner[particle]

    return partner > 0 && merge_partner[partner] == particle
end

function count_merge_pairs(system)
    (; merge_partner) = system.cache

    # Count each pair once, from the particle with the smaller index
    return count(particle -> particle < merge_partner[particle] &&
                             is_merge_pair(merge_partner, particle),
                 each_integrated_particle(system))
end

function apply_merging!(system, v, u, semi)
    (; merge_partner) = system.cache
    (; smoothing_kernel) = system
    ELTYPE = eltype(system)

    periodic_box = get_neighborhood_search(system, semi).periodic_box
    kernel_0 = kernel(smoothing_kernel, zero(ELTYPE), one(ELTYPE))

    # Each pair is merged by the thread of the particle with the smaller index.
    # This thread is the only one that reads or writes the data of both particles.
    @threaded semi for particle in each_integrated_particle(system)
        is_merge_pair(merge_partner, particle) || return

        partner = merge_partner[particle]
        particle < partner || return

        m_a = hydrodynamic_mass(system, particle)
        m_b = hydrodynamic_mass(system, partner)
        m_merged = m_a + m_b

        # The position of `particle` relative to `partner`, across periodic boundaries.
        # Otherwise, a pair on opposite sides of a periodic box would be merged
        # into the middle of the box.
        pos_a = current_coords(u, system, particle)
        pos_b = current_coords(u, system, partner)
        pos_diff = periodic_pos_diff(pos_a - pos_b, periodic_box)
        distance = norm(pos_diff)

        # The center of mass (Eq. 32)
        position = pos_a - m_b / m_merged * pos_diff
        position = PointNeighbors.periodic_coords(position, periodic_box)

        # Smoothing length such that the kernel of the merged particle at its center
        # matches the kernels of both particles there (Eq. 34)
        h_a = smoothing_length(system, particle)
        h_b = smoothing_length(system, partner)
        w_a = m_a * kernel(smoothing_kernel, m_b / m_merged * distance, h_a)
        w_b = m_b * kernel(smoothing_kernel, m_a / m_merged * distance, h_b)
        h_merged = (m_merged * kernel_0 / (w_a + w_b))^(1 / ndims(system))

        # Conserve the momentum (Eq. 33)
        velocity = (m_a * current_velocity(v, system, particle) +
                    m_b * current_velocity(v, system, partner)) / m_merged

        # As in the code of Haftu et al., the pressure is averaged by mass and
        # the density of `particle` is kept.
        # TODO: Averaging the density by mass as well is an option.
        pressure = (m_a * current_pressure(v, system, particle) +
                    m_b * current_pressure(v, system, partner)) / m_merged

        set_particle_position!(u, system, particle, position)
        set_particle_velocity!(v, system, particle, velocity)
        set_particle_pressure!(v, system, particle, pressure)
        set_particle_mass!(system, particle, m_merged)
        set_particle_smoothing_length!(system, particle, h_merged)

        deactivate_particle!(system, partner, v, u)
    end

    return system
end

@inline periodic_pos_diff(pos_diff, ::Nothing) = pos_diff

@inline function periodic_pos_diff(pos_diff, periodic_box)
    return pos_diff - periodic_box.size .* round.(pos_diff ./ periodic_box.size)
end
