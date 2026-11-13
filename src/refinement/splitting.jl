"""
    split_particles!(semi, v_ode, u_ode)

Split all particles of fluid systems with particle refinement whose mass is larger than
`max_mass_factor` times the mass that corresponds to their target spacing
(Alg. 2 in Haftu et al.). The new particles are activated in free slots of the
`SystemBuffer`, which is resized when it is almost full.

!!! warning
    This can change the length of `v_ode` and `u_ode` (see `resize_systems!`).
"""
function split_particles!(semi, v_ode, u_ode)
    n_new_particles = zeros(Int, length(semi.systems))

    foreach_system(semi) do system
        n_new_particles[system_indices(system, semi)] = collect_split_candidates!(system,
                                                                                  u_ode,
                                                                                  semi)
    end

    sum(n_new_particles) == 0 && return semi

    capacities = [split_capacity(system, n_new_particles[i])
                  for (i, system) in enumerate(semi.systems)]
    resize_systems!(v_ode, u_ode, semi, capacities)

    foreach_system(semi) do system
        n_new_particles[system_indices(system, semi)] > 0 || return

        apply_splitting!(system, v_ode, u_ode, semi)
    end

    # The neighborhood searches have to find the new particles
    update_nhs!(semi, u_ode)

    return semi
end

# Return the number of new particles
@inline collect_split_candidates!(system, u_ode, semi) = 0

@inline function collect_split_candidates!(system::AbstractFluidSystem, u_ode, semi)
    return collect_split_candidates!(system, get_refinement(system), u_ode, semi)
end

@inline collect_split_candidates!(system, ::Nothing, u_ode, semi) = 0

function collect_split_candidates!(system, refinement, u_ode, semi)
    (; split_candidate) = system.cache
    (; max_mass_factor, splitting_pattern) = refinement

    set_zero!(split_candidate)

    @threaded semi for particle in each_integrated_particle(system)
        mass = hydrodynamic_mass(system, particle)
        split_candidate[particle] = mass >
                                    max_mass_factor * reference_mass(system, particle)
    end

    foreach_system(semi) do neighbor_system
        skip_splitting_near_boundary!(system, neighbor_system, u_ode, semi)
    end

    n_new_per_particle = n_split_particles(splitting_pattern, Val(ndims(system))) - 1

    return count(split_candidate) * n_new_per_particle
end

@inline skip_splitting_near_boundary!(system, neighbor_system, u_ode, semi) = system

# Do not split a particle if one of the new particles would be closer to a boundary particle
# than half of the boundary particle spacing. With dummy particles, the boundary surface is
# half a particle spacing in front of the boundary particles, so this prevents new particles
# on or behind the boundary surface.
# TODO: This rule is not safe. It measures the distance to the boundary particles,
# not to the boundary surface. Inside the boundary, a point between two neighboring
# boundary particles is `Δs_b / 2` away from both, and a point in the center of a grid cell
# of boundary particles is `Δs_b / √2` away from all corners. Both pass this check.
# The new particles are placed `epsilon * h` away from the particle, which can be much
# larger than `Δs_b` for coarse particles, so they can end up at exactly these positions.
# A safer rule would be a boundary fraction `φ(x) = Σ_b V_b W(x - x_b)` with `V_b = Δs_b^d`,
# which is about 0.5 at a flat boundary surface and about 1 inside the boundary.
# TODO: New particles in open boundary zones follow in the open boundary PR.
function skip_splitting_near_boundary!(system,
                                       neighbor_system::Union{WallBoundarySystem,
                                                              AbstractStructureSystem},
                                       u_ode, semi)
    (; split_candidate) = system.cache
    (; splitting_pattern) = get_refinement(system)
    (; epsilon) = splitting_pattern

    directions = split_directions(splitting_pattern, Val(ndims(system)))
    min_distance = neighbor_system.initial_condition.particle_spacing / 2

    u = wrap_u(u_ode, system, semi)
    u_neighbor = wrap_u(u_ode, neighbor_system, semi)
    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

    # Note that each thread only writes to its own `particle`
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(system)) do particle, _,
                                                                       pos_diff, _
        split_candidate[particle] || return

        # `pos_diff` is the position of `particle` relative to the boundary particle
        offset = epsilon * smoothing_length(system, particle)
        for direction in directions
            if norm(pos_diff + offset * direction) < min_distance
                split_candidate[particle] = false
            end
        end
    end

    return system
end

@inline split_capacity(system, n_new_particles) = nparticles(system)

# Grow the buffer when the new particles would fill more than 95% of it
function split_capacity(system::AbstractFluidSystem, n_new_particles)
    n_new_particles == 0 && return nparticles(system)

    (; buffer) = system
    if !(buffer isa SystemBuffer)
        throw(ArgumentError("splitting particles requires a `SystemBuffer`. " *
                            "Pass `buffer_size` to the fluid system."))
    end

    capacity = nparticles(system)
    n_required = buffer.active_particle_count[] + n_new_particles

    n_required > 0.95 * capacity || return capacity

    return max(round(Int, 1.25 * capacity), n_required + buffer.buffer_size)
end

function apply_splitting!(system, v_ode, u_ode, semi)
    (; splitting_pattern) = get_refinement(system)
    (; epsilon, alpha) = splitting_pattern
    (; split_candidate, target_spacing) = system.cache
    (; active_particle) = system.buffer

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    directions = split_directions(splitting_pattern, Val(ndims(system)))
    n_new_per_particle = length(directions)
    n_split = n_new_per_particle + 1

    # The new particles of the `i`-th candidate are activated in the free slots
    # `(i - 1) * n_new_per_particle + 1` to `i * n_new_per_particle`
    candidates = findall(split_candidate)
    free_slots = findall(!, active_particle)
    @assert length(free_slots) >= length(candidates) * n_new_per_particle

    @threaded semi for i in eachindex(candidates)
        particle = candidates[i]

        h = smoothing_length(system, particle)
        mass = hydrodynamic_mass(system, particle) / n_split
        position = current_coords(u, system, particle)
        velocity = current_velocity(v, system, particle)
        # TODO: The density of EDAC with `SummationDensity` is the one of the last
        # right-hand side evaluation. Check if this causes problems.
        density = current_density(v, system, particle)
        pressure = current_pressure(v, system, particle)
        spacing = target_spacing[particle]

        # The particle itself stays in the center (Vacondio et al. 2016)
        set_particle_mass!(system, particle, mass)
        set_particle_smoothing_length!(system, particle, alpha * h)

        # As in the code of Haftu et al., the new particles copy the properties
        # of the particle
        for j in 1:n_new_per_particle
            new_particle = free_slots[(i - 1) * n_new_per_particle + j]
            new_position = position + epsilon * h * directions[j]

            set_particle_position!(u, system, new_particle, new_position)
            set_particle_velocity!(v, system, new_particle, velocity)
            set_particle_density!(v, system, new_particle, density)
            set_particle_pressure!(v, system, new_particle, pressure)
            set_particle_mass!(system, new_particle, mass)
            set_particle_smoothing_length!(system, new_particle, alpha * h)
            target_spacing[new_particle] = spacing

            active_particle[new_particle] = true
        end
    end

    update_system_buffer!(system.buffer)
    set_zero!(split_candidate)

    return system
end
