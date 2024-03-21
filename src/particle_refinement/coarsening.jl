
struct ParticleCoarsening{NDIMS, ELTYPE}
    candidates           :: Vector{Int}
    potential_candidates :: Vector{Int}
    ranges               :: Vector{UnitRange}
    candidates_mass      :: Vector{ELTYPE}
    system_parent        :: System

    function ParticleCoarsening(system_parent)
        return new{ndims(system_parent),
                   eltype(system_parent)}([], [], [], [], system_parent)
    end
end

check_coarsening_criteria!(system, v_ode, u_ode, semi, t) = system

function check_coarsening_criteria!(system::FluidSystem, v_ode, u_ode, semi, t)
    check_coarsening_criteria!(system, system.particle_coarsening, v_ode, u_ode, semi, t)
end

check_coarsening_criteria!(system, ::Nothing, v_ode, u_ode, semi, t) = system

function check_coarsening_criteria!(system, particle_coarsening::ParticleCoarsening,
                                    v_ode, u_ode, semi, t)
    (; system_parent, potential_candidates, ranges,
    candidates, candidates_mass) = particle_coarsening
    (; refinement_criteria) = system_parent.particle_refinement

    # TODO
    search_radius = 1.9 * system.initial_condition.particle_spacing

    n_children = nchilds(system, system_parent.particle_refinement)

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    Base.resize!(candidates, 0)
    Base.resize!(candidates_mass, 0)
    Base.resize!(potential_candidates, 0)
    Base.resize!(ranges, 0)

    # Search for potential merge candidates
    for particle in each_moving_particle(system)
        for refinement_criterion in refinement_criteria
            if (isempty(potential_candidates) || particle != last(potential_candidates)) &&
               !(refinement_criterion(system, particle, v, u, v_ode, u_ode, semi, t))

                # Add potential candidates for coarsening
                push!(potential_candidates, particle)
            end
        end
    end

    parent_index = 1
    remaining_candidates = copy(potential_candidates)

    nhs = get_neighborhood_search(system, semi)

    # Check each potential merge candidate and add them to `candidates`
    # if surround by sufficient neighboring merge candidates
    for potential_particle in potential_candidates
        particle_coords = current_coords(u, system, potential_particle)

        # Filter the potential neighboring siblings (also contains `potential_particle`)
        # TODO: Benchmark the following options
        # filtered_candidates = collect(Iterators.filter(i -> i in remaining_candidates,
        #                                                eachneighbor(particle_coords, nhs)))
        filtered_candidates = intersect(remaining_candidates,
                                        eachneighbor(particle_coords, nhs))

        # Check if number of neighboring siblings is sufficient
        if length(filtered_candidates) >= n_children
            n_neighbor_sibling = 0

            for neighbor_sibling in filtered_candidates
                sibling_coords = current_coords(u, system, neighbor_sibling)

                pos_diff = particle_coords - sibling_coords
                distance2 = dot(pos_diff, pos_diff)

                if distance2 <= search_radius^2
                    n_neighbor_sibling += 1

                    # Add sibling
                    push!(candidates, neighbor_sibling)
                    push!(candidates_mass, system.mass[neighbor_sibling])
                end

                if n_neighbor_sibling == n_children
                    # Add ranges for one parent particle
                    start_iter = (parent_index - 1) * n_children + 1
                    end_iter = start_iter + n_children - 1

                    push!(ranges, start_iter:end_iter)

                    # Remove already added siblings
                    setdiff!(remaining_candidates, candidates[ranges[parent_index]])

                    parent_index += 1
                    break
                end
            end

            if n_neighbor_sibling < n_children
                # Not enough neighbor siblings. Remove all other neighboring siblings again.
                capacity = length(candidates) - n_neighbor_sibling
                resize!(candidates, capacity)
                resize!(candidates_mass, capacity)
            end
        end
    end

    if !isempty(candidates)
        @boundscheck @assert length(candidates) == length(candidates_mass) ==
                             ranges[end][end]
    end

    return system
end

function coarsen_particles!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)

    # Coarsen particles in all systems
    foreach_system(semi) do system
        coarsen_particles!(system, v_ode, u_ode, _v_cache, _u_cache, callback, semi)
    end
end

coarsen_particles!(system, v_ode, u_ode, _v_cache, _u_cache, callback, semi) = system

function coarsen_particles!(system::FluidSystem, v_ode, u_ode, _v_cache, _u_cache,
                            callback, semi)
    coarsen_particles!(system, system.particle_coarsening, v_ode, u_ode, _v_cache, _u_cache,
                       callback, semi)
end

function coarsen_particles!(system::FluidSystem, ::Nothing, v_ode, u_ode, _v_cache,
                            _u_cache,
                            callback, semi)
    return system
end

function coarsen_particles!(system_child::FluidSystem,
                            particle_coarsening::ParticleCoarsening,
                            v_ode, u_ode, _v_cache, _u_cache, callback, semi)
    (; candidates, candidates_mass, system_parent, ranges) = particle_coarsening

    if !isempty(candidates)
        # Old storage
        v_child = _wrap_v(_v_cache, system_child, semi, callback)
        u_child = _wrap_u(_u_cache, system_child, semi, callback)

        # Resized storage
        v_parent = wrap_v(v_ode, system_parent, semi)
        u_parent = wrap_u(u_ode, system_parent, semi)

        current_n_parents = nparticles(system_parent) - length(ranges)

        for parent_index in eachindex(ranges)
            absolute_index = current_n_parents + parent_index

            # Set zero
            for dim in 1:ndims(system_parent)
                u_parent[dim, absolute_index] = zero(eltype(system_parent))
                v_parent[dim, absolute_index] = zero(eltype(system_parent))
            end

            total_mass = zero(eltype(system_parent))

            # Loop over all merge candidates and find the center of mass
            child_mass_index = 1
            for child in candidates[ranges[parent_index]]
                child_position = current_coords(u_child, system_child, child)

                mass_child = candidates_mass[child_mass_index]

                total_mass += mass_child

                # calculate center of mass
                for dim in 1:ndims(system_child)
                    u_parent[dim, absolute_index] += mass_child * child_position[dim]
                end

                child_mass_index += 1
            end

            for dim in 1:ndims(system_child)
                u_parent[dim, absolute_index] /= total_mass
            end

            parent_position = current_coords(u_parent, system_parent, absolute_index)

            rho_a = zero(eltype(system_parent))
            p_a = zero(eltype(system_parent))
            volume = zero(eltype(system_parent))

            # Interpolate child quantities
            for child in candidates[ranges[parent_index]]
                child_position = current_coords(u_child, system_child, child)

                pos_diff = parent_position - child_position

                kernel_weight = smoothing_kernel(system_child, norm(pos_diff))

                volume += kernel_weight

                v_b = current_velocity(v_child, system_child, child)
                p_b = particle_pressure_parent(v_child, system_child, child)
                rho_b = particle_density_parent(v_child, system_child, child)

                for dim in 1:ndims(system_child)
                    v_parent[dim, absolute_index] += kernel_weight * v_b[dim]
                end

                rho_a += kernel_weight * rho_b
                p_a += kernel_weight * p_b
            end

            for dim in 1:ndims(system_child)
                v_parent[dim, absolute_index] /= volume
            end

            rho_a /= volume
            p_a /= volume

            set_particle_density(absolute_index, v_parent, system_parent.density_calculator,
                                 system_parent, rho_a)
            set_particle_pressure(absolute_index, v_parent, system_parent, p_a)
        end

        resize!(candidates, 0)
        resize!(ranges, 0)
    end
end
