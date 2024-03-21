
mutable struct ParticleCoarsening{NDIMS, ELTYPE, R}
    candidates           :: Vector{Int}
    potential_candidates :: Vector{Int}
    ranges               :: R
    candidates_mass      :: Vector{ELTYPE}
    mass_center          :: Vector{ELTYPE}
    system_parent        :: System

    function ParticleCoarsening()
        return new{}()
    end
end

check_coarsening_criteria!(system, v_ode, u_ode, semi, t) = system

function check_coarsening_criteria!(system::FluidSystem, v_ode, u_ode, semi, t)
    check_coarsening_criteria!(system, system.particle_coarsening, v_ode, u_ode, semi, t)
end

check_coarsening_criteria!(system, ::Nothing, v_ode, u_ode, semi, t) = system

function check_coarsening_criteria!(system, particle_coarsening::ParticleCoarsening,
                                    v_ode, u_ode, semi, t)
    (; system_parent, potential_candidates, mass_center,
    candidates, candidates_mass) = particle_coarsening
    (; refinement_criteria) = system_parent.particle_refinement

    n_children = nchilds(system, system_parent.particle_refinement)

    v = wrap_v(v_ode, system, semi)
    u = wrap_u(u_ode, system, semi)

    Base.resize!(candidates, 0)
    Base.resize!(candidates_mass, 0)
    Base.resize!(mass_center, 0)
    Base.resize!(potential_candidates, 0)

    for particle in each_moving_particle(system)
        for refinement_criterion in refinement_criteria
            if (isempty(potential_candidates) || particle != last(potential_candidates)) &&
               !(refinement_criterion(system, particle, v, u, v_ode, u_ode, semi, t))

                # Add particle to potential candidates for coarsening
                push!(potential_candidates, particle)
            end
        end
    end

    nhs = get_neighborhood_search(system, semi)

    push!(candidates, zeros(Int, n_children))

    candidates_index = 1
    potential_candidates_dyn = copy(potential_candidates)

    for potential_particle in potential_candidates
        particle_coords = current_coords(u, system, potential_particle)

        # Filter the potential neighbor siblings (also contains `potential_particle`)
        # TODO: Benchmark the following options
        # filtered_candidates = collect(Iterators.filter(i -> i in potential_candidates_dyn,
        #                                                eachneighbor(particle_coords, nhs)))
        filtered_candidates = intersect(potential_candidates_dyn,
                                        eachneighbor(particle_coords, nhs))

        if length(filtered_candidates) >= n_children
            sibling_index = 0
            total_mass = zero(eltype(system))
            mass_center[candidates_index] .= zeros(eltype(system), ndims(system))

            for neighbor_sibling in filtered_candidates
                sibling_coords = current_coords(u, system, neighbor_sibling)

                pos_diff = particle_coords - sibling_coords
                distance2 = dot(pos_diff, pos_diff)

                if distance2 <= nhs.search_radius^2
                    sibling_index += 1
                    mass_neighbor = system.mass[neighbor_sibling]

                    candidates[candidates_index][sibling_index] = neighbor_sibling
                    mass_center[candidates_index] .+= mass_neighbor .* sibling_coords

                    total_mass += mass_neighbor
                end

                if sibling_index == n_children
                    # Remove already added candidates
                    setdiff!(potential_candidates_dyn, candidates[candidates_index])

                    # Add total mass
                    push!(candidates_mass, total_mass)

                    # Add center of mass
                    push!(mass_center, mass_center[candidates_index] ./ total_mass)

                    # Add another `Vector`` for potential candidates
                    push!(candidates, zeros(Int, n_children))
                    candidates_index += 1

                    break
                end
            end
        end
    end

    resize!(candidates, candidates_index)

    @boundscheck @assert length(candidates) == length(candidates_mass) ==
                         length(mass_center)

    return system
end

function coarsen_particles!(callback, semi, v_ode, u_ode, _v_cache, _u_cache)

    # Refine particles in all systems
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
    (; candidates, candidates_mass, system_parent) = particle_coarsening
end
