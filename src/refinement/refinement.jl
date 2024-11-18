include("refinement_criteria.jl")

struct ParticleRefinement{SP, RC, ELTYPE}
    splitting_pattern         :: SP
    refinement_criteria       :: RC
    max_spacing_ratio         :: ELTYPE
    mass_ref                  :: Vector{ELTYPE} # length(mass_ref) == nparticles
    n_particles_before_resize :: Int
    n_new_particles           :: Int
end

function ParticleRefinement(; splitting_pattern, max_spacing_ratio,
                            refinement_criteria=SpatialRefinementCriterion())
    mass_ref = Vector{eltype(max_spacing_ratio)}()

    if refinement_criteria isa Tuple
        refinement_criteria = (refinement_criteria,)
    end

    return ParticleRefinement(splitting_pattern, refinement_criteria, max_spacing_ratio,
                              mass_ref, 0, 0)
end

resize_refinement!(system) = system

function resize_refinement!(system::FluidSystem)
    resize_refinement!(system, system.particle_refinement)
end

resize_refinement!(system, ::Nothing) = system

function resize_refinement!(system, particle_refinement)
    resize!(particle_refinement.mass_ref, nparticles(system))

    return system
end

function refinement!(semi, v_ode, u_ode, v_tmp, u_tmp, t)
    check_refinement_criteria!(semi, v_ode, u_ode)

    # Update the spacing of particles (Algorthm 1)
    update_particle_spacing(semi, u_ode)

    # Split the particles (Algorithm 2)
    split_particles!(semi, v_ode, u_ode, v_tmp, u_tmp)

    # Merge the particles (Algorithm 3)

    # Shift the particles

    # Correct the particles

    # Update smoothing lengths

    # Resize neighborhood search

    return semi
end

function update_particle_spacing(semi, u_ode)
    foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        update_particle_spacing(system, u, semi)
    end
end

# The methods for the `FluidSystem` are defined in `src/schemes/fluid/fluid.jl`
@inline update_particle_spacing(system, u, semi) = systemerror

@inline function update_particle_spacing(system::FluidSystem, u, semi)
    update_particle_spacing(system, system.particle_refinement, u, semi)
end

@inline update_particle_spacing(system, ::Nothing, u, semi) = system

@inline function update_particle_spacing(system::FluidSystem, particle_refinement, u, semi)
    (; smoothing_length, smoothing_length_factor) = system.cache
    (; mass_ref) = particle_refinement

    system_coords = current_coordinates(u, system)

    for particle in eachparticle(system)
        dp_min, dp_max, dp_avg = min_max_avg_spacing(system, semi, system_coords, particle)

        if dp_max / dp_min < max_spacing_ratio^3
            new_spacing = min(dp_max, max_spacing_ratio * dp_min)
        else
            new_spacing = dp_avg
        end

        smoothing_length[particle] = smoothing_length_factor * new_spacing
        mass_ref[particle] = system.density[particle] * new_spacing^(ndims(system))
    end

    return system
end

@inline function min_max_avg_spacing(system, semi, system_coords, particle)
    dp_min = Inf
    dp_max = zero(eltype(system))
    dp_avg = zero(eltype(system))
    counter_neighbors = 0

    foreach_system(semi) do neighbor_system
        neighborhood_search = get_neighborhood_search(particle_system, neighbor_system,
                                                      semi)

        u_neighbor_system = wrap_u(u_ode, neighbor, semi)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        PointNeighbors.foreach_neighbor(system_coords, neighbor_coords, neighborhood_search,
                                        particle) do particle, neighbor, pos_diff, distance
            dp_neighbor = particle_spacing(neighbor_system, neighbor)

            dp_min = min(dp_min, dp_neighbor)
            dp_max = max(dp_max, dp_neighbor)
            dp_avg += dp_neighbor

            counter_neighbors += 1
        end
    end

    dp_avg / counter_neighbors

    return dp_min, dp_max, dp_avg
end

@inline particle_spacing(system, particle) = system.initial_condition.particle_spacing

@inline function particle_spacing(system::FluidSystem, particle)
    return particle_spacing(system, system.particle_refinement, particle)
end

@inline particle_spacing(system, ::Nothing, _) = system.initial_condition.particle_spacing

@inline function particle_spacing(system, refinement, particle)
    (; smoothing_length_factor) = system.cache
    return smoothing_length(system, particle) / smoothing_length_factor
end
