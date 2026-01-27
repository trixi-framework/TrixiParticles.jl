# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`.
function PointNeighbors.foreach_point_neighbor(f, system, neighbor_system,
                                               system_coords, neighbor_coords, semi;
                                               points=eachparticle(system),
                                               parallelization_backend=semi.parallelization_backend)
    neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallelization_backend)
end

# For non-TLSPH systems, do nothing
function initialize_self_interaction_nhs(system, neighborhood_search,
                                         parallelization_backend)
    return system
end

function create_neighborhood_search(::Nothing, system, neighbor)
    nhs = TrivialNeighborhoodSearch{ndims(system)}()

    return create_neighborhood_search(nhs, system, neighbor)
end

# Avoid method ambiguity
function create_neighborhood_search(::Nothing, system::TotalLagrangianSPHSystem,
                                    neighbor::TotalLagrangianSPHSystem)
    nhs = TrivialNeighborhoodSearch{ndims(system)}()

    return create_neighborhood_search(nhs, system, neighbor)
end

function create_neighborhood_search(neighborhood_search, system, neighbor)
    return copy_neighborhood_search(neighborhood_search, compact_support(system, neighbor),
                                    nparticles(neighbor))
end

function create_neighborhood_search(neighborhood_search, system::TotalLagrangianSPHSystem,
                                    neighbor::TotalLagrangianSPHSystem)
    # TLSPH self-interaction is using a specialized neighborhood search
    return copy_neighborhood_search(neighborhood_search, zero(eltype(system)),
                                    nparticles(neighbor))
end

@inline function compact_support(system, neighbor)
    (; smoothing_kernel) = system
    # TODO: Variable search radius for NHS?
    return compact_support(smoothing_kernel, initial_smoothing_length(system))
end

@inline function compact_support(system::OpenBoundarySystem,
                                 neighbor::OpenBoundarySystem)
    # This NHS is never used
    return zero(eltype(system))
end

@inline function compact_support(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                                 neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    # Use the compact support of the fluid
    return compact_support(system.fluid_system, neighbor.fluid_system)
end

@inline function compact_support(system::BoundaryDEMSystem, neighbor::BoundaryDEMSystem)
    # This NHS is never used
    return zero(eltype(system))
end

@inline function compact_support(system::BoundaryDEMSystem, neighbor::DEMSystem)
    # Use the compact support of the DEMSystem
    return compact_support(neighbor, system)
end

@inline function compact_support(system::TotalLagrangianSPHSystem,
                                 neighbor::TotalLagrangianSPHSystem)
    (; smoothing_kernel, smoothing_length) = system
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function compact_support(system::Union{TotalLagrangianSPHSystem,
                                               WallBoundarySystem},
                                 neighbor)
    return compact_support(system, system.boundary_model, neighbor)
end

@inline function compact_support(system, model::BoundaryModelMonaghanKajtar, neighbor)
    # Use the compact support of the fluid for structure-fluid interaction
    return compact_support(neighbor, system)
end

@inline function compact_support(system, model::BoundaryModelMonaghanKajtar,
                                 neighbor::WallBoundarySystem)
    # This NHS is never used
    return zero(eltype(system))
end

@inline function compact_support(system, model::BoundaryModelDummyParticles, neighbor)
    # TODO: Monaghan-Kajtar BC are using the fluid's compact support for structure-fluid
    # interaction. Dummy particle BC use the model's compact support, which is also used
    # for density summations.
    (; smoothing_kernel, smoothing_length) = model
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function get_neighborhood_search(system, semi)
    (; neighborhood_searches) = semi

    system_index = system_indices(system, semi)

    return neighborhood_searches[system_index][system_index]
end

@inline function get_neighborhood_search(system::TotalLagrangianSPHSystem, semi)
    # For TLSPH, use the specialized self-interaction neighborhood search
    # for finding neighbors in the initial configuration.
    return system.self_interaction_nhs
end

@inline function get_neighborhood_search(system::TotalLagrangianSPHSystem,
                                         neighbor_system::TotalLagrangianSPHSystem, semi)
    (; neighborhood_searches) = semi

    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)

    if system_index == neighbor_index
        # For TLSPH, use the specialized self-interaction neighborhood search
        # for finding neighbors in the initial configuration.
        return system.self_interaction_nhs
    end

    return neighborhood_searches[system_index][neighbor_index]
end

@inline function get_neighborhood_search(system, neighbor_system, semi)
    (; neighborhood_searches) = semi

    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)

    return neighborhood_searches[system_index][neighbor_index]
end

function initialize_neighborhood_searches!(semi)
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            initialize_neighborhood_search!(semi, system, neighbor)
        end
    end

    return semi
end

function initialize_neighborhood_search!(semi, system, neighbor)
    # TODO Initialize after adapting to the GPU.
    # Currently, this cannot use `semi.parallelization_backend`
    # because data is still on the CPU.
    PointNeighbors.initialize!(get_neighborhood_search(system, neighbor, semi),
                               initial_coordinates(system),
                               initial_coordinates(neighbor),
                               eachindex_y=each_active_particle(neighbor),
                               parallelization_backend=PolyesterBackend())

    return semi
end

function initialize_neighborhood_search!(semi, system::TotalLagrangianSPHSystem,
                                         neighbor::TotalLagrangianSPHSystem)
    # For TLSPH, the self-interaction NHS is already initialized in the system constructor
    return semi
end

function update_nhs!(semi, u_ode)
    # Update NHS for each pair of systems
    foreach_system(semi) do system
        u_system = wrap_u(u_ode, system, semi)

        foreach_system(semi) do neighbor
            u_neighbor = wrap_u(u_ode, neighbor, semi)
            neighborhood_search = get_neighborhood_search(system, neighbor, semi)

            update_nhs!(neighborhood_search, system, neighbor, u_system, u_neighbor, semi)
        end
    end
end

# NHS updates
# To prevent hard-to-find bugs, there is no default version
function update_nhs!(neighborhood_search,
                     system::AbstractFluidSystem,
                     neighbor::Union{AbstractFluidSystem, TotalLagrangianSPHSystem},
                     u_system, u_neighbor, semi)
    # The current coordinates of fluids and structures change over time
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::Union{AbstractFluidSystem,
                                   OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang}},
                     neighbor::WallBoundarySystem,
                     u_system, u_neighbor, semi)
    # Boundary coordinates only change over time when `neighbor.ismoving[]`
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, neighbor.ismoving[]))
end

function update_nhs!(neighborhood_search,
                     system::AbstractFluidSystem, neighbor::OpenBoundarySystem,
                     u_system, u_neighbor, semi)
    # The current coordinates of fluids and open boundaries change over time.

    # TODO: Update only `active_coordinates` of open boundaries.
    # Problem: Removing inactive particles from neighboring lists is necessary.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::OpenBoundarySystem, neighbor::AbstractFluidSystem,
                     u_system, u_neighbor, semi)
    # The current coordinates of both open boundaries and fluids change over time.

    # TODO: Update only `active_coordinates` of open boundaries.
    # Problem: Removing inactive particles from neighboring lists is necessary.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                     neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                     u_system, u_neighbor, semi)
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                     neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem,
                     neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                     u_system, u_neighbor, semi)
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::OpenBoundarySystem, neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem, neighbor::OpenBoundarySystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem, neighbor::AbstractFluidSystem,
                     u_system, u_neighbor, semi)
    # The current coordinates of fluids and structured change over time
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem, neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    # TLSPH systems have their own self-interaction NHS.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem, neighbor::WallBoundarySystem,
                     u_system, u_neighbor, semi)
    # The current coordinates of structured change over time.
    # Boundary coordinates only change over time when `neighbor.ismoving[]`.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, neighbor.ismoving[]))
end

# This function is the same as the one below to avoid ambiguous dispatch when using `Union`
function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                     neighbor::AbstractFluidSystem, u_system, u_neighbor, semi)
    # Depending on the density calculator of the boundary model, this NHS is used for
    # - kernel summation (`SummationDensity`)
    # - continuity equation (`ContinuityDensity`)
    # - pressure extrapolation (`AdamiPressureExtrapolation`)
    #
    # Boundary coordinates only change over time when `neighbor.ismoving[]`.
    # The current coordinates of fluids and structured change over time.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(system.ismoving[], true),
            eachindex_y=each_active_particle(neighbor))
end

# This function is the same as the one above to avoid ambiguous dispatch when using `Union`
function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                     neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                     u_system, u_neighbor, semi)
    # Depending on the density calculator of the boundary model, this NHS is used for
    # - kernel summation (`SummationDensity`)
    # - continuity equation (`ContinuityDensity`)
    # - pressure extrapolation (`AdamiPressureExtrapolation`)
    #
    # Boundary coordinates only change over time when `neighbor.ismoving[]`.
    # The current coordinates of open boundaries change over time.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(system.ismoving[], true),
            eachindex_y=each_active_particle(neighbor))
end

# This function is the same as the one above to avoid ambiguous dispatch when using `Union`
function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                     neighbor::TotalLagrangianSPHSystem, u_system, u_neighbor, semi)
    # Depending on the density calculator of the boundary model, this NHS is used for
    # - kernel summation (`SummationDensity`)
    # - continuity equation (`ContinuityDensity`)
    # - pressure extrapolation (`AdamiPressureExtrapolation`)
    #
    # Boundary coordinates only change over time when `neighbor.ismoving[]`.
    # The current coordinates of fluids and structured change over time.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(system.ismoving[], true))
end

function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                     neighbor::WallBoundarySystem,
                     u_system, u_neighbor, semi)
    # `system` coordinates only change over time when `system.ismoving[]`.
    # `neighbor` coordinates only change over time when `neighbor.ismoving[]`.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(system.ismoving[], neighbor.ismoving[]))
end

function update_nhs!(neighborhood_search,
                     system::DEMSystem, neighbor::DEMSystem,
                     u_system, u_neighbor, semi)
    # Both coordinates change over time
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true))
end

function update_nhs!(neighborhood_search,
                     system::DEMSystem, neighbor::BoundaryDEMSystem,
                     u_system, u_neighbor, semi)
    # DEM coordinates change over time, the boundary coordinates don't
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, false))
end

function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem,
                     neighbor::AbstractFluidSystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::BoundaryDEMSystem,
                     neighbor::Union{DEMSystem, BoundaryDEMSystem},
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::Union{WallBoundarySystem, OpenBoundarySystem},
                     neighbor::Union{WallBoundarySystem, OpenBoundarySystem},
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

# Forward to PointNeighbors.jl
function update!(neighborhood_search, x, y, semi; points_moving=(true, true),
                 eachindex_y=axes(y, 2))
    PointNeighbors.update!(neighborhood_search, x, y; points_moving, eachindex_y,
                           parallelization_backend=semi.parallelization_backend)
end
