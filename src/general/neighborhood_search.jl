# === PointNeighbors integration ===
# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`. If the semidiscretization disables this
# ordered system pair via `has_system_interaction`, the traversal is skipped entirely.
function PointNeighbors.foreach_point_neighbor(f, system, neighbor_system,
                                               system_coords, neighbor_coords, semi;
                                               points=eachparticle(system),
                                               parallelization_backend=semi.parallelization_backend)
    has_system_interaction(system, neighbor_system, semi) || return nothing

    neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallelization_backend)
end

deactivate_out_of_bounds_particles!(system, buffer, nhs, v, u, semi) = system

function deactivate_out_of_bounds_particles!(system, buffer::SystemBuffer,
                                             nhs::GridNeighborhoodSearch, v, u, semi)
    deactivate_out_of_bounds_particles!(system, buffer, nhs, nhs.cell_list, v, u, semi)
end

function deactivate_out_of_bounds_particles!(system, buffer, nhs, cell_list, v, u, semi)
    return system
end

# `GridNeighborhoodSearch` with a `FullGridCellList` requires a bounding box.
# This function deactivates particles that move outside the bounding box to prevent
# simulation crashes.
# Note that deactivating particles is only possible in combination with a 'SystemBuffer'.
function deactivate_out_of_bounds_particles!(system, ::SystemBuffer, nhs,
                                             cell_list::FullGridCellList, v, u, semi)
    @trixi_timeit timer() "deactivate out of bounds particle" begin
        @threaded semi for particle in each_integrated_particle(system)
            particle_position = current_coords(u, system, particle)
            cell = PointNeighbors.cell_coords(particle_position, nhs)

            # This is the same code as is used in PointNeighbors.jl in `check_cell_bounds`.
            # It tests that particles are inside the inner grid (without the padding layer for neighbor query).
            if !all(cell[i] in 2:(size(cell_list.linear_indices, i) - 1)
                    for i in eachindex(cell))
                deactivate_particle!(system, particle, v, u)
            end
        end

        if count(system.buffer.active_particle) != system.buffer.active_particle_count[]
            update_system_buffer!(system.buffer)
        end
    end

    return system
end

@propagate_inbounds function foreach_neighbor(f, system_coords, neighbor_coords,
                                              neighborhood_search, backend, particle)
    PointNeighbors.foreach_neighbor(f, system_coords, neighbor_coords,
                                    neighborhood_search, particle)
end

# We cannot dispatch by `AbstractGPUArray` because this is called from within
# a kernel, where the arrays are device arrays (like `CuDeviceArray`),
# which are not `AbstractGPUArray`s.
@inline function foreach_neighbor(f, system_coords, neighbor_coords, neighborhood_search,
                                  backend::KernelAbstractions.GPU, particle)
    # On GPUs, remove all bounds checks for maximum performance.
    # Note that this is not safe if the neighborhood search was not initialized correctly.
    # For example, this is unsafe when benchmarking `interact!` with the wrong NHS.
    PointNeighbors.foreach_neighbor_unsafe(f, system_coords, neighbor_coords,
                                           neighborhood_search, particle)
end

# === Compact support selection ===
# -- Generic
@inline function compact_support(system, neighbor)
    (; smoothing_kernel) = system
    # TODO: Variable search radius for NHS?
    return compact_support(smoothing_kernel, initial_smoothing_length(system))
end

# -- Open boundary systems
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

@inline function compact_support(system::OpenBoundarySystem, neighbor::RigidBodySystem)
    # Rigid/open-boundary interactions are currently not modeled.
    return zero(eltype(system))
end

# -- DEM boundaries
@inline function compact_support(system::BoundaryDEMSystem, neighbor::BoundaryDEMSystem)
    # This NHS is never used
    return zero(eltype(system))
end

@inline function compact_support(system::BoundaryDEMSystem, neighbor::DEMSystem)
    # Use the compact support of the DEMSystem
    return compact_support(neighbor, system)
end

# -- TLSPH systems
@inline function compact_support(system::TotalLagrangianSPHSystem,
                                 neighbor::TotalLagrangianSPHSystem)
    (; smoothing_kernel, smoothing_length) = system
    return compact_support(smoothing_kernel, smoothing_length)
end

# -- Boundary models
@inline function compact_support(system::Union{TotalLagrangianSPHSystem,
                                               RigidBodySystem,
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

@inline function compact_support(system::WallBoundarySystem,
                                 model::BoundaryModelDummyParticles,
                                 neighbor::RigidBodySystem{Nothing})
    # Contact-only rigid bodies do not participate in wall-side hydrodynamic passes such as
    # density summation, pressure extrapolation, or correction assembly. Keep the reverse
    # wall->rigid search radius at zero so those updates never query rigid hydrodynamic data.
    return zero(eltype(system))
end

@inline function compact_support(system::RigidBodySystem, ::Nothing, neighbor)
    # Fallback for `compact_support(system, system.boundary_model, neighbor)` in the
    # boundary-model-based path used by rigid-fluid interaction.
    return zero(eltype(system))
end

@inline function compact_support(system::RigidBodySystem,
                                 neighbor::WallBoundarySystem)
    # Rigid-wall contact depends on the rigid contact model, not on the hydrodynamic
    # boundary model used for fluid-structure interaction.
    return compact_support(system, system.contact_model, neighbor)
end

@inline function compact_support(system::RigidBodySystem, contact_model::Nothing,
                                 neighbor::WallBoundarySystem)
    return zero(eltype(system))
end

@inline function compact_support(system::RigidBodySystem,
                                 contact_model::RigidContactModel,
                                 neighbor::WallBoundarySystem)
    return contact_model.contact_distance
end

@inline function compact_support(system::RigidBodySystem, neighbor::RigidBodySystem)
    return compact_support(system.contact_model, system, neighbor.contact_model, neighbor)
end

@inline function compact_support(contact_model, system::RigidBodySystem,
                                 contact_model_neighbor, neighbor::RigidBodySystem)
    return zero(eltype(system))
end

@inline function compact_support(contact_model::RigidContactModel,
                                 system::RigidBodySystem,
                                 neighbor_contact_model::RigidContactModel,
                                 neighbor::RigidBodySystem)
    return max(contact_model.contact_distance, neighbor_contact_model.contact_distance)
end

@inline function compact_support(system::RigidBodySystem, neighbor::OpenBoundarySystem)
    # Rigid/open-boundary interactions are currently not modeled.
    return zero(eltype(system))
end

# === Neighborhood search creation ===
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

# === Neighborhood search lookup ===
@inline function get_neighborhood_search(system, semi)
    return get_neighborhood_search(system, semi, system_indices(system, semi))
end

@inline function get_neighborhood_search(system, neighbor_system, semi)
    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)

    return get_neighborhood_search(system, neighbor_system, semi, system_index,
                                   neighbor_index)
end

@inline function get_neighborhood_search(system, semi, system_index::Integer)
    return semi.neighborhood_searches[system_index, system_index]
end

@inline function get_neighborhood_search(system::TotalLagrangianSPHSystem, semi,
                                         system_index::Integer)
    # For TLSPH, use the specialized self-interaction neighborhood search
    # for finding neighbors in the initial configuration.
    return system.self_interaction_nhs
end

@inline function get_neighborhood_search(system, neighbor_system, semi,
                                         system_index::Integer, neighbor_index::Integer)
    return semi.neighborhood_searches[system_index, neighbor_index]
end

@inline function get_neighborhood_search(system::TotalLagrangianSPHSystem,
                                         neighbor_system::TotalLagrangianSPHSystem, semi,
                                         system_index::Integer, neighbor_index::Integer)
    if system_index == neighbor_index
        # For TLSPH, use the specialized self-interaction neighborhood search
        # for finding neighbors in the initial configuration.
        return system.self_interaction_nhs
    end

    return semi.neighborhood_searches[system_index, neighbor_index]
end

# === Initialization ===
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

# For non-TLSPH systems, do nothing
function initialize_self_interaction_nhs(system, neighborhood_search,
                                         parallelization_backend)
    return system
end

# === Neighborhood search updates (per-system) ===
function update_nhs!(semi, u_ode)
    # Update NHS for each enabled ordered pair of systems.
    foreach_system_indexed(semi) do system_index, system
        u_system = wrap_u(u_ode, system, semi, system_index)

        foreach_system_indexed(semi) do neighbor_index, neighbor
            has_system_interaction(semi, system_index, neighbor_index) || return

            u_neighbor = wrap_u(u_ode, neighbor, semi, neighbor_index)
            neighborhood_search = get_neighborhood_search(system, neighbor, semi,
                                                          system_index, neighbor_index)

            update_nhs!(neighborhood_search, system, neighbor, u_system, u_neighbor, semi)
        end
    end
end

# === Neighborhood search updates (per-pair dispatch) ===
# To prevent hard-to-find bugs, there is no default version
# -- Fluid / structure interactions
function update_nhs!(neighborhood_search,
                     system::AbstractFluidSystem,
                     neighbor::Union{AbstractFluidSystem, TotalLagrangianSPHSystem,
                                     RigidBodySystem},
                     u_system, u_neighbor, semi)
    # The current coordinates of fluids and structures change over time
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

# -- Fluid / wall interactions
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

# -- Open boundary interactions
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
                     system::RigidBodySystem,
                     neighbor::OpenBoundarySystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

# -- Open boundary combinations that are never used
function update_nhs!(neighborhood_search,
                     system::OpenBoundarySystem,
                     neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem,
                     neighbor::OpenBoundarySystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

# -- TLSPH interactions
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
                     system::TotalLagrangianSPHSystem, neighbor::RigidBodySystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
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

function update_nhs!(neighborhood_search,
                     system::TotalLagrangianSPHSystem, neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    # TLSPH systems have their own self-interaction NHS.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::RigidBodySystem,
                     neighbor::Union{AbstractFluidSystem, RigidBodySystem},
                     u_system, u_neighbor, semi)
    # The current coordinates of fluids and structures change over time.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, true), eachindex_y=each_active_particle(neighbor))
end

function update_nhs!(neighborhood_search,
                     system::RigidBodySystem, neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

function update_nhs!(neighborhood_search,
                     system::RigidBodySystem, neighbor::WallBoundarySystem,
                     u_system, u_neighbor, semi)
    # The current coordinates of structures change over time.
    # Boundary coordinates only change over time when `neighbor.ismoving[]`.
    update!(neighborhood_search,
            current_coordinates(u_system, system),
            current_coordinates(u_neighbor, neighbor),
            semi, points_moving=(true, neighbor.ismoving[]))
end

# -- Wall dummy particle interactions
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
                     neighbor::TotalLagrangianSPHSystem,
                     u_system, u_neighbor, semi)
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

# Rigid-wall contact is only computed from the rigid system side. `WallBoundarySystem`
# does not actively initiate rigid interactions, so keep the reverse-direction NHS idle.
# Explicitly define this method to avoid ambiguity with the generic no-op method below.
function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                     neighbor::RigidBodySystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

# -- Wall / wall interactions
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

# -- DEM interactions
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

# -- Combinations that are never used
function update_nhs!(neighborhood_search,
                     system::WallBoundarySystem,
                     neighbor::Union{AbstractFluidSystem, RigidBodySystem},
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

function update_nhs!(neighborhood_search,
                     system::OpenBoundarySystem,
                     neighbor::RigidBodySystem,
                     u_system, u_neighbor, semi)
    # Don't update. This NHS is never used.
    return neighborhood_search
end

# === PointNeighbors forwarding ===
function update!(neighborhood_search, x, y, semi; points_moving=(true, true),
                 eachindex_y=axes(y, 2))
    PointNeighbors.update!(neighborhood_search, x, y; points_moving, eachindex_y,
                           parallelization_backend=semi.parallelization_backend)
end
