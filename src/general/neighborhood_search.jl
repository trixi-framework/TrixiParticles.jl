# === PointNeighbors integration ===
# Loop over all pairs of particles and neighbors within the kernel cutoff.
# `f(particle, neighbor, pos_diff, distance)` is called for every particle-neighbor pair.
# By default, loop over `eachparticle(system)`.
@inline function PointNeighbors.foreach_point_neighbor(f, system::AbstractSystem,
                                                       neighbor_system::AbstractSystem,
                                                       system_coords, neighbor_coords, semi;
                                                       points=eachparticle(system),
                                                       parallelization_backend=semi.parallelization_backend)
    neighborhood_search = get_neighborhood_search(system, neighbor_system, semi)
    foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;
                           points, parallelization_backend)
end

# Return the `PeriodicBox` of a neighborhood search, or `nothing` for a non-periodic one.
# `neighborhood_search = nothing` creates a `TrivialNeighborhoodSearch` without periodicity.
extract_periodic_box(::Nothing) = nothing
extract_periodic_box(nhs) = nhs.periodic_box

# The `CombinedNHSHandler` returns a `GridNHSPair`.
# Both searches of the pair are created from the same template with the same periodic box.
function extract_periodic_box(nhs::PointNeighbors.GridNHSPair)
    return extract_periodic_box(nhs.neighbor_search)
end

deactivate_out_of_bounds_particles!(system, buffer, nhs, v, u, semi) = system

function deactivate_out_of_bounds_particles!(system, buffer::SystemBuffer,
                                             nhs::GridNeighborhoodSearch, v, u, semi)
    deactivate_out_of_bounds_particles!(system, buffer, nhs, nhs.cell_list, v, u, semi)
end

function deactivate_out_of_bounds_particles!(system, buffer::SystemBuffer,
                                             nhs::PointNeighbors.GridNHSPair, v, u, semi)
    deactivate_out_of_bounds_particles!(system, buffer, nhs.neighbor_search, v, u, semi)
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

@propagate_inbounds function mapreduce_neighbor(f, op, system_coords, neighbor_coords,
                                                neighborhood_search, backend, particle;
                                                init)
    PointNeighbors.mapreduce_neighbor(f, op, system_coords, neighbor_coords,
                                      neighborhood_search, particle; init)
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

@inline function mapreduce_neighbor(f, op, system_coords, neighbor_coords,
                                    neighborhood_search,
                                    backend::KernelAbstractions.GPU, particle; init)
    # On GPUs, remove all bounds checks for maximum performance.
    # Note that this is not safe if the neighborhood search was not initialized correctly.
    # For example, this is unsafe when benchmarking `interact!` with the wrong NHS.
    PointNeighbors.mapreduce_neighbor_unsafe(f, op, system_coords, neighbor_coords,
                                             neighborhood_search, particle; init)
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

# === Neighborhood search handlers ===
# Handlers manage how neighborhood searches are stored and retrieved for
# different interacting systems.
abstract type AbstractNHSHandler end

function create_neighborhood_search_handler(::Type{Handler}, neighborhood_search,
                                            systems) where {Handler <: AbstractNHSHandler}
    return Handler(neighborhood_search, systems)
end

function create_neighborhood_search_handler(::Type{Handler}, neighborhood_search::Nothing,
                                            systems) where {Handler <: AbstractNHSHandler}
    # `neighborhood_search = nothing` creates a `TrivialNeighborhoodSearch`.
    nhs = TrivialNeighborhoodSearch{ndims(first(systems))}()
    return Handler(nhs, systems)
end

function create_neighborhood_search_handler(handler, neighborhood_search, systems)
    throw(ArgumentError("`neighborhood_search_handler` must be a handler type, " *
                        "for example `PairsNHSHandler` or `SharedNHSHandler`."))
end

# `neighborhood_search_handler = nothing` selects the default handler,
# see `default_neighborhood_search_handler`.
function create_neighborhood_search_handler(::Nothing, neighborhood_search, systems)
    handler = default_neighborhood_search_handler(neighborhood_search, systems)
    return create_neighborhood_search_handler(handler, neighborhood_search, systems)
end

# `neighborhood_search = nothing` creates a `TrivialNeighborhoodSearch`.
function default_neighborhood_search_handler(::Union{Nothing, TrivialNeighborhoodSearch},
                                             systems)
    # Prefer the `PairsNHSHandler` for `TrivialNeighborhoodSearch` because it allows for
    # infinite search radius, which the `SharedNHSHandler` does not support.
    return PairsNHSHandler
end

function default_neighborhood_search_handler(neighborhood_search::GridNeighborhoodSearch,
                                             systems)
    # Use the `CombinedNHSHandler` when all systems interact with the same search radius.
    # This enables the full performance optimizations of the neighborhood search,
    # since the particles of all systems are stored in the same grid.
    if !isnothing(uniform_search_radius(systems))
        return CombinedNHSHandler
    end

    return default_neighborhood_search_handler_(neighborhood_search)
end

function default_neighborhood_search_handler(neighborhood_search::AbstractNeighborhoodSearch,
                                             systems)
    return default_neighborhood_search_handler_(neighborhood_search)
end

function default_neighborhood_search_handler_(neighborhood_search)
    # If the neighborhood search does not require updates when the first system moves,
    # we can query neighbors of arbitrary particles without updating the neighborhood search.
    # In this case, we can use a single neighborhood search per neighbor system,
    # instead of one per pair of systems, which all store the same information.
    if !PointNeighbors.requires_update(neighborhood_search)[1]
        return SharedNHSHandler
    end

    return PairsNHSHandler
end

# Return the search radius if all pairs of systems use the same search radius
# and `nothing` otherwise.
function uniform_search_radius(systems)
    # A compact support of zero means that this neighborhood search is never used.
    # TLSPH self-interaction is using a specialized neighborhood search.
    search_radii = [compact_support(system, neighbor)
                    for system in systems, neighbor in systems
                    if !(system === neighbor && system isa TotalLagrangianSPHSystem)]
    isempty(search_radii) && return nothing

    search_radius = maximum(search_radii)
    if iszero(search_radius) ||
       !all(radius -> iszero(radius) || isapprox(radius, search_radius), search_radii)
        # The search radius is zero or there are pairs of systems that require different
        # non-zero search radii.
        return nothing
    end

    return search_radius
end

first_neighborhood_search(searches::AbstractMatrix) = first(searches)
first_neighborhood_search(searches) = first(first(searches))

function neighborhood_search_name(handler::AbstractNHSHandler)
    search = first_neighborhood_search(handler.neighborhood_searches)

    return nameof(typeof(search))
end

"""
    PairsNHSHandler

Neighborhood search handler that stores one neighborhood search for each ordered
pair of systems.

`PairsNHSHandler` is the fully generic handler. Use it for neighborhood search
implementations whose data are tied to the specific pair of systems used during
an update, e.g., [`PrecomputedNeighborhoodSearch`](@ref).
It is compatible with all neighborhood search implementations and is
the default for searches that cannot use [`SharedNHSHandler`](@ref).

# Examples
```jldoctest semi_example; output=false, setup = :(using TrixiParticles; trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); system1 = fluid_system; system2 = boundary_system)
semi = Semidiscretization(system1, system2;
                          neighborhood_search_handler=PairsNHSHandler)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Semidiscretization                                                                               │
│ ══════════════════                                                                               │
│ #spatial dimensions: ………………………… 2                                                                │
│ #systems: ……………………………………………………… 2                                                                │
│ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
│ total #particles: ………………………………… 636                                                              │
│ eltype: …………………………………………………………… Float64                                                          │
│ coordinates eltype: …………………………… Float64                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct PairsNHSHandler{NHS} <: AbstractNHSHandler
    neighborhood_searches::NHS
end

function PairsNHSHandler(neighborhood_search::AbstractNeighborhoodSearch, systems)
    searches = [create_neighborhood_search(neighborhood_search, system, neighbor)
                for system in systems, neighbor in systems]

    @assert isconcretetype(eltype(searches)) "neighborhood searches are not type-stable"

    return PairsNHSHandler(searches)
end

function get_neighborhood_search(handler::PairsNHSHandler, system_index, neighbor_index,
                                 search_radius)
    return handler.neighborhood_searches[system_index, neighbor_index]
end

"""
    SharedNHSHandler

Neighborhood search handler optimized for reusable neighborhood searches.

`SharedNHSHandler` stores one neighborhood search for each neighbor system and
required search radius, instead of one search for every ordered pair of systems.
For a query involving `(system, neighbor)`, it selects the stored search for the
neighbor system with a radius large enough for that pair's compact support.

This handler reuses the same neighborhood search for all interactions
that have the same neighbor system and search radius. This is only possible if the
neighborhood search stores information about the neighbor particles only.
For example, a grid-based search can be shared because it stores the neighbor
particles in grid cells and it does not store information about the first particle set.
In contrast, a precomputed neighbor list cannot be shared because its
stored neighbors are tied to the specific query particles used during initialization.

The `SharedNHSHandler` is the default handler for compatible neighborhood search
implementations because it avoids storing and updating redundant neighborhood searches.
For a [`GridNeighborhoodSearch`](@ref) where all pairs of systems use the same
search radius, a more efficient internal handler is chosen by default instead
(see [neighborhood search performance](@ref neighborhood_search_performance)).

!!! warning
    This handler is only compatible with neighborhood searches that can query
    neighbors of arbitrary points in space after an update,
    e.g. [`GridNeighborhoodSearch`](@ref) and [`TrivialNeighborhoodSearch`](@ref).
    [`PrecomputedNeighborhoodSearch`](@ref) is not compatible because it can only
    query neighbors for the particles that were used to update the search.

# Examples
```jldoctest semi_example; output=false, setup = :(using TrixiParticles; trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); system1 = fluid_system; system2 = boundary_system)
semi = Semidiscretization(system1, system2;
                          neighborhood_search=GridNeighborhoodSearch{2}(),
                          neighborhood_search_handler=SharedNHSHandler)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Semidiscretization                                                                               │
│ ══════════════════                                                                               │
│ #spatial dimensions: ………………………… 2                                                                │
│ #systems: ……………………………………………………… 2                                                                │
│ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
│ total #particles: ………………………………… 636                                                              │
│ eltype: …………………………………………………………… Float64                                                          │
│ coordinates eltype: …………………………… Float64                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct SharedNHSHandler{SR, NHS} <: AbstractNHSHandler
    search_radii::SR
    neighborhood_searches::NHS
end

function SharedNHSHandler(neighborhood_search::AbstractNeighborhoodSearch, systems)
    if PointNeighbors.requires_update(neighborhood_search)[1]
        throw(ArgumentError("`SharedNHSHandler` is not compatible with neighborhood search " *
                            "implementations that require updates when the first system moves, " *
                            "such as `PrecomputedNeighborhoodSearch`"))
    end

    search_radii = [sort(unique(compact_support(system, neighbor)
                                for system in systems))
                    for neighbor in systems]

    searches = [[copy_neighborhood_search(neighborhood_search, search_radius,
                                          nparticles(neighbor))
                 for search_radius in search_radii[neighbor_index]]
                for (neighbor_index, neighbor) in pairs(systems)]

    searches_are_concrete = isconcretetype(eltype(searches)) &&
                            all(row -> isconcretetype(eltype(row)), searches)
    @assert searches_are_concrete "neighborhood searches are not type-stable"

    return SharedNHSHandler(search_radii, searches)
end

function get_neighborhood_search(handler::SharedNHSHandler, system_index, neighbor_index,
                                 search_radius)
    radii = handler.search_radii[neighbor_index]

    # Subtract `eps` to avoid selecting the next larger radius due to rounding errors.
    radius_index = searchsortedfirst(radii, search_radius - eps(search_radius))

    @boundscheck if radius_index > length(radii)
        throw(ArgumentError("no shared neighborhood search with radius >= $search_radius"))
    end

    return handler.neighborhood_searches[neighbor_index][radius_index]
end

# Neighborhood search handler that stores the particles of all systems in the same grid.
#
# `CombinedNHSHandler` wraps a `CombinedGridNeighborhoodSearch`, which stores one cell list
# per system, but all cell lists use the same grid with the same search radius
# (see `PointNeighbors.CombinedGridNeighborhoodSearch`).
# The cell list of each system is updated only once per update of the neighborhood searches.
# Because the particles of both systems of an interaction are stored in the same grid,
# neighbor loops can traverse the particles of both systems cell by cell, which allows for
# advanced performance optimizations and SIMD vectorization.
#
# This handler is not public API. It is selected automatically when no handler is passed
# to the `Semidiscretization`, a `GridNeighborhoodSearch` is used, and all pairs of
# systems interact with the same search radius (see `default_neighborhood_search_handler`).
struct CombinedNHSHandler{NHS} <: AbstractNHSHandler
    neighborhood_search::NHS
    # Buffer for `update_nhs!` to store which cell lists have to be updated.
    systems_moving::Vector{Bool}
end

function CombinedNHSHandler(neighborhood_search::GridNeighborhoodSearch, systems)
    search_radius = uniform_search_radius(systems)

    if isnothing(search_radius)
        search_radii = unique(compact_support(system, neighbor)
                              for system in systems, neighbor in systems)
        throw(ArgumentError("`CombinedNHSHandler` requires the same nonzero search " *
                            "radius for all pairs of systems, but got the search radii " *
                            "$search_radii"))
    end

    combined_nhs = PointNeighbors.CombinedGridNeighborhoodSearch(neighborhood_search,
                                                                 search_radius,
                                                                 map(nparticles, systems))

    return CombinedNHSHandler(combined_nhs, fill(false, length(systems)))
end

function CombinedNHSHandler(neighborhood_search, systems)
    throw(ArgumentError("`CombinedNHSHandler` requires a `GridNeighborhoodSearch`"))
end

function neighborhood_search_name(handler::CombinedNHSHandler)
    return nameof(typeof(handler.neighborhood_search))
end

@inline function get_neighborhood_search(handler::CombinedNHSHandler, system_index,
                                         neighbor_index, search_radius)
    (; neighborhood_search) = handler

    # Subtract `eps` to avoid an error due to rounding errors.
    @boundscheck if search_radius - eps(search_radius) >
                    PointNeighbors.search_radius(neighborhood_search)
        throw(ArgumentError("no combined neighborhood search with radius >= $search_radius"))
    end

    return PointNeighbors.grid_nhs_pair(neighborhood_search, system_index, neighbor_index)
end

# === Neighborhood search lookup ===
@inline function get_neighborhood_search(system, semi)
    return get_neighborhood_search(system, system, semi)
end

@inline function get_neighborhood_search(system::TotalLagrangianSPHSystem, semi)
    # For TLSPH, use the specialized self-interaction neighborhood search
    # for finding neighbors in the initial configuration.
    return system.self_interaction_nhs
end

@inline function get_neighborhood_search(system, neighbor_system, semi)
    (; neighborhood_search_handler) = semi

    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)
    search_radius = compact_support(system, neighbor_system)

    return get_neighborhood_search(neighborhood_search_handler, system_index,
                                   neighbor_index, search_radius)
end

@inline function get_neighborhood_search(system::TotalLagrangianSPHSystem,
                                         neighbor_system::TotalLagrangianSPHSystem, semi)
    (; neighborhood_search_handler) = semi

    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)

    if system_index == neighbor_index
        # For TLSPH, use the specialized self-interaction neighborhood search
        # for finding neighbors in the initial configuration.
        return system.self_interaction_nhs
    end

    search_radius = compact_support(system, neighbor_system)

    return get_neighborhood_search(neighborhood_search_handler, system_index,
                                   neighbor_index, search_radius)
end

# === Initialization ===

function initialize_neighborhood_searches!(semi, u0_ode, restart_with::Nothing)
    initialize_neighborhood_searches!(semi)
end

function initialize_neighborhood_searches!(semi)
    initialize_neighborhood_searches!(semi, semi.neighborhood_search_handler)
end

function initialize_neighborhood_searches!(semi, handler)
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            initialize_neighborhood_search!(semi, system, neighbor)
        end
    end

    return semi
end

function initialize_neighborhood_searches!(semi, handler::CombinedNHSHandler)
    return initialize_combined_neighborhood_search!(initial_coordinates, semi, handler)
end

# `coordinates(system)` returns the coordinates to initialize the cell list of `system` with.
function initialize_combined_neighborhood_search!(coordinates, semi,
                                                  handler::CombinedNHSHandler)
    foreach_system(semi) do system
        # TODO Initialize after adapting to the GPU.
        # Currently, this cannot use `semi.parallelization_backend`
        # because data is still on the CPU.
        PointNeighbors.initialize!(handler.neighborhood_search,
                                   system_indices(system, semi),
                                   coordinates(system),
                                   eachindex=each_active_particle(system),
                                   parallelization_backend=PolyesterBackend())
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
    update_nhs!(semi, semi.neighborhood_search_handler, u_ode)
end

function update_nhs!(semi, handler, u_ode)
    # Update NHS for each pair of systems
    foreach_system(semi) do system
        u_system = wrap_u(u_ode, system, semi)

        foreach_system(semi) do neighbor
            points_moving_ = points_moving(system, neighbor)

            # Skip the update if no points are moving. This also skips pairs whose NHS
            # is never updated, like the frozen TLSPH self-interaction NHS.
            any(points_moving_) || return nothing

            u_neighbor = wrap_u(u_ode, neighbor, semi)
            neighborhood_search = get_neighborhood_search(system, neighbor, semi)
            neighbor_coords = current_coordinates(u_neighbor, neighbor)

            eachindex_y = each_active_particle(neighbor)
            if eachindex_y != axes(neighbor_coords, 2) &&
               !PointNeighbors.supports_inactive_points(neighborhood_search)
                if PointNeighbors.supports_infinite_domain(neighborhood_search)
                    # If the neighborhood search supports an infinite domain,
                    # we can update it for inactive points as well, which will not interfere
                    # with active points because their coordinates are 1e16.
                    eachindex_y = axes(neighbor_coords, 2)
                else
                    error("the neighborhood search/update strategy " *
                          "$(typeof(neighborhood_search)) does not support inactive points")
                end
            end

            update!(neighborhood_search,
                    current_coordinates(u_system, system),
                    current_coordinates(u_neighbor, neighbor),
                    semi; points_moving=points_moving_, eachindex_y)
        end
    end
end

function update_nhs!(semi, handler::CombinedNHSHandler, u_ode)
    (; neighborhood_search, systems_moving) = handler

    # The per-pair `points_moving` methods define which systems have to be updated.
    # Instead of updating the neighborhood search of each pair, we set a system as moving
    # if it appears as moving in at least one pair.
    # Note that a `GridNHSPair` requires up-to-date cell lists of both systems
    # (see `PointNeighbors.requires_update`), not only of the neighbor system.
    # The systems flagged as moving are then updated only once.
    systems_moving .= false
    foreach_system(semi) do system
        system_index = system_indices(system, semi)

        foreach_system(semi) do neighbor
            neighbor_index = system_indices(neighbor, semi)
            system_moving, neighbor_moving = points_moving(system, neighbor)

            systems_moving[system_index] |= system_moving
            systems_moving[neighbor_index] |= neighbor_moving
        end
    end

    # Update the NHS of each moving system only once.
    foreach_system(semi) do system
        system_index = system_indices(system, semi)

        if systems_moving[system_index]
            u_system = wrap_u(u_ode, system, semi)
            PointNeighbors.update!(neighborhood_search, system_index,
                                   current_coordinates(u_system, system),
                                   eachindex=each_active_particle(system),
                                   parallelization_backend=semi.parallelization_backend)
        end
    end

    return semi
end

# === Which points move between updates (per-pair dispatch) ===
# Returns `(system_moving, neighbor_moving)`.
# To prevent hard-to-find bugs, there is no default version.
# -- Fluid / structure interactions
# The current coordinates of fluids and structures change over time.
function points_moving(system::AbstractFluidSystem,
                       neighbor::Union{AbstractFluidSystem, TotalLagrangianSPHSystem,
                                       RigidBodySystem})
    return (true, true)
end

# -- Fluid / wall interactions
# Boundary coordinates only change over time when `neighbor.ismoving[]`.
# Open boundary particles only need to see walls with `BoundaryModelDynamicalPressureZhang`.
function points_moving(system::Union{AbstractFluidSystem,
                                     OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang}},
                       neighbor::WallBoundarySystem)
    return (true, neighbor.ismoving[])
end

# -- Open boundary interactions
# The current coordinates of fluids and open boundaries change over time.
function points_moving(system::AbstractFluidSystem, neighbor::OpenBoundarySystem)
    return (true, true)
end

# The current coordinates of both open boundaries and fluids change over time.
function points_moving(system::OpenBoundarySystem, neighbor::AbstractFluidSystem)
    return (true, true)
end

# Open boundary particles only have to see each other
# with `BoundaryModelDynamicalPressureZhang`.
function points_moving(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                       neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    return (true, true)
end

# Open boundary particles only have to see walls with `BoundaryModelDynamicalPressureZhang`.
function points_moving(system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                       neighbor::TotalLagrangianSPHSystem)
    return (true, true)
end

function points_moving(system::TotalLagrangianSPHSystem,
                       neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    return (true, true)
end

# -- TLSPH interactions
# The current coordinates of fluids and structures change over time.
function points_moving(system::TotalLagrangianSPHSystem, neighbor::AbstractFluidSystem)
    return (true, true)
end

# Boundary coordinates only change over time when `neighbor.ismoving[]`.
function points_moving(system::TotalLagrangianSPHSystem, neighbor::WallBoundarySystem)
    return (true, neighbor.ismoving[])
end

# -- Rigid body interactions
function points_moving(system::RigidBodySystem,
                       neighbor::Union{AbstractFluidSystem, RigidBodySystem})
    return (true, true)
end

function points_moving(system::RigidBodySystem, neighbor::WallBoundarySystem)
    return (true, neighbor.ismoving[])
end

# -- Wall dummy particle interactions
# Depending on the density calculator of the boundary model, this NHS is used for
# - kernel summation (`SummationDensity`)
# - continuity equation (`ContinuityDensity`)
# - pressure extrapolation (`AdamiPressureExtrapolation`)
#
# Boundary coordinates only change over time when `system.ismoving[]`.
# The current coordinates of fluids, open boundaries and structures change over time.
# These functions are the same to avoid ambiguous dispatch when using `Union`.
function points_moving(system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                       neighbor::AbstractFluidSystem)
    return (system.ismoving[], true)
end

function points_moving(system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                       neighbor::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang})
    return (system.ismoving[], true)
end

function points_moving(system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                       neighbor::TotalLagrangianSPHSystem)
    return (system.ismoving[], true)
end

# -- Wall / wall interactions
function points_moving(system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                       neighbor::WallBoundarySystem)
    return (system.ismoving[], neighbor.ismoving[])
end

# -- DEM interactions
# Both coordinates change over time.
function points_moving(system::DEMSystem, neighbor::DEMSystem)
    return (true, true)
end

# DEM coordinates change over time, the boundary coordinates don't.
function points_moving(system::DEMSystem, neighbor::BoundaryDEMSystem)
    return (true, false)
end

# -- Combinations that are never used.
# Either neither system moves or this NHS is never used.
function points_moving(system::RigidBodySystem, neighbor::OpenBoundarySystem)
    return (false, false)
end

function points_moving(system::OpenBoundarySystem, neighbor::TotalLagrangianSPHSystem)
    return (false, false)
end

function points_moving(system::TotalLagrangianSPHSystem, neighbor::OpenBoundarySystem)
    return (false, false)
end

function points_moving(system::TotalLagrangianSPHSystem, neighbor::RigidBodySystem)
    return (false, false)
end

# TLSPH systems have their own self-interaction NHS.
function points_moving(system::TotalLagrangianSPHSystem,
                       neighbor::TotalLagrangianSPHSystem)
    return (false, false)
end

function points_moving(system::RigidBodySystem, neighbor::TotalLagrangianSPHSystem)
    return (false, false)
end

# Rigid-wall contact is only computed from the rigid system side.
function points_moving(system::WallBoundarySystem{<:BoundaryModelDummyParticles},
                       neighbor::RigidBodySystem)
    return (false, false)
end

function points_moving(system::WallBoundarySystem,
                       neighbor::Union{AbstractFluidSystem, RigidBodySystem})
    return (false, false)
end

function points_moving(system::BoundaryDEMSystem,
                       neighbor::Union{DEMSystem, BoundaryDEMSystem})
    return (false, false)
end

function points_moving(system::Union{WallBoundarySystem, OpenBoundarySystem},
                       neighbor::Union{WallBoundarySystem, OpenBoundarySystem})
    return (false, false)
end

function points_moving(system::OpenBoundarySystem, neighbor::RigidBodySystem)
    return (false, false)
end

# === PointNeighbors forwarding ===
function update!(neighborhood_search, x, y, semi; points_moving=(true, true),
                 eachindex_y=axes(y, 2))
    PointNeighbors.update!(neighborhood_search, x, y; points_moving, eachindex_y,
                           parallelization_backend=semi.parallelization_backend)
end
