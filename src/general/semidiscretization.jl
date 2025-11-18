"""
    Semidiscretization(systems...; neighborhood_search=GridNeighborhoodSearch{NDIMS}())

The semidiscretization couples the passed systems to one simulation.

# Arguments
- `systems`: Systems to be coupled in this semidiscretization

# Keywords
- `neighborhood_search`:    The neighborhood search to be used in the simulation.
                            By default, the [`GridNeighborhoodSearch`](@ref) is used.
                            Use `nothing` to loop over all particles (no neighborhood search).
                            To use other neighborhood search implementations, pass a template
                            of a neighborhood search. See [`copy_neighborhood_search`](@ref)
                            and the examples below for more details.
                            To use a periodic domain, pass a [`PeriodicBox`](@ref) to the
                            neighborhood search.
- `threaded_nhs_update=true`:   Can be used to deactivate thread parallelization in the neighborhood search update.
                                This can be one of the largest sources of variations between simulations
                                with different thread numbers due to particle ordering changes.

# Examples
```jldoctest; output = false, setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); ref_system = fluid_system)
semi = Semidiscretization(fluid_system, boundary_system)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=SerialUpdate()))

periodic_box = PeriodicBox(min_corner = [0.0, 0.0], max_corner = [1.0, 1.0])
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=PrecomputedNeighborhoodSearch{2}())

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=nothing)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Semidiscretization                                                                               │
│ ══════════════════                                                                               │
│ #spatial dimensions: ………………………… 2                                                                │
│ #systems: ……………………………………………………… 2                                                                │
│ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
│ total #particles: ………………………………… 636                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct Semidiscretization{BACKEND, S, RU, RV, NS, UCU, IT}
    systems                 :: S
    ranges_u                :: RU
    ranges_v                :: RV
    neighborhood_searches   :: NS
    parallelization_backend :: BACKEND
    update_callback_used    :: UCU
    integrate_tlsph         :: IT # `false` if TLSPH integration is decoupled

    # Dispatch at `systems` to distinguish this constructor from the one below when
    # 4 systems are passed.
    # This is an internal constructor only used in `test/count_allocations.jl`
    # and by Adapt.jl.
    function Semidiscretization(systems::Tuple, ranges_u, ranges_v, neighborhood_searches,
                                parallelization_backend::PointNeighbors.ParallelizationBackend,
                                update_callback_used, integrate_tlsph)
        new{typeof(parallelization_backend), typeof(systems), typeof(ranges_u),
            typeof(ranges_v), typeof(neighborhood_searches),
            typeof(update_callback_used),
            typeof(integrate_tlsph)}(systems, ranges_u, ranges_v,
                                     neighborhood_searches, parallelization_backend,
                                     update_callback_used, integrate_tlsph)
    end
end

function Semidiscretization(systems::Union{AbstractSystem, Nothing}...;
                            neighborhood_search=GridNeighborhoodSearch{ndims(first(systems))}(),
                            parallelization_backend=PolyesterBackend())
    systems = filter(system -> !isnothing(system), systems)

    # Check e.g. that the boundary systems are using a state equation if EDAC is not used.
    # Other checks might be added here later.
    check_configuration(systems, neighborhood_search)

    sizes_u = [u_nvariables(system) * n_integrated_particles(system)
               for system in systems]
    ranges_u = Tuple((sum(sizes_u[1:(i - 1)]) + 1):sum(sizes_u[1:i])
                     for i in eachindex(sizes_u))
    sizes_v = [v_nvariables(system) * n_integrated_particles(system)
               for system in systems]
    ranges_v = Tuple((sum(sizes_v[1:(i - 1)]) + 1):sum(sizes_v[1:i])
                     for i in eachindex(sizes_v))

    # Create a tuple of n neighborhood searches for each of the n systems.
    # We will need one neighborhood search for each pair of systems.
    searches = Tuple(Tuple(create_neighborhood_search(neighborhood_search,
                                                      system, neighbor)
                           for neighbor in systems)
                     for system in systems)

    # These will be set to true inside the `UpdateCallback`.
    # Some techniques require the use of this callback, and this flag can be used
    # to determine if the callback is used in a simulation.
    update_callback_used = Ref(false)

    # Always integrate TLSPH systems together with other systems.
    # For split integration, a copy of the semidiscretization will be created
    # with this set to false.
    integrate_tlsph = Ref(true)

    return Semidiscretization(systems, ranges_u, ranges_v, searches,
                              parallelization_backend, update_callback_used,
                              integrate_tlsph)
end

# Inline show function e.g. Semidiscretization(neighborhood_search=...)
function Base.show(io::IO, semi::Semidiscretization)
    @nospecialize semi # reduce precompilation time

    print(io, "Semidiscretization(")
    for system in semi.systems
        print(io, system, ", ")
    end
    print(io, "neighborhood_search=")
    print(io, semi.neighborhood_searches |> eltype |> eltype |> nameof)
    print(io, ")")
end

# Show used during summary printout
function Base.show(io::IO, ::MIME"text/plain", semi::Semidiscretization)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "Semidiscretization")
        summary_line(io, "#spatial dimensions", ndims(semi.systems[1]))
        summary_line(io, "#systems", length(semi.systems))
        summary_line(io, "neighborhood search",
                     semi.neighborhood_searches |> eltype |> eltype |> nameof)
        summary_line(io, "total #particles", sum(nparticles.(semi.systems)))
        summary_footer(io)
    end
end

function create_neighborhood_search(::Nothing, system, neighbor)
    nhs = TrivialNeighborhoodSearch{ndims(system)}()

    return create_neighborhood_search(nhs, system, neighbor)
end

function create_neighborhood_search(neighborhood_search, system, neighbor)
    return copy_neighborhood_search(neighborhood_search, compact_support(system, neighbor),
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

@inline function get_neighborhood_search(system, neighbor_system, semi)
    (; neighborhood_searches) = semi

    system_index = system_indices(system, semi)
    neighbor_index = system_indices(neighbor_system, semi)

    return neighborhood_searches[system_index][neighbor_index]
end

@inline function system_indices(system, semi)
    # Note that this takes only about 5 ns, while mapping systems to indices with a `Dict`
    # is ~30x slower because `hash(::System)` is very slow.
    index = findfirst(==(system), semi.systems)

    if isnothing(index)
        throw(ArgumentError("system is not in the semidiscretization"))
    end

    return index
end

# This is just for readability to loop over all systems without allocations
@inline function foreach_system(f, semi::Union{NamedTuple, Semidiscretization})
    return foreach_noalloc(f, semi.systems)
end

@inline foreach_system(f, systems) = foreach_noalloc(f, systems)

"""
    semidiscretize(semi, tspan; reset_threads=true)

Create an `ODEProblem` from the semidiscretization with the specified `tspan`.

# Arguments
- `semi`: A [`Semidiscretization`](@ref) holding the systems involved in the simulation.
- `tspan`: The time span over which the simulation will be run.

# Keywords
- `reset_threads`: A boolean flag to reset Polyester.jl threads before the simulation (default: `true`).
  After an error within a threaded loop, threading might be disabled. Resetting the threads before the simulation
  ensures that threading is enabled again for the simulation.
  See also [trixi-framework/Trixi.jl#1583](https://github.com/trixi-framework/Trixi.jl/issues/1583).

# Returns
A `DynamicalODEProblem` (see [the OrdinaryDiffEq.jl docs](https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/))
to be integrated with [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl).
Note that this is not a true `DynamicalODEProblem` where the acceleration does not depend on the velocity.
Therefore, not all integrators designed for `DynamicalODEProblem`s will work properly.
However, all integrators designed for `ODEProblem`s can be used.
See [time integration](@ref time_integration) for more details.

# Examples
```jldoctest; output = false, filter = r"u0: .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), sol=nothing); ref_system = fluid_system)
semi = Semidiscretization(fluid_system, boundary_system)
tspan = (0.0, 1.0)
ode_problem = semidiscretize(semi, tspan)

# output
ODEProblem with uType RecursiveArrayTools.ArrayPartition{Float64, Tuple{TrixiParticles.ThreadedBroadcastArray{Float64, 1, Vector{Float64}, PolyesterBackend}, TrixiParticles.ThreadedBroadcastArray{Float64, 1, Vector{Float64}, PolyesterBackend}}} and tType Float64. In-place: true
Non-trivial mass matrix: false
timespan: (0.0, 1.0)
u0: ([...], [...]) *this line is ignored by filter*
```
"""
function semidiscretize(semi, tspan; reset_threads=true)
    (; systems) = semi

    @assert all(system -> eltype(system) === eltype(systems[1]), systems)
    ELTYPE = eltype(systems[1])

    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    sizes_u = (u_nvariables(system) * n_integrated_particles(system) for system in systems)
    sizes_v = (v_nvariables(system) * n_integrated_particles(system) for system in systems)

    # Use either the specified backend, e.g., `CUDABackend` or `MetalBackend` or
    # use CPU vectors for all CPU backends.
    u0_ode_ = allocate(semi.parallelization_backend, ELTYPE, sum(sizes_u))
    v0_ode_ = allocate(semi.parallelization_backend, ELTYPE, sum(sizes_v))

    if semi.parallelization_backend isa KernelAbstractions.Backend
        u0_ode = u0_ode_
        v0_ode = v0_ode_
    else
        # CPU vectors are wrapped in `ThreadedBroadcastArray`s
        # to make broadcasting (which is done by OrdinaryDiffEq.jl) multithreaded.
        # See https://github.com/trixi-framework/TrixiParticles.jl/pull/722 for more details.
        u0_ode = ThreadedBroadcastArray(u0_ode_;
                                        parallelization_backend=semi.parallelization_backend)
        v0_ode = ThreadedBroadcastArray(v0_ode_;
                                        parallelization_backend=semi.parallelization_backend)
    end

    # Set initial condition
    foreach_system(semi) do system
        u0_system = wrap_u(u0_ode, system, semi)
        v0_system = wrap_v(v0_ode, system, semi)

        write_u0!(u0_system, system)
        write_v0!(v0_system, system)
    end

    # TODO initialize after adapting to the GPU.
    # Requires https://github.com/trixi-framework/PointNeighbors.jl/pull/86.
    initialize_neighborhood_searches!(semi)

    if semi.parallelization_backend isa KernelAbstractions.Backend
        # Convert all arrays to the correct array type.
        # When e.g. `parallelization_backend=CUDABackend()`, this will convert all `Array`s
        # to `CuArray`s, moving data to the GPU.
        # See the comments in general/gpu.jl for more details.
        semi_ = Adapt.adapt(semi.parallelization_backend, semi)

        # We now have a new `Semidiscretization` with new systems.
        # This means that systems linking to other systems still point to old systems.
        # Therefore, we have to re-link them, which yields yet another `Semidiscretization`.
        # Note that this re-creates systems containing links, so it only works as long
        # as systems don't link to other systems containing links.
        semi_new = Semidiscretization(set_system_links.(semi_.systems, Ref(semi_)),
                                      semi_.ranges_u, semi_.ranges_v,
                                      semi_.neighborhood_searches,
                                      semi_.parallelization_backend,
                                      semi_.update_callback_used, semi_.integrate_tlsph)
    else
        semi_new = semi
    end

    # Initialize all particle systems
    foreach_system(semi_new) do system
        # Initialize this system
        initialize!(system, semi_new)
    end

    # Reset callback flag that will be set by the `UpdateCallback`
    semi_new.update_callback_used[] = false

    return DynamicalODEProblem(kick!, drift!, v0_ode, u0_ode, tspan, semi_new)
end

"""
    restart_with!(semi, sol)

Set the initial coordinates and velocities of all systems in `semi` to the final values
in the solution `sol`.
[`semidiscretize`](@ref) has to be called again afterwards, or another
[`Semidiscretization`](@ref) can be created with the updated systems.

# Arguments
- `semi`:   The semidiscretization
- `sol`:    The `ODESolution` returned by `solve` of `OrdinaryDiffEq`
"""
function restart_with!(semi, sol; reset_threads=true)
    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    initialize_neighborhood_searches!(semi)

    foreach_system(semi) do system
        v = wrap_v(sol.u[end].x[1], system, semi)
        u = wrap_u(sol.u[end].x[2], system, semi)

        restart_with!(system, v, u)
    end

    # Reset callback flag that will be set by the `UpdateCallback`
    semi.update_callback_used[] = false

    return semi
end

function initialize_neighborhood_searches!(semi)
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            # TODO Initialize after adapting to the GPU.
            # Currently, this cannot use `semi.parallelization_backend`
            # because data is still on the CPU.
            PointNeighbors.initialize!(get_neighborhood_search(system, neighbor, semi),
                                       initial_coordinates(system),
                                       initial_coordinates(neighbor),
                                       eachindex_y=each_active_particle(neighbor),
                                       parallelization_backend=PolyesterBackend())
        end
    end

    return semi
end

# We have to pass `system` here for type stability,
# since the type of `system` determines the return type.
@inline function wrap_v(v_ode, system, semi)
    (; ranges_v) = semi

    range = ranges_v[system_indices(system, semi)]

    @boundscheck @assert length(range) ==
                         v_nvariables(system) * n_integrated_particles(system)

    return wrap_array(v_ode, range,
                      (StaticInt(v_nvariables(system)), n_integrated_particles(system)))
end

@inline function wrap_u(u_ode, system, semi)
    (; ranges_u) = semi

    range = ranges_u[system_indices(system, semi)]

    @boundscheck @assert length(range) ==
                         u_nvariables(system) * n_integrated_particles(system)

    return wrap_array(u_ode, range,
                      (StaticInt(u_nvariables(system)), n_integrated_particles(system)))
end

@inline function wrap_array(array::Array, range, size)
    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(array), 2}, pointer(view(array, range)), size)
    return PtrArray(pointer(view(array, range)), size)
end

@inline function wrap_array(array::ThreadedBroadcastArray, range, size)
    return ThreadedBroadcastArray(wrap_array(parent(array), range, size))
end

@inline function wrap_array(array, range, size)
    # For non-`Array`s (typically GPU arrays), just reshape. Calling the `PtrArray` code
    # above for a `CuArray` yields another `CuArray` (instead of a `PtrArray`)
    # and is 8 times slower with double the allocations.
    #
    # Note that `size` might contain `StaticInt`s, so convert to `Int` first.
    return reshape(view(array, range), Int.(size))
end

function calculate_dt(v_ode, u_ode, cfl_number, semi::Semidiscretization)
    (; systems) = semi

    return minimum(system -> calculate_dt(v_ode, u_ode, cfl_number, system, semi), systems)
end

function drift!(du_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "drift!" begin
        @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du_ode)

        @trixi_timeit timer() "velocity" begin
            # Set velocity and add acceleration for each system
            foreach_system(semi) do system
                du = wrap_u(du_ode, system, semi)
                v = wrap_v(v_ode, system, semi)
                u = wrap_u(u_ode, system, semi)

                @threaded semi for particle in each_integrated_particle(system)
                    # This can be dispatched per system
                    add_velocity!(du, v, u, particle, system, semi, t)
                end
            end
        end
    end

    return du_ode
end

@inline function add_velocity!(du, v, u, particle, system, semi::Semidiscretization, t)
    add_velocity!(du, v, u, particle, system, t)
end

@inline function add_velocity!(du, v, u, particle, system::TotalLagrangianSPHSystem,
                               semi::Semidiscretization, t)
    # Only add velocity for TLSPH systems if they are integrated
    if semi.integrate_tlsph[]
        add_velocity!(du, v, u, particle, system, t)
    end
end

@inline function add_velocity!(du, v, u, particle, system, t)
    # Generic fallback for all systems that don't define this function
    for i in 1:ndims(system)
        @inbounds du[i, particle] = v[i, particle]
    end

    return du
end

# Solid wall boundary system doesn't integrate the particle positions
@inline add_velocity!(du, v, u, particle, system::WallBoundarySystem, t) = du

@inline function add_velocity!(du, v, u, particle, system::AbstractFluidSystem, t)
    # This is zero unless a shifting technique is used
    delta_v_ = delta_v(system, particle)

    for i in 1:ndims(system)
        @inbounds du[i, particle] = v[i, particle] + delta_v_[i]
    end

    return du
end

function kick!(dv_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "kick!" begin
        # Check that the `UpdateCallback` is used if required
        check_update_callback(semi)

        @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode)

        @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                              semi, t)

        @trixi_timeit timer() "system interaction" system_interaction!(dv_ode, v_ode, u_ode,
                                                                       semi)

        @trixi_timeit timer() "source terms" add_source_terms!(dv_ode, v_ode, u_ode,
                                                               semi, t)
    end

    return dv_ode
end

# Update the systems and neighborhood searches (NHS) for a simulation
# before calling `interact!` to compute forces.
function update_systems_and_nhs(v_ode, u_ode, semi, t)
    # First update step before updating the NHS
    # (for example for writing the current coordinates in the TLSPH system)
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_positions!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Update NHS
    @trixi_timeit timer() "update nhs" update_nhs!(semi, u_ode)

    # Second update step.
    # This is used to calculate density and pressure of the fluid systems
    # before updating the boundary systems,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    end

    update_implicit_sph!(semi, v_ode, u_ode, t)

    # Perform correction and pressure calculation
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    end

    # This update depends on the computed quantities of the fluid system and therefore
    # needs to be after `update_quantities!`.
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_boundary_interpolation!(system, v, u, v_ode, u_ode, semi, t)
    end

    # Final update step for all remaining systems
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        update_final!(system, v, u, v_ode, u_ode, semi, t)
    end
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

# The `SplitIntegrationCallback` overwrites `semi_wrap` to use a different
# semidiscretization for wrapping arrays.
# TODO `semi` is not used yet, but will be used when the source terms API is modified
# to match the custom quantities API.
function add_source_terms!(dv_ode, v_ode, u_ode, semi, t; semi_wrap=semi)
    foreach_system(semi_wrap) do system
        dv = wrap_v(dv_ode, system, semi_wrap)
        v = wrap_v(v_ode, system, semi_wrap)
        u = wrap_u(u_ode, system, semi_wrap)

        @threaded semi for particle in each_integrated_particle(system)
            # Dispatch by system type to exclude boundary systems.
            # `integrate_tlsph` is extracted from the `semi_wrap`, so that this function
            # can be used in the `SplitIntegrationCallback` as well.
            add_acceleration!(dv, particle, system, semi_wrap.integrate_tlsph[])
            add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t,
                                    semi_wrap.integrate_tlsph[])
        end
    end

    return dv_ode
end

@inline source_terms(system) = nothing
@inline source_terms(system::Union{AbstractFluidSystem, AbstractStructureSystem}) = system.source_terms

@inline function add_acceleration!(dv, particle, system, integrate_tlsph)
    add_acceleration!(dv, particle, system)
end

@inline function add_acceleration!(dv, particle, system::TotalLagrangianSPHSystem,
                                   integrate_tlsph)
    integrate_tlsph && add_acceleration!(dv, particle, system)
end

@inline add_acceleration!(dv, particle, system) = dv

@inline function add_acceleration!(dv, particle,
                                   system::Union{AbstractFluidSystem,
                                                 AbstractStructureSystem})
    (; acceleration) = system

    for i in 1:ndims(system)
        dv[i, particle] += acceleration[i]
    end

    return dv
end

@inline function add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t,
                                         integrate_tlsph)
    add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t)
end

@inline function add_source_terms_inner!(dv, v, u, particle,
                                         system::TotalLagrangianSPHSystem,
                                         source_terms_, t, integrate_tlsph)
    integrate_tlsph && add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t)
end

@inline function add_source_terms_inner!(dv, v, u, particle, system, source_terms_, t)
    coords = current_coords(u, system, particle)
    velocity = current_velocity(v, system, particle)
    density = current_density(v, system, particle)
    pressure = current_pressure(v, system, particle)

    source = source_terms_(coords, velocity, density, pressure, t)

    # Loop over `eachindex(source)`, so that users could also pass source terms for
    # the density when using `ContinuityDensity`.
    for i in eachindex(source)
        dv[i, particle] += source[i]
    end

    return dv
end

@inline add_source_terms_inner!(dv, v, u, particle, system, source_terms_::Nothing, t) = dv

@doc raw"""
    SourceTermDamping(; damping_coefficient)

A source term to be used when a damping step is required before running a full simulation.
The term ``-c \cdot v_a`` is added to the acceleration ``\frac{\mathrm{d}v_a}{\mathrm{d}t}``
of particle ``a``, where ``c`` is the damping coefficient and ``v_a`` is the velocity of
particle ``a``.

# Keywords
- `damping_coefficient`:    The coefficient ``d`` above. A higher coefficient means more
                            damping. A coefficient of `1e-4` is a good starting point for
                            damping a fluid at rest.

# Examples
```jldoctest; output = false
source_terms = SourceTermDamping(; damping_coefficient=1e-4)

# output
SourceTermDamping{Float64}(0.0001)
```
"""
struct SourceTermDamping{ELTYPE}
    damping_coefficient::ELTYPE

    function SourceTermDamping(; damping_coefficient)
        return new{typeof(damping_coefficient)}(damping_coefficient)
    end
end

@inline function (source_term::SourceTermDamping)(coords, velocity, density, pressure, t)
    (; damping_coefficient) = source_term

    return -damping_coefficient * velocity
end

function system_interaction!(dv_ode, v_ode, u_ode, semi)
    # Call `interact!` for each pair of systems
    foreach_system(semi) do system
        foreach_system(semi) do neighbor
            # Construct string for the interactions timer.
            # Avoid allocations from string construction when no timers are used.
            if timeit_debug_enabled()
                system_index = system_indices(system, semi)
                neighbor_index = system_indices(neighbor, semi)
                timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
            else
                timer_str = ""
            end

            interact!(dv_ode, v_ode, u_ode, system, neighbor, semi, timer_str=timer_str)
        end
    end

    return dv_ode
end

# Function barrier to make benchmarking interactions easier.
# One can benchmark, e.g. the fluid-fluid interaction, with:
# dv_ode, du_ode = copy(sol.u[end]).x; v_ode, u_ode = copy(sol.u[end]).x;
# @btime TrixiParticles.interact!($dv_ode, $v_ode, $u_ode, $fluid_system, $fluid_system, $semi);
@inline function interact!(dv_ode, v_ode, u_ode, system, neighbor, semi; timer_str="")
    dv = wrap_v(dv_ode, system, semi)
    v_system = wrap_v(v_ode, system, semi)
    u_system = wrap_u(u_ode, system, semi)

    v_neighbor = wrap_v(v_ode, neighbor, semi)
    u_neighbor = wrap_u(u_ode, neighbor, semi)

    @trixi_timeit timer() timer_str begin
        interact!(dv, v_system, u_system, v_neighbor, u_neighbor, system, neighbor, semi)
    end
end

# NHS updates
# To prevent hard-to-find bugs, there is not default version
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
    # Don't update. Neighborhood search works on the initial coordinates, which don't change.
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

function check_update_callback(semi)
    foreach_system(semi) do system
        # This check will be optimized away if the system does not require the callback
        if requires_update_callback(system) && !semi.update_callback_used[]
            system_name = system |> typeof |> nameof
            throw(ArgumentError("`UpdateCallback` is required for `$system_name`"))
        end
    end
end

function check_configuration(systems,
                             nhs::Union{Nothing, PointNeighbors.AbstractNeighborhoodSearch})
    foreach_system(systems) do system
        check_configuration(system, systems, nhs)
    end

    check_system_color(systems)
end

check_configuration(system::AbstractSystem, systems, nhs) = nothing

function check_system_color(systems)
    if any(system isa AbstractFluidSystem && !(system isa ParticlePackingSystem) &&
           !isnothing(system.surface_tension)
           for system in systems)

        # System indices of all systems that are either a fluid or a boundary system
        system_ids = findall(system isa Union{AbstractFluidSystem, WallBoundarySystem}
                             for system in systems)

        if length(system_ids) > 1 && sum(i -> systems[i].cache.color, system_ids) == 0
            throw(ArgumentError("If a surface tension model is used the values of at least one system needs to have a color different than 0."))
        end
    end
end

function check_configuration(fluid_system::AbstractFluidSystem, systems, nhs)
    if !(fluid_system isa ParticlePackingSystem) && !isnothing(fluid_system.surface_tension)
        foreach_system(systems) do neighbor
            if neighbor isa AbstractFluidSystem &&
               isnothing(fluid_system.surface_tension) &&
               isnothing(fluid_system.surface_normal_method)
                throw(ArgumentError("either none or all fluid systems in a simulation need " *
                                    "to use a surface tension model or a surface normal method."))
            end
        end
    end
end

function check_configuration(system::WallBoundarySystem, systems, nhs)
    (; boundary_model) = system

    foreach_system(systems) do neighbor
        if neighbor isa WeaklyCompressibleSPHSystem &&
           boundary_model isa BoundaryModelDummyParticles &&
           isnothing(boundary_model.state_equation)
            throw(ArgumentError("`WeaklyCompressibleSPHSystem` cannot be used without " *
                                "setting a `state_equation` for all boundary models"))
        end
    end
end

function check_configuration(system::TotalLagrangianSPHSystem, systems, nhs)
    (; boundary_model) = system

    foreach_system(systems) do neighbor
        if neighbor isa AbstractFluidSystem && boundary_model === nothing
            throw(ArgumentError("a boundary model for `TotalLagrangianSPHSystem` must be " *
                                "specified when simulating a fluid-structure interaction."))
        end
    end

    if boundary_model isa BoundaryModelDummyParticles &&
       boundary_model.density_calculator isa ContinuityDensity
        throw(ArgumentError("`BoundaryModelDummyParticles` with density calculator " *
                            "`ContinuityDensity` is not yet supported for a `TotalLagrangianSPHSystem`"))
    end
end

function check_configuration(system::ImplicitIncompressibleSPHSystem, systems, nhs)
    (; time_step, omega) = system
    foreach_system(systems) do neighbor
        if neighbor isa WeaklyCompressibleSPHSystem
            throw(ArgumentError("`ImplicitIncompressibleSPHSystem` cannot be used together with
            `WeaklyCompressibleSPHSystem`"))
        end
        if neighbor isa WallBoundarySystem
            if (neighbor.boundary_model isa BoundaryModelDummyParticles &&
                neighbor.boundary_model.density_calculator isa PressureBoundaries)
                time_step_boundary = neighbor.boundary_model.density_calculator.time_step
                omega_boundary = neighbor.boundary_model.density_calculator.omega
                if !(time_step==time_step_boundary && omega==omega_boundary)
                    throw(ArgumentError("`PressureBoundaries` parameters have to be the same as the
                    `ImplicitIncompressibleSPHSystem` parameters"))
                end
            end
        end
    end
end

function check_configuration(system::OpenBoundarySystem, systems,
                             neighborhood_search::PointNeighbors.AbstractNeighborhoodSearch)
    (; boundary_model, boundary_zones) = system

    # Store index of the fluid system. This is necessary for re-linking
    # in case we use Adapt.jl to create a new semidiscretization.
    fluid_system_index = findfirst(==(system.fluid_system), systems)
    system.fluid_system_index[] = fluid_system_index

    if boundary_model isa BoundaryModelCharacteristicsLastiwka &&
       any(zone -> isnothing(zone.flow_direction), boundary_zones)
        throw(ArgumentError("`BoundaryModelCharacteristicsLastiwka` needs a specific flow direction. " *
                            "Please specify `InFlow()` and `OutFlow()`."))
    end

    if first(PointNeighbors.requires_update(neighborhood_search))
        throw(ArgumentError("`OpenBoundarySystem` requires a neighborhood search " *
                            "that does not require an update for the first set of coordinates (e.g. `GridNeighborhoodSearch`). " *
                            "See the PointNeighbors.jl documentation for more details."))
    end
end

# After `adapt`, the system type information may change.
# This means that systems linking to other systems still point to old systems.
# Therefore, we have to re-link them based on the stored system index.
set_system_links(system, semi) = system

function set_system_links(system::OpenBoundarySystem, semi)
    fluid_system = semi.systems[system.fluid_system_index[]]

    return OpenBoundarySystem(system.boundary_model,
                              system.initial_condition,
                              fluid_system, # link to fluid system
                              system.fluid_system_index,
                              system.smoothing_kernel,
                              system.smoothing_length,
                              system.mass,
                              system.volume,
                              system.boundary_candidates,
                              system.fluid_candidates,
                              system.boundary_zone_indices,
                              system.boundary_zones,
                              system.buffer,
                              system.pressure_acceleration_formulation,
                              system.shifting_technique,
                              system.cache)
end
