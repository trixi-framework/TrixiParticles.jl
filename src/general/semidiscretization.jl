"""
    Semidiscretization(systems...; neighborhood_search=nothing, damping_coefficient=nothing)

The semidiscretization couples the passed systems to one simulation.

The type of neighborhood search to be used in the simulation can be specified with
the keyword argument `neighborhood_search`. A value of `nothing` means no neighborhood search.

# Examples
```julia
semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search=GridNeighborhoodSearch, damping_coefficient=nothing)
```
"""
struct Semidiscretization{S, RU, RV, NS, DC}
    systems               :: S
    ranges_u              :: RU
    ranges_v              :: RV
    neighborhood_searches :: NS
    damping_coefficient   :: DC

    function Semidiscretization(systems...; neighborhood_search=nothing,
                                periodic_box_min_corner=nothing,
                                periodic_box_max_corner=nothing,
                                damping_coefficient=nothing)
        sizes_u = [u_nvariables(system) * n_moving_particles(system)
                   for system in systems]
        ranges_u = Tuple((sum(sizes_u[1:(i - 1)]) + 1):sum(sizes_u[1:i])
                         for i in eachindex(sizes_u))
        sizes_v = [v_nvariables(system) * n_moving_particles(system)
                   for system in systems]
        ranges_v = Tuple((sum(sizes_v[1:(i - 1)]) + 1):sum(sizes_v[1:i])
                         for i in eachindex(sizes_v))

        check_configuration(systems)

        # Create (and initialize) a tuple of n neighborhood searches for each of the n systems
        # We will need one neighborhood search for each pair of systems.
        searches = Tuple(Tuple(create_neighborhood_search(system, neighbor,
                                                          Val(neighborhood_search),
                                                          periodic_box_min_corner,
                                                          periodic_box_max_corner)
                               for neighbor in systems)
                         for system in systems)

        new{typeof(systems), typeof(ranges_u), typeof(ranges_v),
            typeof(searches), typeof(damping_coefficient)}(systems, ranges_u, ranges_v,
                                                           searches, damping_coefficient)
    end
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
        summary_line(io, "damping coefficient", semi.damping_coefficient)
        summary_line(io, "total #particles", sum(nparticles.(semi.systems)))
        summary_footer(io)
    end
end

function create_neighborhood_search(system, neighbor, ::Val{nothing},
                                    min_corner, max_corner)
    radius = compact_support(system, neighbor)
    TrivialNeighborhoodSearch{ndims(system)}(radius, eachparticle(neighbor),
                                             min_corner=min_corner, max_corner=max_corner)
end

function create_neighborhood_search(system, neighbor, ::Val{GridNeighborhoodSearch},
                                    min_corner, max_corner)
    radius = compact_support(system, neighbor)
    search = GridNeighborhoodSearch{ndims(system)}(radius, nparticles(neighbor),
                                                   min_corner=min_corner,
                                                   max_corner=max_corner)

    # Initialize neighborhood search
    initialize!(search, initial_coordinates(neighbor))

    return search
end

@inline function compact_support(system, neighbor)
    (; smoothing_kernel, smoothing_length) = system
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function compact_support(system::TotalLagrangianSPHSystem,
                                 neighbor::TotalLagrangianSPHSystem)
    (; smoothing_kernel, smoothing_length) = system
    return compact_support(smoothing_kernel, smoothing_length)
end

@inline function compact_support(system::Union{TotalLagrangianSPHSystem, BoundarySPHSystem},
                                 neighbor)
    return compact_support(system, system.boundary_model, neighbor)
end

@inline function compact_support(system, model, neighbor)
    # Use the compact support of the fluid for solid-fluid interaction
    return compact_support(neighbor, system)
end

@inline function compact_support(system, model::BoundaryModelDummyParticles, neighbor)
    # TODO: Monaghan-Kajtar BC are using the fluid's compact support for solid-fluid
    # interaction. Dummy particle BC use the model's compact support, which is also used
    # for density summations.
    (; smoothing_kernel, smoothing_length) = model
    return compact_support(smoothing_kernel, smoothing_length)
end

"""
    semidiscretize(semi, tspan)

Create an `ODEProblem` from the semidiscretization with the specified `tspan`.
"""
function semidiscretize(semi, tspan; reset_threads=true)
    (; systems, neighborhood_searches) = semi

    @assert all(system -> eltype(system) === eltype(systems[1]),
                systems)
    ELTYPE = eltype(systems[1])

    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    # Initialize all particle systems
    @trixi_timeit timer() "initialize particle systems" begin
        for (system_index, system) in pairs(systems)
            # Get the neighborhood search for this system
            neighborhood_search = neighborhood_searches[system_index][system_index]

            # Initialize this system
            initialize!(system, neighborhood_search)
        end
    end

    sizes_u = (u_nvariables(system) * n_moving_particles(system)
               for system in systems)
    sizes_v = (v_nvariables(system) * n_moving_particles(system)
               for system in systems)
    u0_ode = Vector{ELTYPE}(undef, sum(sizes_u))
    v0_ode = Vector{ELTYPE}(undef, sum(sizes_v))

    for (system_index, system) in pairs(systems)
        u0_system = wrap_u(u0_ode, system_index, system, semi)
        v0_system = wrap_v(v0_ode, system_index, system, semi)

        write_u0!(u0_system, system)
        write_v0!(v0_system, system)
    end

    return DynamicalODEProblem(kick!, drift!, v0_ode, u0_ode, tspan, semi)
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
    (; systems) = semi

    # Optionally reset Polyester.jl threads. See
    # https://github.com/trixi-framework/Trixi.jl/issues/1583
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if reset_threads
        Polyester.reset_threads!()
    end

    foreach_enumerate(systems) do (system_index, system)
        v = wrap_v(sol[end].x[1], system_index, system, semi)
        u = wrap_u(sol[end].x[2], system_index, system, semi)

        restart_with!(system, v, u)
    end

    return semi
end

# We have to pass `system` here for type stability,
# since the type of `system` determines the return type.
@inline function wrap_u(u_ode, i, system, semi)
    (; ranges_u) = semi

    range = ranges_u[i]

    @boundscheck begin
        @assert length(range) ==
                u_nvariables(system) * n_moving_particles(system)
    end

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(u_ode), 2}, pointer(view(u_ode, range)),
    #                    (u_nvariables(system), n_moving_particles(system)))
    return PtrArray(pointer(view(u_ode, range)),
                    (StaticInt(u_nvariables(system)), n_moving_particles(system)))
end

@inline function wrap_v(v_ode, i, system, semi)
    (; ranges_v) = semi

    range = ranges_v[i]

    @boundscheck begin
        @assert length(range) ==
                v_nvariables(system) * n_moving_particles(system)
    end

    return PtrArray(pointer(view(v_ode, range)),
                    (StaticInt(v_nvariables(system)), n_moving_particles(system)))
end

function drift!(du_ode, v_ode, u_ode, semi, t)
    (; systems) = semi

    @trixi_timeit timer() "drift!" begin
        @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du_ode)

        @trixi_timeit timer() "velocity" begin
            # Set velocity and add acceleration for each system
            foreach_enumerate(systems) do (system_index, system)
                du = wrap_u(du_ode, system_index, system, semi)
                v = wrap_v(v_ode, system_index, system, semi)

                @threaded for particle in each_moving_particle(system)
                    # This can be dispatched per system
                    add_velocity!(du, v, particle, system)
                end
            end
        end
    end

    return du_ode
end

@inline function add_velocity!(du, v, particle, system)
    for i in 1:ndims(system)
        du[i, particle] = v[i, particle]
    end

    return du
end

@inline add_velocity!(du, v, particle, system::BoundarySPHSystem) = du

function kick!(dv_ode, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "kick!" begin
        @trixi_timeit timer() "reset ∂v/∂t" set_zero!(dv_ode)

        @trixi_timeit timer() "update systems and nhs" update_systems_and_nhs(v_ode, u_ode,
                                                                              semi, t)

        @trixi_timeit timer() "gravity and damping" gravity_and_damping!(dv_ode, v_ode,
                                                                         semi)

        @trixi_timeit timer() "system interaction" system_interaction!(dv_ode, v_ode, u_ode,
                                                                       semi)
    end

    return dv_ode
end

# Update the systems and neighborhood searches (NHS) for a simulation before calling `interact!` to compute forces
function update_systems_and_nhs(v_ode, u_ode, semi, t)
    (; systems) = semi

    # First update step before updating the NHS
    # (for example for writing the current coordinates in the solid system)
    foreach_enumerate(systems) do (system_index, system)
        v = wrap_v(v_ode, system_index, system, semi)
        u = wrap_u(u_ode, system_index, system, semi)

        update_positions!(system, system_index, v, u, v_ode, u_ode, semi, t)
    end

    # Update NHS
    @trixi_timeit timer() "update nhs" update_nhs(u_ode, semi)

    # Second update step.
    # This is used to calculate density and pressure of the fluid systems
    # before updating the boundary systems,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_enumerate(systems) do (system_index, system)
        v = wrap_v(v_ode, system_index, system, semi)
        u = wrap_u(u_ode, system_index, system, semi)

        update_quantities!(system, system_index, v, u, v_ode, u_ode, semi, t)
    end

    # Perform correction and pressure calculation
    foreach_enumerate(systems) do (system_index, system)
        v = wrap_v(v_ode, system_index, system, semi)
        u = wrap_u(u_ode, system_index, system, semi)

        update_pressure!(system, system_index, v, u, v_ode, u_ode, semi, t)
    end

    # Final update step for all remaining systems
    foreach_enumerate(systems) do (system_index, system)
        v = wrap_v(v_ode, system_index, system, semi)
        u = wrap_u(u_ode, system_index, system, semi)

        update_final!(system, system_index, v, u, v_ode, u_ode, semi, t)
    end
end

function update_nhs(u_ode, semi)
    (; systems, neighborhood_searches) = semi

    # Update NHS for each pair of systems
    foreach_enumerate(systems) do (system_index, system)
        foreach_enumerate(systems) do (neighbor_index, neighbor)
            u_neighbor = wrap_u(u_ode, neighbor_index, neighbor, semi)
            neighborhood_search = neighborhood_searches[system_index][neighbor_index]

            update!(neighborhood_search, nhs_coords(system, neighbor, u_neighbor))
        end
    end
end

function gravity_and_damping!(dv_ode, v_ode, semi)
    (; systems, damping_coefficient) = semi

    # Set velocity and add acceleration for each system
    foreach_enumerate(systems) do (system_index, system)
        dv = wrap_v(dv_ode, system_index, system, semi)
        v = wrap_v(v_ode, system_index, system, semi)

        @threaded for particle in each_moving_particle(system)
            # This can be dispatched per system
            add_acceleration!(dv, particle, system)
            add_damping_force!(dv, damping_coefficient, v, particle, system)
        end
    end

    return dv_ode
end

@inline function add_acceleration!(dv, particle, system)
    (; acceleration) = system

    for i in 1:ndims(system)
        dv[i, particle] += acceleration[i]
    end

    return dv
end

@inline add_acceleration!(dv, particle, system::BoundarySPHSystem) = dv

@inline function add_damping_force!(dv, damping_coefficient, v, particle,
                                    system::FluidSystem)
    for i in 1:ndims(system)
        dv[i, particle] -= damping_coefficient * v[i, particle]
    end

    return dv
end

# Currently no damping for non-fluid systems
@inline add_damping_force!(dv, damping_coefficient, v, particle, system) = dv
@inline add_damping_force!(dv, ::Nothing, v, particle, system::FluidSystem) = dv

function system_interaction!(dv_ode, v_ode, u_ode, semi)
    (; systems, neighborhood_searches) = semi

    # Call `interact!` for each pair of systems
    foreach_enumerate(systems) do (system_index, system)
        dv = wrap_v(dv_ode, system_index, system, semi)
        v_system = wrap_v(v_ode, system_index, system, semi)
        u_system = wrap_u(u_ode, system_index, system, semi)

        foreach_enumerate(systems) do (neighbor_index, neighbor)
            v_neighbor = wrap_v(v_ode, neighbor_index, neighbor, semi)
            u_neighbor = wrap_u(u_ode, neighbor_index, neighbor, semi)
            neighborhood_search = neighborhood_searches[system_index][neighbor_index]

            timer_str = "$(timer_name(system))$system_index-$(timer_name(neighbor))$neighbor_index"
            @trixi_timeit timer() timer_str begin
                interact!(dv, v_system, u_system, v_neighbor, u_neighbor,
                          neighborhood_search, system, neighbor)
            end
        end
    end

    return dv_ode
end

# NHS updates
function nhs_coords(system::FluidSystem,
                    neighbor::FluidSystem, u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::FluidSystem,
                    neighbor::TotalLagrangianSPHSystem, u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::FluidSystem,
                    neighbor::BoundarySPHSystem, u)
    if neighbor.ismoving[1]
        return current_coordinates(u, neighbor)
    end

    # Don't update
    return nothing
end

function nhs_coords(system::TotalLagrangianSPHSystem,
                    neighbor::FluidSystem, u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::TotalLagrangianSPHSystem,
                    neighbor::TotalLagrangianSPHSystem, u)
    # Don't update
    return nothing
end

function nhs_coords(system::TotalLagrangianSPHSystem,
                    neighbor::BoundarySPHSystem, u)
    if neighbor.ismoving[1]
        return current_coordinates(u, neighbor)
    end

    # Don't update
    return nothing
end

function nhs_coords(system::BoundarySPHSystem,
                    neighbor::FluidSystem, u)
    # Don't update
    return nothing
end

function nhs_coords(system::BoundarySPHSystem{<:BoundaryModelDummyParticles},
                    neighbor::FluidSystem, u)
    return current_coordinates(u, neighbor)
end

function nhs_coords(system::BoundarySPHSystem,
                    neighbor::TotalLagrangianSPHSystem, u)
    # Don't update
    return nothing
end

function nhs_coords(system::BoundarySPHSystem,
                    neighbor::BoundarySPHSystem, u)
    # Don't update
    return nothing
end

function check_configuration(systems)
    for sys in systems
        check_configuration(sys, systems)
    end
end

check_configuration(sys, systems) = systems

function check_configuration(bnd_sys::BoundarySPHSystem, systems)
    boundary_model = bnd_sys.boundary_model
    for neighbor in systems
        if neighbor isa WeaklyCompressibleSPHSystem &&
           boundary_model isa BoundaryModelDummyParticles &&
           isnothing(boundary_model.state_equation)
            throw(ArgumentError("`WeaklyCompressibleSPHSystem` cannot be used without setting a state_equation for boundary."))
        end
    end
end
