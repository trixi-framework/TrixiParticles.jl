"""
    Semidiscretization(particle_containers...; neighborhood_search=nothing, damping_coefficient=nothing)

The semidiscretization couples the passed particle containers to one simulation.

The type of neighborhood search to be used in the simulation can be specified with
the keyword argument `neighborhood_search`. A value of `nothing` means no neighborhood search.

# Examples
```julia
semi = Semidiscretization(fluid_container, boundary_container; neighborhood_search=SpatialHashingSearch, damping_coefficient=nothing)
```
"""
struct Semidiscretization{PC, RU, RV, NS, DC}
    particle_containers::PC
    ranges_u::RU
    ranges_v::RV
    neighborhood_searches::NS
    damping_coefficient::DC

    function Semidiscretization(particle_containers...; neighborhood_search=nothing,
                                damping_coefficient=nothing)
        sizes_u = [u_nvariables(container) * n_moving_particles(container)
                   for container in particle_containers]
        ranges_u = Tuple((sum(sizes_u[1:(i - 1)]) + 1):sum(sizes_u[1:i])
                         for i in eachindex(sizes_u))
        sizes_v = [v_nvariables(container) * n_moving_particles(container)
                   for container in particle_containers]
        ranges_v = Tuple((sum(sizes_v[1:(i - 1)]) + 1):sum(sizes_v[1:i])
                         for i in eachindex(sizes_v))

        # Create (and initialize) a tuple of n neighborhood searches for each of the n containers
        # We will need one neighborhood search for each pair of containers.
        searches = Tuple(Tuple(create_neighborhood_search(container, neighbor,
                                                          Val(neighborhood_search))
                               for neighbor in particle_containers)
                         for container in particle_containers)

        new{typeof(particle_containers), typeof(ranges_u), typeof(ranges_v),
            typeof(searches), typeof(damping_coefficient)}(particle_containers, ranges_u,
                                                           ranges_v, searches,
                                                           damping_coefficient)
    end
end

# Inline show function e.g. Semidiscretization(neighborhood_search=...)
function Base.show(io::IO, semi::Semidiscretization)
    @nospecialize semi # reduce precompilation time

    print(io, "Semidiscretization(")
    for container in semi.particle_containers
        print(io, container, ", ")
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
        summary_line(io, "#spatial dimensions", ndims(semi.particle_containers[1]))
        summary_line(io, "#containers", length(semi.particle_containers))
        summary_line(io, "neighborhood search",
                     semi.neighborhood_searches |> eltype |> eltype |> nameof)
        summary_line(io, "damping coefficient", semi.damping_coefficient)
        summary_line(io, "total #particles", sum(nparticles.(semi.particle_containers)))
        summary_footer(io)
    end
end

function create_neighborhood_search(_, neighbor, ::Val{nothing})
    TrivialNeighborhoodSearch(eachparticle(neighbor))
end

function create_neighborhood_search(container, neighbor, ::Val{SpatialHashingSearch})
    radius = nhs_radius(container, neighbor)
    search = SpatialHashingSearch{ndims(container)}(radius, nparticles(neighbor))

    # Initialize neighborhood search
    initialize!(search, i -> get_particle_coords(i, neighbor.initial_coordinates, neighbor))

    return search
end

@inline function nhs_radius(container, neighbor)
    return compact_support(container)
end

@inline function nhs_radius(container::Union{SolidParticleContainer,
                                             BoundaryParticleContainer},
                            neighbor)
    return nhs_radius(container, container.boundary_model, neighbor)
end

@inline function nhs_radius(container::SolidParticleContainer,
                            neighbor::SolidParticleContainer)
    return compact_support(container)
end

@inline function nhs_radius(container, model, neighbor)
    # This NHS is never used.
    # To keep actions on the tuple of NHS type-stable,
    # we still create a NHS, which is just never initialized or updated.
    return 0.0
end

@inline function nhs_radius(container, model::BoundaryModelDummyParticles, neighbor)
    return compact_support(model)
end

# Create Tuple of containers for single container
digest_containers(boundary_condition) = (boundary_condition,)
digest_containers(boundary_condition::Tuple) = boundary_condition

"""
    semidiscretize(semi, tspan)

Create an `ODEProblem` from the semidiscretization with the specified `tspan`.
"""
function semidiscretize(semi, tspan)
    @unpack particle_containers, neighborhood_searches = semi

    @assert all(container -> eltype(container) === eltype(particle_containers[1]),
                particle_containers)
    ELTYPE = eltype(particle_containers[1])

    # Initialize all particle containers
    @trixi_timeit timer() "initialize particle containers" begin for (container_index, container) in pairs(particle_containers)
        # Get the neighborhood search for this container
        neighborhood_search = neighborhood_searches[container_index][container_index]

        # Initialize this container
        initialize!(container, neighborhood_search)
    end end

    sizes_u = (u_nvariables(container) * n_moving_particles(container)
               for container in particle_containers)
    sizes_v = (v_nvariables(container) * n_moving_particles(container)
               for container in particle_containers)
    u0_ode = Vector{ELTYPE}(undef, sum(sizes_u))
    v0_ode = Vector{ELTYPE}(undef, sum(sizes_v))

    for (container_index, container) in pairs(particle_containers)
        u0_container = wrap_u(u0_ode, container_index, container, semi)
        v0_container = wrap_v(v0_ode, container_index, container, semi)

        write_u0!(u0_container, container)
        write_v0!(v0_container, container)
    end

    return DynamicalODEProblem(kick!, drift!, v0_ode, u0_ode, tspan, semi)
end

"""
    restart_with!(semi, sol)

Set the initial coordinates and velocities of all containers in `semi` to the final values
in the solution `sol`.
[`semidiscretize`](@ref) has to be called again afterwards, or another
[`Semidiscretization`](@ref) can be created with the updated containers.

# Arguments
- `semi`:   The semidiscretization
- `sol`:    The `ODESolution` returned by `solve` of `OrdinaryDiffEq`
"""
function restart_with!(semi, sol)
    @unpack particle_containers = semi

    foreach_enumerate(particle_containers) do (container_index, container)
        v_end = wrap_v(sol[end].x[1], container_index, container, semi)
        u_end = wrap_u(sol[end].x[2], container_index, container, semi)

        for particle in each_moving_particle(container)
            container.initial_coordinates[:, particle] .= u_end[:, particle]
            container.initial_velocity[:, particle] .= v_end[1:ndims(container), particle]
        end
    end

    return semi
end

# We have to pass `container` here for type stability,
# since the type of `container` determines the return type.
@inline function wrap_u(u_ode, i, container, semi)
    @unpack particle_containers, ranges_u = semi

    range = ranges_u[i]

    @boundscheck begin
        @assert length(range) ==
                u_nvariables(container) * n_moving_particles(container)
    end

    # This is a non-allocating version of:
    # return unsafe_wrap(Array{eltype(u_ode), 2}, pointer(view(u_ode, range)),
    #                    (u_nvariables(container), n_moving_particles(container)))
    return PtrArray(pointer(view(u_ode, range)),
                    (StaticInt(u_nvariables(container)), n_moving_particles(container)))
end

@inline function wrap_v(v_ode, i, container, semi)
    @unpack particle_containers, ranges_v = semi

    range = ranges_v[i]

    @boundscheck begin
        @assert length(range) ==
                v_nvariables(container) * n_moving_particles(container)
    end

    return PtrArray(pointer(view(v_ode, range)),
                    (StaticInt(v_nvariables(container)), n_moving_particles(container)))
end

function drift!(du_ode, v_ode, u_ode, semi, t)
    @unpack particle_containers = semi

    @trixi_timeit timer() "drift!" begin
        @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du_ode)

        @trixi_timeit timer() "velocity" begin
        # Set velocity and add acceleration for each container
        foreach_enumerate(particle_containers) do (container_index, container)
            du = wrap_u(du_ode, container_index, container, semi)
            v = wrap_v(v_ode, container_index, container, semi)

            @threaded for particle in each_moving_particle(container)
                # This can be dispatched per container
                add_velocity!(du, v, particle, container)
            end
        end end
    end

    return du_ode
end

@inline function add_velocity!(du, v, particle, container)
    for i in 1:ndims(container)
        du[i, particle] = v[i, particle]
    end

    return du
end

@inline add_velocity!(du, v, particle, container::BoundaryParticleContainer) = du

function kick!(dv_ode, v_ode, u_ode, semi, t)
    @unpack particle_containers, neighborhood_searches = semi

    @trixi_timeit timer() "kick!" begin
        @trixi_timeit timer() "reset ∂v/∂t" reset_du!(dv_ode)

        @trixi_timeit timer() "update containers and nhs" update_containers_and_nhs(v_ode,
                                                                                    u_ode,
                                                                                    semi, t)

        @trixi_timeit timer() "gravity and damping" gravity_and_damping!(dv_ode, v_ode,
                                                                         semi)

        @trixi_timeit timer() "container interaction" container_interaction!(dv_ode,
                                                                             v_ode, u_ode,
                                                                             semi)
    end

    return dv_ode
end

@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end

function update_containers_and_nhs(v_ode, u_ode, semi, t)
    @unpack particle_containers = semi

    # First update step before updating the NHS
    # (for example for writing the current coordinates in the solid container)
    foreach_enumerate(particle_containers) do (container_index, container)
        v = wrap_v(v_ode, container_index, container, semi)
        u = wrap_u(u_ode, container_index, container, semi)

        update1!(container, container_index, v, u, v_ode, u_ode, semi, t)
    end

    # Update NHS
    @trixi_timeit timer() "update nhs" update_nhs(u_ode, semi)

    # Second update step.
    # This is used to calculate density and pressure of the fluid containers
    # before updating the boundary containers,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_enumerate(particle_containers) do (container_index, container)
        v = wrap_v(v_ode, container_index, container, semi)
        u = wrap_u(u_ode, container_index, container, semi)

        update2!(container, container_index, v, u, v_ode, u_ode, semi, t)
    end

    # Final update step for all remaining containers
    foreach_enumerate(particle_containers) do (container_index, container)
        v = wrap_v(v_ode, container_index, container, semi)
        u = wrap_u(u_ode, container_index, container, semi)

        update3!(container, container_index, v, u, v_ode, u_ode, semi, t)
    end
end

function update_nhs(u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi

    # Update NHS for each pair of containers
    foreach_enumerate(particle_containers) do (container_index, container)
        foreach_enumerate(particle_containers) do (neighbor_index, neighbor)
            u_neighbor = wrap_u(u_ode, neighbor_index, neighbor, semi)
            neighborhood_search = neighborhood_searches[container_index][neighbor_index]

            update!(neighborhood_search,
                    nhs_coords_function(container, neighbor, u_neighbor))
        end
    end
end

function gravity_and_damping!(dv_ode, v_ode, semi)
    @unpack particle_containers, damping_coefficient = semi

    # Set velocity and add acceleration for each container
    foreach_enumerate(particle_containers) do (container_index, container)
        dv = wrap_v(dv_ode, container_index, container, semi)
        v = wrap_v(v_ode, container_index, container, semi)

        @threaded for particle in each_moving_particle(container)
            # This can be dispatched per container
            add_acceleration!(dv, particle, container)
            add_damping_force!(dv, damping_coefficient, v, particle, container)
        end
    end

    return dv_ode
end

@inline function add_acceleration!(dv, particle, container)
    @unpack acceleration = container

    for i in 1:ndims(container)
        dv[i, particle] += acceleration[i]
    end

    return dv
end

@inline add_acceleration!(dv, particle, container::BoundaryParticleContainer) = dv

@inline function add_damping_force!(dv, damping_coefficient::Float64, v, particle,
                                    container)
    for i in 1:ndims(container)
        dv[i, particle] -= damping_coefficient * v[i, particle]
    end

    return dv
end

@inline add_damping_force!(dv, ::Nothing, v, particle, container) = dv

function container_interaction!(dv_ode, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi

    # Call `interact!` for each pair of containers
    foreach_enumerate(particle_containers) do (container_index, container)
        dv = wrap_v(dv_ode, container_index, container, semi)
        v_container = wrap_v(v_ode, container_index, container, semi)
        u_container = wrap_u(u_ode, container_index, container, semi)

        foreach_enumerate(particle_containers) do (neighbor_index, neighbor)
            v_neighbor = wrap_v(v_ode, neighbor_index, neighbor, semi)
            u_neighbor = wrap_u(u_ode, neighbor_index, neighbor, semi)
            neighborhood_search = neighborhood_searches[container_index][neighbor_index]

            interact!(dv, v_container, u_container, v_neighbor, u_neighbor,
                      neighborhood_search, container, neighbor)
        end
    end

    return dv_ode
end

##### Updates

# Container update orders, see comments in update_containers_and_nhs!
function update1!(container, container_index, v, u, v_ode, u_ode, semi, t)
    return container
end

function update1!(container::SolidParticleContainer, container_index, v, u,
                  v_ode, u_ode, semi, t)
    # Only update solid containers
    update!(container, container_index, v, u, v_ode, u_ode, semi, t)
end

function update2!(container, container_index, v, u, v_ode, u_ode, semi, t)
    return container
end

function update2!(container::FluidParticleContainer, container_index, v, u,
                  v_ode, u_ode, semi, t)
    # Only update fluid containers
    update!(container, container_index, v, u, v_ode, u_ode, semi, t)
end

function update3!(container, container_index, v, u, v_ode, u_ode, semi, t)
    # Update all other containers
    update!(container, container_index, v, u, v_ode, u_ode, semi, t)
end

function update3!(container::SolidParticleContainer, container_index, v, u, v_ode, u_ode,
                  semi, t)
    @unpack boundary_model = container

    # Only update boundary model
    update!(boundary_model, container, container_index, v, u, v_ode, u_ode, semi)
end

function update3!(container::FluidParticleContainer, container_index, v, u, v_ode, u_ode,
                  semi, t)
    return container
end

# NHS updates
function nhs_coords_function(container::FluidParticleContainer,
                             neighbor::FluidParticleContainer, u)
    return i -> get_particle_coords(i, u, neighbor)
end

function nhs_coords_function(container::FluidParticleContainer,
                             neighbor::SolidParticleContainer, u)
    return i -> get_particle_coords(i, neighbor.current_coordinates, neighbor)
end

function nhs_coords_function(container::FluidParticleContainer,
                             neighbor::BoundaryParticleContainer, u)
    if neighbor.ismoving[1]
        return i -> get_particle_coords(i, neighbor.initial_coordinates, neighbor)
    end

    # Don't update
    return nothing
end

function nhs_coords_function(container::SolidParticleContainer,
                             neighbor::FluidParticleContainer, u)
    return i -> get_particle_coords(i, u, neighbor)
end

function nhs_coords_function(container::SolidParticleContainer,
                             neighbor::SolidParticleContainer, u)
    # Don't update
    return nothing
end

function nhs_coords_function(container::SolidParticleContainer,
                             neighbor::BoundaryParticleContainer, u)
    if neighbor.ismoving[1]
        return i -> get_particle_coords(i, neighbor.initial_coordinates, neighbor)
    end

    # Don't update
    return nothing
end

function nhs_coords_function(container::BoundaryParticleContainer,
                             neighbor::FluidParticleContainer, u)
    @unpack boundary_model = container

    return nhs_coords_function(container, neighbor, boundary_model, u)
end

function nhs_coords_function(container::BoundaryParticleContainer,
                             neighbor::FluidParticleContainer,
                             boundary_model, u)
    # Don't update
    return nothing
end

function nhs_coords_function(container::BoundaryParticleContainer,
                             neighbor::FluidParticleContainer,
                             boundary_model::BoundaryModelDummyParticles, u)
    return i -> get_particle_coords(i, u, neighbor)
end

function nhs_coords_function(container::BoundaryParticleContainer,
                             neighbor::SolidParticleContainer, u)
    # Don't update
    return nothing
end

function nhs_coords_function(container::BoundaryParticleContainer,
                             neighbor::BoundaryParticleContainer, u)
    # Don't update
    return nothing
end
