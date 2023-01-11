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
struct Semidiscretization{PC, R, NS, DC}
    particle_containers::PC
    ranges::R
    neighborhood_searches::NS
    damping_coefficient::DC

    function Semidiscretization(particle_containers...; neighborhood_search=nothing,  damping_coefficient=nothing)
        sizes = [nvariables(container) * n_moving_particles(container) for container in particle_containers]
        ranges = Tuple(sum(sizes[1:i-1])+1:sum(sizes[1:i]) for i in eachindex(sizes))

        # Create (and initialize) a tuple of n neighborhood searches for each of the n containers
        # We will need one neighborhood search for each pair of containers.
        searches = Tuple(Tuple(create_neighborhood_search(container, neighbor,
                                                          Val(neighborhood_search))
            for neighbor in particle_containers) for container in particle_containers)



        new{typeof(particle_containers), typeof(ranges), typeof(searches), typeof(damping_coefficient)}(particle_containers, ranges, searches, damping_coefficient)
    end
end


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

function Base.show(io::IO, ::MIME"text/plain", semi::Semidiscretization)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "Semidiscretization")
        summary_line(io, "#spatial dimensions", ndims(semi.particle_containers[1]))
        summary_line(io, "#containers", length(semi.particle_containers))
        summary_line(io, "neighborhood search", semi.neighborhood_searches |> eltype |> eltype |> nameof)
        summary_line(io, "damping coefficient", semi.damping_coefficient)
        summary_line(io, "total #particles", sum(nparticles.(semi.particle_containers)))
        summary_footer(io)
    end
end


create_neighborhood_search(_, neighbor, ::Val{nothing}) = TrivialNeighborhoodSearch(neighbor)


function create_neighborhood_search(container, neighbor, ::Val{SpatialHashingSearch})
    @unpack smoothing_kernel, smoothing_length = container

    radius = compact_support(smoothing_kernel, smoothing_length)
    search = SpatialHashingSearch{ndims(container)}(radius)

    # Initialize neighborhood search
    initialize!(search, neighbor.initial_coordinates, neighbor)

    return search
end


function create_neighborhood_search(container::SolidParticleContainer,
                                    neighbor::FluidParticleContainer, ::Val{SpatialHashingSearch})
    @unpack smoothing_kernel, smoothing_length = neighbor

    radius = compact_support(smoothing_kernel, smoothing_length)
    search = SpatialHashingSearch{ndims(container)}(radius)

    # Initialize neighborhood search
    initialize!(search, neighbor.initial_coordinates, neighbor)

    return search
end


function create_neighborhood_search(container::BoundaryParticleContainer, neighbor, search::Val{SpatialHashingSearch})
    @unpack boundary_model = container

    create_neighborhood_search(container, neighbor, boundary_model, search)
end

function create_neighborhood_search(container::BoundaryParticleContainer, _,
                                    boundary_model, ::Val{SpatialHashingSearch})
    # This NHS will never be used, so we just return an empty NHS.
    # To keep actions on the tuple of NHS type-stable, we return something of the same type as the other NHS.
    return SpatialHashingSearch{ndims(container)}(0.0)
end

function create_neighborhood_search(container::BoundaryParticleContainer, neighbor,
                                    boundary_model::BoundaryModelDummyParticles,
                                    ::Val{SpatialHashingSearch})
    @unpack smoothing_kernel, smoothing_length = boundary_model

    radius = compact_support(smoothing_kernel, smoothing_length)
    search = SpatialHashingSearch{ndims(container)}(radius)

    # Initialize neighborhood search
    initialize!(search, neighbor.initial_coordinates, neighbor)

    return search
end


# Create Tuple of containers for single container
digest_containers(boundary_condition) = (boundary_condition, )
digest_containers(boundary_condition::Tuple) = boundary_condition


"""
    semidiscretize(semi, tspan)

Create an `ODEProblem` from the semidiscretization with the specified `tspan`.
"""
function semidiscretize(semi, tspan)
    @unpack particle_containers, ranges, neighborhood_searches = semi

    @assert all(container -> eltype(container) === eltype(particle_containers[1]), particle_containers)
    ELTYPE = eltype(particle_containers[1])

    # Initialize all particle containers
    @pixie_timeit timer() "initialize particle containers" begin
        for (container_index, container) in pairs(particle_containers)
            # Get the neighborhood search for this container
            neighborhood_search = neighborhood_searches[container_index][container_index]

            # Initialize this container
            initialize!(container, neighborhood_search)
        end
    end

    sizes = (nvariables(container) * n_moving_particles(container) for container in particle_containers)
    u0_ode = Vector{ELTYPE}(undef, sum(sizes))

    for (container_index, container) in pairs(particle_containers)
        u0_container = wrap_array(u0_ode, container_index, container, semi)

        write_variables!(u0_container, container)
    end

    return ODEProblem(rhs!, u0_ode, tspan, semi)
end


# We have to pass `container` here for type stability,
# since the type of `container` determines the return type.
@inline function wrap_array(u_ode, i, container, semi)
    @unpack particle_containers, ranges = semi

    range = ranges[i]

    @boundscheck begin
        @assert length(range) == nvariables(container) * n_moving_particles(container)
    end

    # This is a non-allocation version of:
    # return unsafe_wrap(Array{eltype(u_ode), 2}, pointer(view(u_ode, range)),
    #                    (nvariables(container), n_moving_particles(container)))
    return PtrArray(pointer(view(u_ode, range)), (StaticInt(nvariables(container)), n_moving_particles(container)))
end


function rhs!(du_ode, u_ode, semi, t)
    @unpack particle_containers, neighborhood_searches = semi

    @pixie_timeit timer() "rhs!" begin
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du_ode)

        @pixie_timeit timer() "update containers and nhs" update_containers_and_nhs(u_ode, semi, t)

        @pixie_timeit timer() "velocity and gravity" velocity_and_gravity!(du_ode, u_ode, semi)

        @pixie_timeit timer() "container interaction" container_interaction!(du_ode, u_ode, semi)
    end

    return du_ode
end


@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end


function update_containers_and_nhs(u_ode, semi, t)
    @unpack particle_containers = semi

    # First update step before updating the NHS
    # (for example for writing the current coordinates in the solid container)
    foreach_enumerate(particle_containers) do (container_index, container)
        u = wrap_array(u_ode, container_index, container, semi)

        update1!(container, container_index, u, u_ode, semi, t)
    end

    # Update NHS
    @pixie_timeit timer() "update nhs" update_nhs(u_ode, semi)

    # Second update step.
    # This is used to calculate density and pressure of the fluid containers
    # before updating the boundary containers,
    # since the fluid pressure is needed by the Adami interpolation.
    foreach_enumerate(particle_containers) do (container_index, container)
        u = wrap_array(u_ode, container_index, container, semi)

        update2!(container, container_index, u, u_ode, semi, t)
    end

    # Final update step for all remaining containers
    foreach_enumerate(particle_containers) do (container_index, container)
        u = wrap_array(u_ode, container_index, container, semi)

        update3!(container, container_index, u, u_ode, semi, t)
    end
end


function update_nhs(u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi

    # Update NHS for each pair of containers
    foreach_enumerate(particle_containers) do (container_index, container)
        foreach_enumerate(particle_containers) do (neighbor_index, neighbor)
            u_neighbor = wrap_array(u_ode, neighbor_index, neighbor, semi)
            neighborhood_search = neighborhood_searches[container_index][neighbor_index]

            update!(neighborhood_search, u_neighbor, container, neighbor)
        end
    end
end


function velocity_and_gravity!(du_ode, u_ode, semi)
    @unpack particle_containers, damping_coefficient = semi

    # Set velocity and add acceleration for each container
    foreach_enumerate(particle_containers) do (container_index, container)
        du = wrap_array(du_ode, container_index, container, semi)
        u = wrap_array(u_ode, container_index, container, semi)

        @threaded for particle in each_moving_particle(container)
            # These can be dispatched per container
            add_velocity!(du, u, particle, container)
            add_acceleration!(du, particle, container)
            add_damping_force!(damping_coefficient, du, u, particle, container)
        end
    end

    return du_ode
end


@inline function add_velocity!(du, u, particle, container)
    for i in 1:ndims(container)
        du[i, particle] = u[i + ndims(container), particle]
    end

    return du
end

@inline add_velocity!(du, u, particle, container::BoundaryParticleContainer) = du


@inline function add_acceleration!(du, particle, container)
    @unpack acceleration = container

    for i in 1:ndims(container)
        du[i+ndims(container), particle] += acceleration[i]
    end

    return du
end

@inline add_acceleration!(du, particle, container::BoundaryParticleContainer) = du

@inline function add_damping_force!(damping_coefficient::Float64, du, u, particle, container)
    for i in 1:ndims(container)
        du[i+ndims(container), particle] -= damping_coefficient * u[i+ndims(container), particle]
    end

    return du
end

@inline add_damping_force!(::Nothing, du, u, particle, container) = 0


function container_interaction!(du_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi

    # Call `interact!` for each pair of containers
    foreach_enumerate(particle_containers) do (container_index, container)
        du = wrap_array(du_ode, container_index, container, semi)
        u_container = wrap_array(u_ode, container_index, container, semi)

        foreach_enumerate(particle_containers) do (neighbor_index, neighbor)
            u_neighbor = wrap_array(u_ode, neighbor_index, neighbor, semi)
            neighborhood_search = neighborhood_searches[container_index][neighbor_index]

            interact!(du, u_container, u_neighbor,
                      neighborhood_search, container, neighbor)
        end
    end

    return du_ode
end


##### Updates

# Container update orders, see comments in update_containers_and_nhs!
function update1!(container, container_index, u, u_ode, semi, t)
    return container
end

function update1!(container::SolidParticleContainer, container_index, u, u_ode, semi, t)
    u = wrap_array(u_ode, container_index, container, semi)

    # Only update solid containers
    update!(container, container_index, u, u_ode, semi, t)
end


function update2!(container, container_index, u, u_ode, semi, t)
    return container
end

function update2!(container::FluidParticleContainer, container_index, u, u_ode, semi, t)
    # Only update fluid containers
    update!(container, container_index, u, u_ode, semi, t)
end


function update3!(container, container_index, u, u_ode, semi, t)
    # Update all other containers
    update!(container, container_index, u, u_ode, semi, t)
end

function update3!(container::Union{SolidParticleContainer, FluidParticleContainer},
                  container_index, u, u_ode, semi, t)
    return container
end


# NHS updates
function update!(neighborhood_search, u, container::FluidParticleContainer, neighbor::FluidParticleContainer)
    update!(neighborhood_search, u, neighbor)
end

function update!(neighborhood_search, u, container::FluidParticleContainer, neighbor::SolidParticleContainer)
    update!(neighborhood_search, neighbor.current_coordinates, neighbor)
end

function update!(neighborhood_search, u, container::FluidParticleContainer, neighbor::BoundaryParticleContainer)
    if neighbor.ismoving[1]
        update!(neighborhood_search, neighbor.initial_coordinates, neighbor)
    end
end

function update!(neighborhood_search, u, container::SolidParticleContainer, neighbor::FluidParticleContainer)
    update!(neighborhood_search, u, neighbor)
end

function update!(neighborhood_search, u, container::SolidParticleContainer, neighbor::SolidParticleContainer)
    return neighborhood_search
end

function update!(neighborhood_search, u, container::SolidParticleContainer, neighbor::BoundaryParticleContainer)
    if neighbor.ismoving[1]
        update!(neighborhood_search, neighbor.initial_coordinates, neighbor)
    end
end

function update!(neighborhood_search, u, container::BoundaryParticleContainer, neighbor::FluidParticleContainer)
    @unpack boundary_model = container

    update!(neighborhood_search, u, container, neighbor, boundary_model)
end

function update!(neighborhood_search, u, container::BoundaryParticleContainer, neighbor::FluidParticleContainer,
                 boundary_model)
    return neighborhood_search
end

function update!(neighborhood_search, u, container::BoundaryParticleContainer, neighbor::FluidParticleContainer,
                 boundary_model::BoundaryModelDummyParticles)
    update!(neighborhood_search, u, neighbor)
end

function update!(neighborhood_search, u, container::BoundaryParticleContainer, neighbor::SolidParticleContainer)
    return neighborhood_search
end

function update!(neighborhood_search, u, container::BoundaryParticleContainer, neighbor::BoundaryParticleContainer)
    return neighborhood_search
end
