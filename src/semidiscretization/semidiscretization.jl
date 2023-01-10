"""
    Semidiscretization(particle_containers...; neighborhood_search=nothing)

The semidiscretization couples the passed particle containers to one simulation.

The type of neighborhood search to be used in the simulation can be specified with
the keyword argument `neighborhood_search`. A value of `nothing` means no neighborhood search.

# Examples
```julia
semi = Semidiscretization(fluid_container, boundary_container; neighborhood_search=SpatialHashingSearch)
```
"""
struct Semidiscretization{PC, R, NS}
    particle_containers::PC
    ranges::R
    neighborhood_searches::NS

    function Semidiscretization(particle_containers...; neighborhood_search=nothing)
        sizes = [nvariables(container) * n_moving_particles(container) for container in particle_containers]
        ranges = Tuple(sum(sizes[1:i-1])+1:sum(sizes[1:i]) for i in eachindex(sizes))

        # Create (and initialize) a tuple of n neighborhood searches for each of the n containers
        # We will need one neighborhood search for each pair of containers.
        searches = Tuple(Tuple(create_neighborhood_search(container, neighbor,
                                                          Val(neighborhood_search))
            for neighbor in particle_containers) for container in particle_containers)

        new{typeof(particle_containers), typeof(ranges), typeof(searches)}(particle_containers, ranges, searches)
    end
end


create_neighborhood_search(_, neighbor, ::Val{nothing}) = TrivialNeighborhoodSearch(neighbor)

function create_neighborhood_search(container::BoundaryParticleContainer, _, ::Val{SpatialHashingSearch})
    # This NHS will never be used, so we just return an empty NHS.
    # To keep actions on the tuple of NHS type-stable, we return something of the same type as the other NHS.
    return SpatialHashingSearch{ndims(container)}(0.0)
end

function create_neighborhood_search(container, neighbor, ::Val{SpatialHashingSearch})
    @unpack smoothing_kernel, smoothing_length = container

    radius = compact_support(smoothing_kernel, smoothing_length)
    search = SpatialHashingSearch{ndims(container)}(radius)

    # Initialize neighborhood search
    initialize!(search, neighbor.initial_coordinates, neighbor)

    return search
end

function create_neighborhood_search(container::SolidParticleContainer,
                                    neighbor::FluidParticleContainer,
                                    ::Val{SpatialHashingSearch})
    # Here, we need the compact support of the fluid container's smoothing kernel
    @unpack smoothing_kernel, smoothing_length = neighbor

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

    return PtrArray(pointer(view(u_ode, range)), (StaticInt(nvariables(container)), n_moving_particles(container)))
end


function rhs!(du_ode, u_ode, semi, t)
    @unpack particle_containers, neighborhood_searches = semi

    @pixie_timeit timer() "rhs!" begin
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du_ode)

        @pixie_timeit timer() "update containers" update_containers(u_ode, semi, t)

        @pixie_timeit timer() "update nhs" update_nhs(u_ode, semi)

        @pixie_timeit timer() "velocity and gravity" velocity_and_gravity!(du_ode, u_ode, semi)

        @pixie_timeit timer() "container interaction" container_interaction!(du_ode, u_ode, semi)
    end

    return du_ode
end


@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end


function update_containers(u_ode, semi, t)
    @unpack particle_containers, neighborhood_searches = semi

    foreach_enumerate(particle_containers) do (container_index, container)
        u = wrap_array(u_ode, container_index, container, semi)
        neighborhood_search = neighborhood_searches[container_index][container_index]

        update!(container, u, u_ode, neighborhood_search, semi, t)
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
    @unpack particle_containers = semi

    # Set velocity and add acceleration for each container
    foreach_enumerate(particle_containers) do (container_index, container)
        du = wrap_array(du_ode, container_index, container, semi)
        u = wrap_array(u_ode, container_index, container, semi)

        @threaded for particle in each_moving_particle(container)
            for i in 1:ndims(container)
                du[i, particle] = u[i + ndims(container), particle]
            end

            # Acceleration can be dispatched per container
            add_acceleration!(du, particle, container)
            add_damping_force!(du, u, particle, container)
        end
    end

    return du_ode
end


@inline function add_acceleration!(du, particle, container)
    @unpack acceleration = container

    for i in 1:ndims(container)
        du[i+ndims(container), particle] += acceleration[i]
    end

    return du
end


@inline function add_damping_force!(du, u, particle, container)
    @unpack damping_coefficient = container

    #damping_coefficient = 1E-3
    for i in 1:ndims(container)
        du[i+ndims(container), particle] -= damping_coefficient[] * u[i+ndims(container), particle]
    end

    return du
end


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


# NHS updates
function update!(neighborhood_search, u, container, neighbor)
    return neighborhood_search
end

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

function update!(neighborhood_search, u, container::SolidParticleContainer, neighbor::BoundaryParticleContainer)
    if neighbor.ismoving[1]
        update!(neighborhood_search, neighbor.initial_coordinates, neighbor)
    end
end

