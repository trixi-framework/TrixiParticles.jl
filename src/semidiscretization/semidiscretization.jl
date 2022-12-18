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
create_neighborhood_search(::BoundaryParticleContainer, _, ::Val{SpatialHashingSearch}) = nothing
create_neighborhood_search(::BoundaryParticleContainer, _, ::Val{nothing}) = nothing

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
        u0_container = wrap_array(u0_ode, container_index, semi)

        write_variables!(u0_container, container)
    end

    return ODEProblem(rhs!, u0_ode, tspan, semi)
end


@inline function wrap_array(u_ode, i, semi)
    @unpack particle_containers, ranges = semi

    range = ranges[i]
    container = particle_containers[i]

    @boundscheck begin
        @assert length(range) == nvariables(container) * n_moving_particles(container)
    end

    return unsafe_wrap(Array{eltype(u_ode), 2}, pointer(view(u_ode, range)),
                       (nvariables(container), n_moving_particles(container)))
end


function rhs!(du_ode, u_ode, semi, t)
    @unpack particle_containers, neighborhood_searches = semi

    @pixie_timeit timer() "rhs!" begin
        # Reset du
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du_ode)

        # Update all particle containers
        @pixie_timeit timer() "update particle containers" for (container_index, container) in pairs(particle_containers)
            neighborhood_search = neighborhood_searches[container_index][container_index]
            u = wrap_array(u_ode, container_index, semi)

            update!(container, u, u_ode, neighborhood_search, semi)
        end

        # Update all neighborhood searches
        @pixie_timeit timer() "update neighborhood searches" for (container_index, container) in pairs(particle_containers)
            for (neighbor_index, neighbor) in pairs(particle_containers)
                neighborhood_search = neighborhood_searches[container_index][neighbor_index]
                u_neighbor = wrap_array(u_ode, neighbor_index, semi)

                update!(neighborhood_search, u_neighbor, container, neighbor)
            end
        end

        @pixie_timeit timer() "main loop" for (particle_container_index, particle_container) in pairs(particle_containers)
            du = wrap_array(du_ode, particle_container_index, semi)
            u_particle_container = wrap_array(u_ode, particle_container_index, semi)

            # Set velocity and add acceleration
            @threaded for particle in each_moving_particle(particle_container)
                for i in 1:ndims(particle_container)
                    du[i, particle] = u_particle_container[i + ndims(particle_container), particle]
                end

                add_acceleration!(du, particle, particle_container)
                add_damping_force!(u_particle_container, du, particle, particle_container)
            end

            # Neighbor interaction
            for (neighbor_container_index, neighbor_container) in pairs(particle_containers)
                u_neighbor_container = wrap_array(u_ode, neighbor_container_index, semi)

                interact!(du, u_particle_container, u_neighbor_container,
                          neighborhood_searches[particle_container_index][neighbor_container_index],
                          particle_container, neighbor_container)
            end
        end
    end

    return du_ode
end


function update!(neighborhood_search, u, container, neighbor)
    return neighborhood_search
end

function update!(neighborhood_search, u, container::FluidParticleContainer, neighbor::FluidParticleContainer)
    update!(neighborhood_search, u, neighbor)
end

function update!(neighborhood_search, u, container::FluidParticleContainer, neighbor::SolidParticleContainer)
    update!(neighborhood_search, neighbor.current_coordinates, neighbor)
end

function update!(neighborhood_search, u, container::SolidParticleContainer, neighbor::FluidParticleContainer)
    update!(neighborhood_search, u, neighbor)
end


@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end


@inline function add_acceleration!(du, particle, container)
    @unpack acceleration = container

    for i in 1:ndims(container)
        du[i+ndims(container), particle] += acceleration[i]
    end

    return du
end

@inline function add_damping_force!(u, du, particle, container)
    @unpack damping_coefficient = container

    #damping_coefficient = 1E-3
    for i in 1:ndims(container)
        du[i+ndims(container), particle] -= damping_coefficient[] * u[i+ndims(container), particle]
    end

    return du
end
