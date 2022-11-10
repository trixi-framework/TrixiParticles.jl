struct Semidiscretization{PC, R}
    particle_containers::PC
    ranges::R

    function Semidiscretization(particle_containers...)
        sizes = [nvariables(container) * n_moving_particles(container) for container in particle_containers]
        ranges = Tuple(sum(sizes[1:i-1])+1:sum(sizes[1:i]) for i in eachindex(sizes))

        new{typeof(particle_containers), typeof(ranges)}(particle_containers, ranges)
    end
end

# Create Tuple of containers for single container
digest_containers(boundary_condition) = (boundary_condition, )
digest_containers(boundary_condition::Tuple) = boundary_condition


function semidiscretize(semi, tspan)
    @unpack particle_containers, ranges = semi

    @assert all(container -> eltype(container) === eltype(particle_containers[1]), particle_containers)
    ELTYPE = eltype(particle_containers[1])

    # Initialize all particle containers
    @pixie_timeit timer() "initialize particle containers" for container in particle_containers
        initialize!(container)
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
    @unpack particle_containers = semi

    @pixie_timeit timer() "rhs!" begin
        # Reset du
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du_ode)

        # Update all particle containers
        @pixie_timeit timer() "update particle containers" for (container_index, container) in pairs(particle_containers)
            u = wrap_array(u_ode, container_index, semi)
            update!(container, u, u_ode, semi)
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
            end

            # Neighbor interaction
            for (neighbor_container_index, neighbor_container) in pairs(particle_containers)
                u_neighbor_container = wrap_array(u_ode, neighbor_container_index, semi)

                calc_du!(du, u_particle_container, u_neighbor_container, particle_container, neighbor_container)
            end
        end
    end

    return du_ode
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
