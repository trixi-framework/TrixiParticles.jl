struct SystemBuffer{VB, RI, VI, NP}
    active_particle       :: VB  # Vector{Bool}
    active_particle_count :: RI  # Ref{Int}
    candidates            :: VI  # Vector{Int}
    particle_outside      :: VB  # Vector{Bool}
    available_particles   :: VI  # Vector{Int}
    next_particle         :: NP  # Vector{Int32}
    eachparticle          :: VI  # Vector{Int}
    buffer_size           :: Int
end

function SystemBuffer(active_size, buffer_size::Integer)
    # We cannot use a `BitVector` here, as writing to a `BitVector` is not thread-safe
    active_particle = vcat(fill(true, active_size), fill(false, buffer_size))
    candidates = collect(eachindex(active_particle))
    particle_outside = vcat(fill(false, active_size), fill(true, buffer_size))
    available_particles = collect(eachindex(active_particle))
    eachparticle = collect(eachindex(active_particle))

    return SystemBuffer(active_particle, Ref(active_size), candidates, particle_outside,
                        available_particles, Int32[1], eachparticle, buffer_size)
end

allocate_buffer(initial_condition, ::Nothing) = initial_condition

function allocate_buffer(initial_condition, buffer::SystemBuffer)
    (; buffer_size) = buffer

    # Initialize particles far away from simulation domain
    coordinates = fill(eltype(initial_condition)(1e16), ndims(initial_condition),
                       buffer_size)

    if all(rho -> isapprox(rho, first(initial_condition.density), atol=eps(), rtol=eps()),
           initial_condition.density)
        density = first(initial_condition.density)
    else
        throw(ArgumentError("`initial_condition.density` needs to be constant when using `SystemBuffer`"))
    end

    particle_spacing = initial_condition.particle_spacing

    buffer_ic = InitialCondition(; coordinates, density, particle_spacing)

    return union(initial_condition, buffer_ic)
end

@inline update_system_buffer!(buffer::Nothing, semi) = buffer

# TODO `resize` allocates. Find a non-allocating version
@inline function update_system_buffer!(buffer::SystemBuffer, semi)
    (; active_particle) = buffer

    buffer.active_particle_count[] = count(active_particle)
    buffer.eachparticle .= -1

    @threaded semi for i in 1:buffer.active_particle_count[]
        active = 0
        for j in eachindex(active_particle)
            if active_particle[j]
                active += 1
                if active == i
                    buffer.eachparticle[i] = j
                    break
                end
            end
        end
    end

    return buffer
end

@inline each_moving_particle(system,
                             buffer) = view(buffer.eachparticle,
                                            1:buffer.active_particle_count[])

@inline active_coordinates(u, system, buffer) = view(u, :, buffer.active_particle)

@inline active_particles(system,
                         buffer) = view(buffer.eachparticle,
                                        1:buffer.active_particle_count[])

@inline function deactivate_particle!(system, particle, u)
    (; active_particle) = system.buffer

    # Set particle far away from simulation domain
    for dim in 1:ndims(system)
        # Inf or NaN causes instability outcome.
        u[dim, particle] = eltype(system)(1e16)
    end

    # To ensure thread safety, the buffer particle is only released for reuse
    # after the write operation (`u`) has been completed.
    # This guarantees that no other thread can access the active particle prematurely,
    # avoiding race conditions.
    active_particle[particle] = false

    return system
end
