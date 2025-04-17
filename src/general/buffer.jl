struct SystemBuffer{V}
    active_particle :: Vector{Bool}
    eachparticle    :: V # Vector{Int}
    buffer_size     :: Int

    function SystemBuffer(active_size, buffer_size::Integer)
        # We cannot use a `BitVector` here, as writing to a `BitVector` is not thread-safe
        active_particle = vcat(fill(true, active_size), fill(false, buffer_size))
        eachparticle = collect(1:active_size)

        return new{typeof(eachparticle)}(active_particle, eachparticle, buffer_size)
    end
end

allocate_buffer(initial_condition, ::Nothing) = initial_condition

function allocate_buffer(initial_condition, buffer::SystemBuffer)
    (; buffer_size) = buffer

    # Initialize particles far away from simulation domain
    coordinates = fill(eltype(initial_condition)(1e-16), ndims(initial_condition),
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

@inline update_system_buffer!(buffer::Nothing) = buffer

# TODO `resize` allocates. Find a non-allocating version
@inline function update_system_buffer!(buffer::SystemBuffer)
    (; active_particle) = buffer

    resize!(buffer.eachparticle, count(active_particle))

    i = 1
    for j in eachindex(active_particle)
        if active_particle[j]
            buffer.eachparticle[i] = j
            i += 1
        end
    end

    return buffer
end

@inline each_moving_particle(system, buffer) = buffer.eachparticle

@inline active_coordinates(u, system, buffer) = view(u, :, buffer.active_particle)

@inline active_particles(system, buffer) = buffer.eachparticle

@inline function activate_next_particle(system)
    (; active_particle) = system.buffer

    for particle in eachindex(active_particle)
        if !active_particle[particle]
            # Activate this particle. The return value is the old value.
            # If this is `true`, the particle was active before and we need to continue.
            # This happens because a particle might have been activated by another thread
            # between the condition and the line below.
            was_active = PointNeighbors.Atomix.@atomicswap active_particle[particle] = true

            !was_active && return particle
        end
    end

    error("0 out of $(system.buffer.buffer_size) buffer particles available")
end

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
