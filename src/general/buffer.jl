struct SystemBuffer{V}
    active_particle :: BitVector
    eachparticle    :: V # Vector{Int}
    buffer_size     :: Int

    function SystemBuffer(active_size, buffer_size::Integer)
        active_particle = vcat(trues(active_size), falses(buffer_size))
        eachparticle = collect(1:active_size)

        return new{typeof(eachparticle)}(active_particle, eachparticle, buffer_size)
    end
end

allocate_buffer(initial_condition, buffer) = initial_condition

function allocate_buffer(initial_condition, buffer::SystemBuffer)
    (; buffer_size) = buffer

    # Initialize particles far away from simulation domain
    coordinates = fill(1e16, ndims(initial_condition), buffer_size)

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

# `view(eachindex(buffer.active_particle), buffer.active_particle)` is a allocation
# (but thread supporting) version of:
# `(i for i in eachindex(buffer.active_particle) if buffer.active_particle[i])`
# TODO: Find a non-allocation version

# This is also a allocation version but only in every `update!(buffer)` call
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

    next_particle = findfirst(active_particle)

    if isnothing(next_particle)
        error("0 out of $(system.buffer.buffer_size) buffer particles available")
    end

    active_particle[next_particle] = true

    return next_particle
end

@inline function deactivate_particle!(system, particle, u)
    (; active_particle) = system.buffer

    active_particle[particle] = false

    # Set particle far away from simulation domain
    for dim in 1:ndims(system)
        # Inf or NaN causes instability outcome.
        u[dim, particle] = 1e16
    end

    return system
end
