struct SystemBuffer{}
    active_particle :: BitVector
    eachparticle    :: Vector{Int}
    buffer_size     :: Int

    function SystemBuffer(active_size, buffer_size)
        if !(buffer_size isa Int)
            throw(ArgumentError("`buffer_size` must be of type Int"))
        end

        active_particle = vcat(trues(active_size), falses(buffer_size))
        eachparticle = collect(1:active_size)

        return new{}(active_particle, eachparticle, buffer_size)
    end
end

allocate_buffer(initial_condition, buffer) = initial_condition

function allocate_buffer(initial_condition, buffer::SystemBuffer)
    (; buffer_size) = buffer
    NDIMS = ndims(initial_condition)

    # Initialize particles far away from simulation domain
    coordinates = inv(eps()) * ones(ndims(initial_condition), buffer_size)

    if all(rho -> rho ≈ initial_condition.density[1], initial_condition.density)
        density = initial_condition.density[1]
    else
        throw(ArgumentError("`density` needs to be constant when using `SystemBuffer`"))
    end

    particle_spacing = initial_condition.particle_spacing

    buffer_ic = InitialCondition(; coordinates, density, particle_spacing)

    return union(initial_condition, buffer_ic)
end

@inline update!(buffer::Nothing) = buffer

# `view(eachindex(buffer.active_particle), buffer.active_particle)` is a allocation
# (but thread supporting) version of:
# `(i for i in eachindex(buffer.active_particle) if buffer.active_particle[i])`
# TODO: Find a non-allocation version

# This is also a allocation version but only in every `update!(buffer)` call
@inline function update!(buffer::SystemBuffer)
    (; active_particle) = buffer

    new_eachparticle = [i for i in eachindex(active_particle) if active_particle[i]]
    resize!(buffer.eachparticle, length(new_eachparticle))

    buffer.eachparticle .= new_eachparticle

    return buffer
end

@inline each_moving_particle(system, ::Nothing) = Base.OneTo(n_moving_particles(system))

@inline each_moving_particle(system, buffer) = buffer.eachparticle

@inline active_coordinates(u, system, ::Nothing) = current_coordinates(u, system)

@inline active_coordinates(u, system, buffer) = view(u, :, buffer.active_particle)

@inline active_particles(system, buffer) = buffer.eachparticle

@inline function available_particle(system)
    (; active_particle) = system.buffer

    for particle in eachindex(active_particle)
        if !active_particle[particle]
            active_particle[particle] = true

            return particle
        end
    end

    error("0 out of $(system.buffer.buffer_size) buffer particles available")
end

@inline function deactivate_particle!(system, particle, u)
    (; active_particle) = system.buffer

    active_particle[particle] = false

    # Set particle far away from simulation domain
    for dim in 1:ndims(system)
        # Inf or NaN causes instability outcome.
        u[dim, particle] = inv(eps())
    end

    return system
end
