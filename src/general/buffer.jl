struct SystemBuffer{}
    active_particle :: BitVector
    eachparticle    :: Vector{Int}
    buffer_size     :: Int

    function SystemBuffer(active_size, buffer_size)
        if !(buffer_size isa Int)
            throw(ArgumentError("invalid buffer: $buffer_size of type $(eltype(buffer_size))"))
        end

        active_particle = vcat(trues(active_size), falses(buffer_size))
        eachparticle = collect(1:active_size)

        return new{}(active_particle, eachparticle, buffer_size)
    end
end

# See comments in `each_moving_particle()` why update is needed
@inline function update!(buffer::SystemBuffer)
    (; active_particle) = buffer

    new_eachparticle = [i for i in eachindex(active_particle) if active_particle[i]]
    resize!(buffer.eachparticle, length(new_eachparticle))

    buffer.eachparticle .= new_eachparticle
end

@inline update!(buffer::Nothing) = buffer

allocate_buffer(initial_condition, buffer) = initial_condition

function allocate_buffer(initial_condition, buffer::SystemBuffer)
    (; buffer_size) = buffer
    coordinates = rand(100.0:0.05:200.0, ndims(initial_condition), buffer_size)
    velocities = rand(100.0:0.05:200.0, ndims(initial_condition), buffer_size)
    masses = initial_condition.mass[1] * ones(buffer_size)
    densities = initial_condition.density[1] * ones(buffer_size)
    pressure = initial_condition.pressure[1] * ones(buffer_size)
    particle_spacing = initial_condition.particle_spacing

    buffer_ic = InitialCondition(coordinates, velocities, masses, densities,
                                 pressure=pressure, particle_spacing=particle_spacing)
    return union(initial_condition, buffer_ic)
end

@inline function each_moving_particle(system, buffer::SystemBuffer)

    # `view(eachindex(buffer.active_particle), buffer.active_particle)` is a allocation
    # (but thread supporting) version of:
    # `(i for i in eachindex(buffer.active_particle) if buffer.active_particle[i])`
    # TODO: Find a non-allocation version

    # This is also a allocation version but only in every `update!()` call
    return buffer.eachparticle
end
