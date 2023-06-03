struct SystemBuffer{}
    active_particle :: BitVector
    eachparticle    :: Vector{Int}
    buffer_size     :: Int

    function SystemBuffer(active_size, buffer_size)
        active_particle = vcat(trues(active_size), falses(buffer_size))
        eachparticle = collect(1:active_size)

        return new{}(active_particle, eachparticle, buffer_size)
    end
end

# See comments in `each_moving_particle()` why update is needed
@inline function update!(buffer::SystemBuffer)
    @unpack active_particle = buffer

    new_eachparticle = [i for i in eachindex(active_particle) if active_particle[i]]
    resize!(buffer.eachparticle, length(new_eachparticle))

    buffer.eachparticle .= new_eachparticle
end

function allocate_buffer!(coords, velocities, masses, densities, buffer)
    return coords, velocities, masses, densities, pressure
end

function allocate_buffer!(coordinates, velocity, mass, density, buffer::SystemBuffer)
    @unpack buffer_size = buffer

    coordinates = hcat(coordinates, similar(coordinates, size(coordinates, 1), buffer_size))
    velocity = hcat(velocity, similar(velocity, size(velocity, 1), buffer_size))
    mass = mass[1] * ones(eltype(mass), length(mass) + buffer_size)
    density = density[1] * ones(eltype(density), length(density) + buffer_size)

    return coordinates, velocity, mass, density
end

@inline function each_moving_particle(system, buffer::SystemBuffer)

    # `view(eachindex(buffer.active_particle), buffer.active_particle)` is a allocation
    # (but thread supporting) version of:
    # `(i for i in eachindex(buffer.active_particle) if buffer.active_particle[i])`
    # TODO: Find a non-allocation version

    # This is also a allocation version but only in every `update!()` call
    return buffer.eachparticle
end
