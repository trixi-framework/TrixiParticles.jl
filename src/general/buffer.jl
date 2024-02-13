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

    if all(rho -> rho ≈ density[1], density)
        density = initial_condition.density[1]
    else
        throw(ArgumentError("`density` needs to be constant when using a `SystemBuffer`"))
    end

    particle_spacing = initial_condition.particle_spacing

    buffer_ic = InitialCondition{NDIMS}(coordinates, density, particle_spacing)

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

@inline eachparticle(system, ::Nothing) = Base.OneTo(nparticles(system))

@inline eachparticle(system, buffer) = buffer.eachparticle

@inline each_moving_particle(system, ::Nothing) = Base.OneTo(n_moving_particles(system))

@inline each_moving_particle(system, buffer) = buffer.eachparticle

@inline active_coordinates(u, system, ::Nothing) = current_coordinates(u, system)

@inline active_coordinates(u, system, buffer) = view(u, :, buffer.active_particle)
