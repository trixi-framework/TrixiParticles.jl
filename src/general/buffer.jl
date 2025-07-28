struct SystemBuffer{AP, APC, EP}
    active_particle       :: AP # Vector{Bool}
    active_particle_count :: APC
    eachparticle          :: EP # Vector{Int}
    buffer_size           :: Int
end

function SystemBuffer(active_size, buffer_size::Integer)
    # Using a `BitVector` is not an option as writing to it is not thread-safe.
    active_particle = vcat(fill(true, active_size), fill(false, buffer_size))
    eachparticle = collect(eachindex(active_particle))

    return SystemBuffer(active_particle, Ref(active_size), eachparticle, buffer_size)
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

    # TODO: Parallelize (see https://github.com/trixi-framework/TrixiParticles.jl/issues/810)
    # Update the number of active particles and the active particle indices
    buffer.active_particle_count[] = count(active_particle)
    buffer.eachparticle[1:buffer.active_particle_count[]] .= findall(active_particle)

    return buffer
end

@inline each_moving_particle(system, buffer) = active_particles(system, buffer)

@inline active_coordinates(u, system, buffer) = view(u, :, active_particles(system, buffer))

@inline function active_particles(system, buffer)
    return view(buffer.eachparticle, 1:buffer.active_particle_count[])
end

@inline function deactivate_particle!(system, particle, u)
    (; active_particle) = system.buffer

    # Set particle far away from simulation domain
    for dim in 1:ndims(system)
        # Inf or NaN causes instability outcome.
        u[dim, particle] = eltype(system)(1e16)
    end

    # `deactivate_particle!` and `active_particle[particle] = true`
    # are never called on the same buffer inside a kernel,
    # so we don't have any race conditions on this `active_particle` vector.
    active_particle[particle] = false

    return system
end
