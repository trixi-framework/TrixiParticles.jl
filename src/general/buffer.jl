struct SystemBuffer{AP, APC, EP}
    active_particle       :: AP # Vector{Bool}
    active_particle_count :: APC
    eachparticle          :: EP # Vector{Int}
    buffer_size           :: Int
end

function SystemBuffer(active_size, buffer_size::Integer)
    # Using a `BitVector` is not an option as writing to it is not thread-safe.
    # Also, to ensure thread-safe particle activation, we use an `atomic_cas` operation.
    # Thus, `active_particle` is defined as a `Vector{UInt32}` because CUDA.jl
    # does not support atomic operations on `Bool`.
    # https://github.com/JuliaGPU/CUDA.jl/blob/2cc9285676a4cd28d0846ca62f0300c56d281d38/src/device/intrinsics/atomics.jl#L243
    active_particle = vcat(fill(UInt32(true), active_size), fill(UInt32(false), buffer_size))
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
    buffer.active_particle_count[] = sum(active_particle)
    buffer.eachparticle[1:buffer.active_particle_count[]] .= findall(x -> x == true,
                                                                     active_particle)

    return buffer
end

@inline each_moving_particle(system, buffer) = active_particles(system, buffer)

@inline active_coordinates(u, system, buffer) = view(u, :, active_particles(system, buffer))

@inline function active_particles(system, buffer)
    return view(buffer.eachparticle, 1:buffer.active_particle_count[])
end

@inline function activate_next_particle(system)
    (; active_particle) = system.buffer

    for particle in eachindex(active_particle)
        if PointNeighbors.Atomix.@atomic(active_particle[particle]) == false
            # After we go into this condition, another thread might still activate this particle
            # before we do. Therefore, we use an atomic swap, which activates the particle,
            # but also returns the old value.
            # If the old value is `true`, the particle was active before and we need to continue.
            # Note: This doesn't work with Metal.jl. No error is thrown, but the operation is simply ignored.
            # An atomic compare-and-swap operation is probably implemented for Metal.jl here:
            # https://github.com/JuliaGPU/Metal.jl/blob/caf299690aa52448ee72ffc5688939b157fc1ba2/src/device/intrinsics/atomics.jl#L42
            was_active = PointNeighbors.Atomix.@atomicswap active_particle[particle] = true

            if was_active == false
                return particle
            end
        end
    end

    error("No buffer particles available")
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
