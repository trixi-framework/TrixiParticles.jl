initialize!(system, neighborhood_search) = system

@inline Base.ndims(::System{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(system::System) = eltype(system.initial_condition)

# Number of integrated variables in the first component of the ODE system (coordinates)
@inline u_nvariables(system) = ndims(system)

# Number of integrated variables in the second component
# of the ODE system (velocity and sometimes density)
@inline v_nvariables(system) = ndims(system)

# Number of particles in the system
@inline nparticles(system) = length(system.mass)

# Number of particles in the system whose positions are to be integrated (corresponds to the size of u and du)
@inline n_moving_particles(system) = nparticles(system)

@inline eachparticle(system) = Base.OneTo(nparticles(system))
@inline each_moving_particle(system) = Base.OneTo(n_moving_particles(system))

# This should not be dispatched by system type. We always expect to get a column of `A`.
@inline function extract_svector(A, system, i)
    extract_svector(A, Val(ndims(system)), i)
end

# Return the `i`-th column of the array `A` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    return SVector(ntuple(@inline(dim->A[dim, i]), NDIMS))
end

# Return `A[:, :, i]` as an `SMatrix`.
@inline function extract_smatrix(A, system, particle)
    # Extract the matrix elements for this particle as a tuple to pass to SMatrix
    return SMatrix{ndims(system), ndims(system)}(
                                                 # Convert linear index to Cartesian index
                                                 ntuple(@inline(i->A[mod(i - 1, ndims(system)) + 1,
                                                                     div(i - 1, ndims(system)) + 1,
                                                                     particle]),
                                                        Val(ndims(system)^2)))
end

# Specifically get the current coordinates of a particle for all system types.
@inline function current_coords(u, system, particle)
    return extract_svector(current_coordinates(u, system), system, particle)
end

# This can be dispatched by system type, since for some systems, the current coordinates
# are stored in u, for others in the system itself. By default, try to extract them from u.
@inline current_coordinates(u, system) = u

# Specifically get the initial coordinates of a particle for all system types.
@inline function initial_coords(system, particle)
    return extract_svector(initial_coordinates(system), system, particle)
end

# This can be dispatched by system type.
@inline initial_coordinates(system) = system.initial_condition.coordinates

@inline current_velocity(v, system, particle) = extract_svector(v, system, particle)

@inline function current_acceleration(system, particle)
    # TODO: Return `dv` of solid particles
    return SVector(ntuple(_ -> 0.0, Val(ndims(system))))
end

@inline function smoothing_kernel(system, distance)
    (; smoothing_kernel, smoothing_length) = system
    return kernel(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_deriv(system, distance)
    (; smoothing_kernel, smoothing_length) = system
    return kernel_deriv(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_grad(system, pos_diff, distance)
    return kernel_grad(system.smoothing_kernel, pos_diff, distance, system.smoothing_length)
end

@inline function smoothing_kernel_grad(system::BoundarySystem, pos_diff, distance)
    (; smoothing_kernel, smoothing_length) = system.boundary_model

    return kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
end

@inline function smoothing_kernel_grad(system, pos_diff, distance, particle)
    return corrected_kernel_grad(system.smoothing_kernel, pos_diff, distance,
    system.smoothing_length, system.correction, system, particle)
end

# System update orders. This can be dispatched if needed.
function update_positions!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_quantities!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_pressure!(system, v, u, v_ode, u_ode, semi, t)
    return system
end

function update_final!(system, v, u, v_ode, u_ode, semi, t)
    return system
end
