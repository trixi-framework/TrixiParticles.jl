# Implies number of dimensions and a field with the name `initial_coordinates`.
abstract type ParticleContainer{NDIMS} end

initialize!(container, neighborhood_search) = container
update!(container, container_index, v, u, v_ode, u_ode, semi, t) = container

@inline Base.ndims(::ParticleContainer{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(container::ParticleContainer) = eltype(container.initial_coordinates)

# Number of integrated variables in the first component of the ODE system (coordinates)
@inline u_nvariables(container) = ndims(container)

# Number of integrated variables in the second component
# of the ODE system (velocity and sometimes density)
@inline v_nvariables(container) = ndims(container)

# Number of particles in the container
@inline nparticles(container) = length(container.mass)

# Number of particles in the container whose positions are to be integrated (corresponds to the size of u and du)
@inline n_moving_particles(container) = nparticles(container)

@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline each_moving_particle(container) = Base.OneTo(n_moving_particles(container))

# This should not be dispatched by container type. We always expect to get a column of `A`.
@inline function extract_svector(A, container, i)
    extract_svector(A, Val(ndims(container)), i)
end

# Return the `i`-th column of the array `A` as an `SVector`.
@inline function extract_svector(A, ::Val{NDIMS}, i) where {NDIMS}
    return SVector(ntuple(@inline(dim->A[dim, i]), NDIMS))
end

# Return `A[:, :, i]` as an `SMatrix`.
@inline function extract_smatrix(A, container, particle)
    # Extract the matrix elements for this particle as a tuple to pass to SMatrix
    return SMatrix{ndims(container), ndims(container)}(
                                                       # Convert linear index to Cartesian index
                                                       ntuple(@inline(i->A[mod(i - 1, ndims(container)) + 1,
                                                                           div(i - 1, ndims(container)) + 1,
                                                                           particle]),
                                                              Val(ndims(container)^2)))
end

# Specifically get the current coordinates of a particle for all container types.
@inline function current_coords(u, container, particle)
    return extract_svector(current_coordinates(u, container), container, particle)
end

# This can be dispatched by container type, since for some containers, the current coordinates
# are stored in u, for others in the container itself. By default, try to extract them from u.
@inline current_coordinates(u, container) = u

# Specifically get the initial coordinates of a particle for all container types.
@inline function initial_coords(container, particle)
    return extract_svector(initial_coordinates(container), container, particle)
end

# This can be dispatched by container type.
@inline initial_coordinates(container) = container.initial_coordinates

@inline current_velocity(v, container, particle) = extract_svector(v, container, particle)

@inline function smoothing_kernel(container, distance)
    @unpack smoothing_kernel, smoothing_length = container
    return kernel(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_deriv(container, distance)
    @unpack smoothing_kernel, smoothing_length = container
    return kernel_deriv(smoothing_kernel, distance, smoothing_length)
end

@inline function smoothing_kernel_grad(container, pos_diff, distance)
    @unpack smoothing_kernel, smoothing_length = container
    return kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
end
