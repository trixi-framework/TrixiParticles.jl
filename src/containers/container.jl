abstract type ParticleContainer{NDIMS} end

initialize!(container, neighborhood_search) = container
update!(container, container_index, v, u, v_ode, u_ode, semi, t) = container

@inline Base.ndims(::ParticleContainer{NDIMS}) where {NDIMS} = NDIMS

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
@inline Base.eltype(container::ParticleContainer) = eltype(container.initial_coordinates)

# Return the `i`-th column of the array `A` as an `SVector`.
# This should not be dispatched by container type. We always expect to get a column of `A`.
@inline function extract_svector(A, container, i)
    return SVector(ntuple(@inline(dim->A[dim, i]), Val(ndims(container))))
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
# This can be dispatched by container types, since for some containers, the current coordinates
# are stored in `u`, for others in the container itself.
# By default, try to extract them from `u`.
@inline current_coords(u, container, particle) = extract_svector(u, container, particle)
@inline current_velocity(v, container, particle) = extract_svector(v, container, particle)

include("fluid_container.jl")
include("solid_container.jl")
include("boundary_container.jl") # This depends on fluid and solid containers
