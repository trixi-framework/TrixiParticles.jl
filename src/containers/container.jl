abstract type ParticleContainer{NDIMS} end

initialize!(container, neighborhood_search) = container
update!(container, container_index, u, u_ode, semi, t) = container

@inline Base.ndims(::ParticleContainer{NDIMS}) where NDIMS = NDIMS

# Number of integrated variables in the ODE system (coordinates, velocity, sometimes density)
@inline nvariables(container) = 2 * ndims(container)

# Number of particles in the container
@inline nparticles(container) = length(container.mass)

# Number of particles in the container whose positions are to be integrated (corresponds to the size of u and du)
@inline n_moving_particles(container) = nparticles(container)

@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline each_moving_particle(container) = Base.OneTo(n_moving_particles(container))
@inline Base.eltype(container::ParticleContainer) = eltype(container.mass)

# Specifically get the current coordinates of a particle for all container types.
# This can be dispatched by container types, since for some containers, the current coordinates
# are stored in u, for others in the container itself. By default, try to extract them from u.
@inline get_current_coords(particle, u, container) = get_particle_coords(particle, u, container)

# Return the `particle`-th column of the array `coords`.
# This should not be dispatched by container type. We always expect to get a column of the array `coords`.
@inline function get_particle_coords(particle, coords, container)
    return SVector(ntuple(@inline(dim -> coords[dim, particle]), Val(ndims(container))))
end

@inline function get_particle_vel(particle, u, container)
    return SVector(ntuple(@inline(dim -> u[dim + ndims(container), particle]), Val(ndims(container))))
end


include("fluid_container.jl")
include("solid_container.jl")
include("boundary_container.jl") # This depends on fluid and solid containers
