# Implies number of dimensions and a field with the name `initial_coordinates`.
abstract type ParticleContainer{NDIMS} end

struct DefaultStore end
struct StoreAll end

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

# Specifically get the current coordinates of a particle for all container types.
@inline function get_current_coords(particle, u, container)
    get_particle_coords(particle, current_coordinates(u, container), container)
end

# This can be dispatched by container types, since for some containers, the current coordinates
# are stored in u, for others in the container itself. By default, try to extract them from u.
@inline current_coordinates(u, container) = u

# Return the `particle`-th column of the array `field`.
# This should not be dispatched by container type. We always expect to get a column of the array `field`.
@inline function get_particle_coords(particle, field, container)
    return SVector(ntuple(@inline(dim->field[dim, particle]), Val(ndims(container))))
end

@inline function get_particle_vel(particle, v, container)
    return get_particle_coords(particle, v, container)
end

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

@inline function compact_support(container)
    @unpack smoothing_kernel, smoothing_length = container
    return compact_support(smoothing_kernel, smoothing_length)
end

include("fluid_container.jl")
include("solid_container.jl")
include("boundary_container.jl") # This depends on fluid and solid containers
