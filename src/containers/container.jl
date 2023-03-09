abstract type ParticleContainer{NDIMS} end

struct DefaultStore end
struct StoreAll end

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

# Specifically get the current coordinates of a particle for all container types.
# This can be dispatched by container types, since for some containers, the current coordinates
# are stored in u, for others in the container itself. By default, try to extract them from u.
@inline function get_current_coords(particle, u, container)
    get_vec_field(particle, u, container)
end

# Return the `particle`-th column of the array `field`.
# This should not be dispatched by container type. We always expect to get a column of the array `field`.
@inline function get_vec_field(particle, field, container)
    return SVector(ntuple(@inline(dim->field[dim, particle]), Val(ndims(container))))
end

@inline function get_particle_vel(particle, v, container)
    return get_vec_field(particle, v, container)
end

struct State{ELTYPE}
    density     :: ELTYPE
    pressure    :: ELTYPE
    temperature :: ELTYPE

    function State(density, pressure, temperature)
        ELTYPE = typeof(density)
        return new{ELTYPE}(density, pressure, temperature)
    end
end

include("fluid_container.jl")
include("solid_container.jl")
include("boundary_container.jl") # This depends on fluid and solid containers
