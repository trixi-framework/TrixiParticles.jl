@doc raw"""
    ParticleContainer
"""
abstract type ParticleContainer{NDIMS} end

@inline Base.ndims(::ParticleContainer{NDIMS}) where NDIMS = NDIMS

@inline nvariables(container) = 2 * ndims(container)
@inline nparticles(container) = length(container.mass)
@inline n_moving_particles(container) = nparticles(container)
@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline each_moving_particle(container) = Base.OneTo(n_moving_particles(container))
@inline Base.eltype(container::ParticleContainer) = eltype(container.mass)


@inline get_current_coords(particle, u, container) = get_particle_coords(particle, u, container)

@inline function get_particle_coords(particle, coords, container)
    return SVector(ntuple(@inline(dim -> coords[dim, particle]), Val(ndims(container))))
end

@inline function get_particle_vel(particle, u, container)
    return SVector(ntuple(@inline(dim -> u[dim + ndims(container), particle]), Val(ndims(container))))
end


@doc raw"""
    SummationDensity()

Density calculator to use the summation formula
```math
\rho(r) = \sum_{b} m_b W(\Vert r - r_b \Vert, h),
```
for the density estimation,
where ``r_b`` denotes the coordinates and ``m_b`` the mass of particle ``b``.
"""
struct SummationDensity end

@doc raw"""
    ContinuityDensity()

Density calculator to integrate the density from the continuity equation
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
```
where ``\rho_a`` denotes the density of particle ``a``, ``r_a`` and ``r_b`` denote the coordinates
of particles ``a`` and ``b`` respectively, and ``v_{ab} = v_a - v_b`` is the difference of the
velocities of particles ``a`` and ``b``.
"""
struct ContinuityDensity end


@inline get_particle_density(particle, u, container) = get_particle_density(particle, u, container.density_calculator, container)

@inline function get_particle_density(particle, u, ::SummationDensity, container)
    @unpack cache = container

    return cache.density[particle]
end

@inline function get_particle_density(particle, u, ::ContinuityDensity, container)
    return u[end, particle]
end


include("fluid_container.jl")
include("solid_container.jl")
include("boundary_container.jl")
