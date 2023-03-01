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

# This is dispatched in boundary_container.jl
@inline function particle_density(v, container, particle)
    particle_density(v, container.density_calculator, container, particle)
end

@inline function particle_density(v, ::SummationDensity, container, particle)
    @unpack cache = container

    return cache.density[particle]
end

@inline function particle_density(v, ::ContinuityDensity, container, particle)
    return v[end, particle]
end

# This is dispatched in fluid_container.jl and boundary_container.jl
@inline function hydrodynamic_mass(container, particle)
    @unpack boundary_model = container
    hydrodynamic_mass(container, boundary_model, particle)
end
