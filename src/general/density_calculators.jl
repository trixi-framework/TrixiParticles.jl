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
    ContinuityDensity(initial_densities)

# Arguments
    - `initial_densities`:    Initial density of each particle

Density calculator to integrate the density from the continuity equation
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
```
where ``\rho_a`` denotes the density of particle ``a``, ``r_a`` and ``r_b`` denote the coordinates
of particles ``a`` and ``b`` respectively, and ``v_{ab} = v_a - v_b`` is the difference of the
velocities of particles ``a`` and ``b``.
"""
struct ContinuityDensity{ELTYPE}
    initial_density::Vector{ELTYPE} # [particle]

    function ContinuityDensity(densities)
        return new{eltype(densities)}(densities)
    end
end

@inline function particle_density(v, system, particle)
    particle_density(v, system.density_calculator, system, particle)
end

@inline function particle_density(v, ::SummationDensity, system, particle)
    return system.cache.density[particle]
end

@inline function particle_density(v, ::ContinuityDensity, system, particle)
    return v[end, particle]
end

# *Note* that these functions are intended to internally set the density for buffer particles
# and density correction. It cannot be used to set up an initial condition,
# as the particle density depends on the particle positions.
@inline function set_particle_density(particle, v, system, density)
    set_particle_density(particle, v, system.density_calculator, system, density)
end

@inline function set_particle_density(particle, v, ::SummationDensity, system, density)
    system.cache.density[particle] = density
end

@inline function set_particle_density(particle, v, ::ContinuityDensity, system, density)
    v[end, particle] = density
end
