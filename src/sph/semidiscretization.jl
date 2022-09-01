@doc raw"""
    SummationDensity()

Density calculator to use the summation formula (considering ``\rho`` as a normal scalar field)
```math
\rho(r) = \sum_{b} m_b W(\Vert r - r_b \Vert, h),
```
for the density estimation,
where ``r_b`` denotes the coordinates and ``m_b`` the mass of particle ``b``.

!!! note "Multiple fluids"
    Note, when the systems involves multiple fluids with large density ratios (``\geq 2``) in contact, the following expression is more accurate.
    ```math
    \frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \rho_a \sum_{b} \frac{m_b}{\rho_a} v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
    ```    
    References:
    - Joseph J. Monaghan. "Smoothed particle hydrodynamics".
      In: Reports on Progress in Physics, Volume 68, Number 8 (2005), pages 1703–1759.
      [doi: 10.1088/0034-4885/68/8/R01](http://dx.doi.org/10.1088/0034-4885/68/8/R01)
"""
struct SummationDensity end

@doc raw"""
    ContinuityDensity()

Density calculator to integrate the density from the continuity equation (SPH approximation of  continuity equation)
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
```
where ``\rho_a`` denotes the density of particle ``a``, ``r_a`` and ``r_b`` denote the coordinates
of particles ``a`` and ``b`` respectively, and ``v_{ab} = v_a - v_b`` is the difference of the
velocities of particles ``a`` and ``b``.

"""
struct ContinuityDensity end

struct WCSPHSemidiscretization{NDIMS, ELTYPE<:Real, DC, SE, K, V, BC, NS, C}
    density_calculator  ::DC
    state_equation      ::SE
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    viscosity           ::V
    boundary_conditions ::BC
    gravity             ::SVector{NDIMS, ELTYPE}
    neighborhood_search ::NS
    cache               ::C

    function WCSPHSemidiscretization{NDIMS}(particle_masses,
                                          density_calculator, state_equation,
                                          smoothing_kernel, smoothing_length;
                                          viscosity=NoViscosity(),
                                          boundary_conditions=nothing,
                                          gravity=ntuple(_ -> 0.0, Val(NDIMS)),
                                          neighborhood_search=nothing) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make boundary_conditions a tuple
        boundary_conditions_ = digest_boundary_conditions(boundary_conditions)

        # Make gravity an SVector
        gravity_ = SVector(gravity...)

        cache = (; create_cache(particle_masses, density_calculator, ELTYPE, nparticles)...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(boundary_conditions_),
                   typeof(neighborhood_search), typeof(cache)}(
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, boundary_conditions_, gravity_, neighborhood_search, cache)
    end
end



struct EISPHSemidiscretization{NDIMS, ELTYPE<:Real, DC, PPE, K, V, BC, NS, C}
    density_calculator  ::DC
    pressure_poisson_eq :: PPE
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    viscosity           ::V
    boundary_conditions ::BC
    gravity             ::SVector{NDIMS, ELTYPE}
    neighborhood_search ::NS
    cache               ::C

    function EISPHSemidiscretization{NDIMS}(particle_masses,
                                          density_calculator, pressure_poisson_eq,
                                          smoothing_kernel, smoothing_length;
                                          viscosity=NoViscosity(),
                                          boundary_conditions=nothing,
                                          gravity=ntuple(_ -> 0.0, Val(NDIMS)),
                                          neighborhood_search=nothing) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make boundary_conditions a tuple
        boundary_conditions_ = digest_boundary_conditions(boundary_conditions)

        # Make gravity an SVector
        gravity_ = SVector(gravity...)

        cache = (; create_cache(particle_masses, density_calculator, ELTYPE, nparticles)...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(pressure_poisson_eq),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(boundary_conditions_),
                   typeof(neighborhood_search), typeof(cache)}(
            density_calculator, pressure_poisson_eq, smoothing_kernel, smoothing_length,
            viscosity, boundary_conditions_, gravity_, neighborhood_search, cache)
    end
end


function create_cache(mass, density_calculator, eltype, nparticles)
    pressure = Vector{eltype}(undef, nparticles)
    return (; mass, pressure, create_cache(density_calculator, eltype, nparticles)...)
end

function create_cache(::SummationDensity, eltype, nparticles)
    density = Vector{eltype}(undef, nparticles)

    return (; density)
end

function create_cache(::ContinuityDensity, eltype, nparticles)
    return (; )
end

