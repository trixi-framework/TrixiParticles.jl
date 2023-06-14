# Sorted in order of computational cost

@doc raw"""
Apply free surface correction to a system using the Akinci et al. method.

# Notes
- The free surface correction adjusts the viscosity, pressure, and surface tension forces near free surfaces.
- The computation time added by this method is +2-3%.

## References
- Akinci, N., Akinci, G., & Teschner, M. (2013).
  "Versatile Surface Tension and Adhesion for SPH Fluids".
  ACM Transactions on Graphics (TOG), 32(6), 182.
  [doi: 10.1145/2508363.2508405](https://doi.org/10.1145/2508363.2508405)
"""
struct AkinciFreeSurfaceCorrection{ELTYPE}
    rho0::ELTYPE

    function AkinciFreeSurfaceCorrection(rho0)
        ELTYPE = eltype(rho0)
        return new{ELTYPE}(rho0)
    end
end

@doc raw"""
Apply free surface correction to a particle system using the appropriate method.

# Arguments
- `correction`: Correction object representing the chosen correction method.
- `particle_system`: Particle system to which the correction will be applied.
- `rho_mean`: Mean density of the fluid near the free surface.

# Returns
- A tuple `(viscosity_correction, pressure_correction, surface_tension_correction)` representing the correction terms.

# Notes
- The `correction` is only implemented for `AkinciFreeSurfaceCorrection`.
- The `particle_system` argument represents the fluid particle system to which the correction is applied.
- The `rho_mean` is the mean density of the fluid near the free surface.
- This function internally calls specific correction methods to compute the correction terms.
- For `AkinciFreeSurfaceCorrection`, the correction terms are computed using the Akinci et al. method.

"""
@inline function free_surface_correction(correction::AkinciFreeSurfaceCorrection,
                                         particle_system,
                                         rho_mean)
    return akinci_free_surface_correction(correction.rho0, rho_mean)
end

@inline function free_surface_correction(correction, particle_system, rho_mean)
    return 1.0, 1.0, 1.0
end

@doc raw"""
Compute the correction terms for free surfaces following (@ref)

# Arguments
- `rho0`: Reference density of the fluid.
- `rho_mean`: Mean density of the fluid near the free surface.

# Returns
- A tuple `(viscosity_correction, pressure_correction, surface_tension_correction)` representing the correction terms.

# Notes
- The correction terms are used to adjust the viscosity, pressure, and surface tension forces near free surfaces.
- In the case of a free surface, the mean density (`rho_mean`) is typically lower than the reference density (`rho0`), resulting in reduced surface tension and viscosity forces.
- It's important to note that this correlation is unphysical and serves as an approximation.

## References
- Akinci, N., Akinci, G., & Teschner, M. (2013).
  "Versatile Surface Tension and Adhesion for SPH Fluids".
  ACM Transactions on Graphics (TOG), 32(6), 182.
  [doi: 10.1145/2508363.2508405](https://doi.org/10.1145/2508363.2508405)
"""
@inline function akinci_free_surface_correction(rho0, rho_mean)
    # equation 4 in ref
    k = rho0 / rho_mean

    # viscosity, pressure, surface_tension
    return k, 1.0, k
end

@doc raw"""
    ShepardKernelCorrection()

Kernel correction uses Shepard interpolation to obtain a 0-th order accurate result, which
was first proposed by Li et al.

The kernel correction coefficient is determined by
```math
c(x) = \sum_{b=1}^{N} V_b W_b(x)
```

This correction is applied with SummationDensity to correct the density and leads to an improvement
as especially for free surfaces.

# Notes
- Also referred to as 0th order correction (2D: +5-6% computational time)


## References:
- J. Bonet, T.-S.L. Lok.
  "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Computer Methods in Applied Mechanics and Engineering 180 (1999), pages 97-115.
  [doi: 10.1016/S0045-7825(99)00051-1](https://doi.org/10.1016/S0045-7825(99)00051-1)
- Mihai Basa, Nathan Quinlan, Martin Lastiwka.
  "Robustness and accuracy of SPH formulations for viscous flow".
  In: International Journal for Numerical Methods in Fluids 60 (2009), pages 1127-1148.
  [doi: 10.1002/fld.1927](https://doi.org/10.1002/fld.1927)
- S.F. Li, W.K. Liu, "Moving least square Kernel Galerkin method (II) Fourier analysis",
  Computer Methods in Applied Mechanics and Engineering., 139 (1996) pages 159ff
  [doi:10.1016/S0045-7825(96)01082-1] (https://doi.org/10.1016/S0045-7825(96)01082-1).
"""
struct ShepardKernelCorrection end
