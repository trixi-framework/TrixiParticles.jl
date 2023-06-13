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

# Sorted in order of computational cost

# Use the free surface correction as used in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids" (2D: +1-2% computational time)
struct AkinciFreeSurfaceCorrection{ELTYPE}
    rho0::ELTYPE

    function AkinciFreeSurfaceCorrection(rho0)
        ELTYPE = eltype(rho0)
        return new{ELTYPE}(rho0)
    end
end

# Also referred to as 0th order correction (2D: +5-6% computational time)
struct ShepardKernelCorrection end

@inline function fluid_corrections(correction::AkinciFreeSurfaceCorrection, particle_system,
                                   rho_mean)
    return akinci_free_surface_correction(correction.rho0, rho_mean)
end

@inline function fluid_corrections(correction, particle_system, rho_mean)
    return 1.0, 1.0, 1.0
end

# Correction term for free surfaces
@inline function akinci_free_surface_correction(rho0, rho_mean)
    # at a free surface rho_mean < rho0 as such the surface tension and viscosity force are reduced
    # this is an unphysical correlation!

    # equation 4 in ref
    k = rho0 / rho_mean

    # viscosity, pressure, surface_tension
    return k, 1.0, k
end
