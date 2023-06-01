@doc raw"""
    KernelCorrection()
    KernelGradientCorrection()

Kernel correction uses Shepard interpolation to obtain a 0-th order accurate result.

## References:
- J. Bonet, T.-S.L. Lok.
  "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Computer Methods in Applied Mechanics and Engineering 180 (1999), pages 97-115.
  [doi: 10.1016/S0045-7825(99)00051-1](https://doi.org/10.1016/S0045-7825(99)00051-1)
- Mihai Basa, Nathan Quinlan, Martin Lastiwka.
  "Robustness and accuracy of SPH formulations for viscous flow".
  In: International Journal for Numerical Methods in Fluids 60 (2009), pages 1127-1148.
  [doi: 10.1002/fld.1927](https://doi.org/10.1002/fld.1927)
"""

# sorted in order of computational cost

# also referred to as 0th order correction (cheapest)
struct KernelCorrection end

# Use the free surface correction as used in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
struct AkinciFreeSurfaceCorrection end

# number of correction values
@inline ncvals(::Any) = 3

@inline function fluid_corrections(::AkinciFreeSurfaceCorrection, particle_container,
                                   rho_mean)
    return akinci_free_surface_correction(particle_container, rho_mean)
end

@inline function fluid_corrections(::Any, particle_container, rho_mean)
    return ones(SVector{ncvals(particle_container), eltype(particle_container)})
end

# correction term for free surfaces
@inline function akinci_free_surface_correction(particle_container, rho_mean)
    # at a free surface rho_mean < rho0 as such the surface tension and viscosity force are reduced
    # this is an unphysical correlation!

    # equation 4 in ref
    k = particle_container.rho0 / rho_mean

    # viscosity, pressure, surface_tension
    return k, 1.0, k
end
