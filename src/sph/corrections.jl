struct NoCorrection end

@doc raw"""
KernelCorrection()
KernelGradientCorrection()

Kernel correction uses Shepard interpolation to obtain a 0-th order accurate result.

## References:
- Bonet and Lok. "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic formulations".
  In: Comput. Methods Appl. Mech. Eng. (1999), pages 97-115.
- Basa et al. "Robustness and accuracy of SPH formulations for viscous flow".
  In: Int. J. Numer. Meth. Fluids (2009), pages 1127-1148.
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
