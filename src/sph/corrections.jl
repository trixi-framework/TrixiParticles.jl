# Use the surface correction as used in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
abstract type AkinciFreeSurfaceCorrection end

# number of correction values
@inline ncvals(::Any) = 3

@inline function fluid_corrections(::AkinciTypeSurfaceTension, particle_container, rho_mean)
    return akinci_free_surface_correction(particle_container, rho_mean)
end

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
    return k, 1.0, k
end
