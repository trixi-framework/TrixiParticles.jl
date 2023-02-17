abstract type AkinciFreeSurfaceCorrection end

# number of correction values
@inline ncvals(::Any) = 3

# index for the correction values
@inline viscosity_K_id() = 1
@inline pressure_K_id() = 2
@inline surface_tension_K_id() = 3

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

# equation 4 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# correction term for free surfaces
@inline function akinci_free_surface_correction(particle_container, rho_mean)
    @unpack ref_density = particle_container
    k = ref_density / rho_mean
    return SVector{ncvals(particle_container), eltype(particle_container)}(k, 1.0, k)
end
