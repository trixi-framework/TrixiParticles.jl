# Sorted in order of computational cost

@doc raw"""
    AkinciFreeSurfaceCorrection(rho0)

Free surface correction according to Akinci et al. (2013).
At a free surface, the mean density is typically lower than the reference density,
resulting in reduced surface tension and viscosity forces.
The free surface correction adjusts the viscosity, pressure, and surface tension forces
near free surfaces to counter this effect.
It's important to note that this correlation is unphysical and serves as an approximation.
The computation time added by this method is about 2-3%.

# Arguments
- `rho0`: Reference density.

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

# `rho_mean` is the mean density of the fluid, which is used to determine correction values near the free surface.
#  Return a tuple `(viscosity_correction, pressure_correction, surface_tension_correction)` representing the correction terms.
@inline function free_surface_correction(correction::AkinciFreeSurfaceCorrection,
                                         particle_system, rho_mean)
    # Equation 4 in ref
    k = correction.rho0 / rho_mean

    # Viscosity, pressure, surface_tension
    return k, 1.0, k
end

@inline function free_surface_correction(correction, particle_system, rho_mean)
    return 1.0, 1.0, 1.0
end

@doc raw"""
    ShepardKernelCorrection()
    KernelGradientCorrection()

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
struct KernelGradientCorrection end

function kernel_correction_coefficient(container, particle)
    kernel_correction_coefficient(container, particle, container.correction)
end

function kernel_correction_coefficient(container, particle, correction)
    #skip
end

function kernel_correction_coefficient(container, particle,
                                       ::Union{KernelCorrection, KernelGradientCorrection})
    return container.cache.kernel_correction_coefficient[particle]
end

function kernel_correct_value(container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack cache = container
    @unpack kernel_correction_coefficient = cache

    kernel_correction_coefficient .= zero(eltype(kernel_correction_coefficient))

    # Use all other containers for the density summation
    @trixi_timeit timer() "compute kernel correction value" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                                       neighbor_container)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index, neighbor_container,
                                      semi)
        v_neighbor_container = wrap_v(v_ode, neighbor_container_index, neighbor_container,
                                      semi)

        container_coords = current_coordinates(u, container)
        neighbor_coords = current_coordinates(u_neighbor_container, neighbor_container)

        neighborhood_search = neighborhood_searches[container_index][neighbor_container_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        for_particle_neighbor(container, neighbor_container, container_coords,
                              neighbor_coords,
                              neighborhood_search) do particle, neighbor, pos_diff, distance
            rho_b = particle_density(v_neighbor_container, neighbor_container, neighbor)
            m_b = hydrodynamic_mass(neighbor_container, neighbor)
            volume = m_b / rho_b

            kernel_correction_coefficient[particle] += volume *
                                                       smoothing_kernel(container, distance)
        end
    end
end

function dw_gamma(container, particle)
    dw_gamma(container, particle, container.correction)
end

function dw_gamma(container, particle, ::Any)
    #skip
end

function dw_gamma(container, particle, ::KernelGradientCorrection)
    return extract_svector(container.cache.dw_gamma, container, particle)
end

function kernel_gradient_correct_value(container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack cache = container
    @unpack kernel_correction_coefficient, dw_gamma = cache

    kernel_correction_coefficient .= zero(eltype(kernel_correction_coefficient))
    dw_gamma .= zero(eltype(dw_gamma))

    # Use all other containers for the density summation
    @trixi_timeit timer() "compute kernel gradient correction value" foreach_enumerate(particle_containers) do (neighbor_container_index,
                                                                                                                neighbor_container)
        u_neighbor_container = wrap_u(u_ode, neighbor_container_index, neighbor_container,
                                      semi)
        v_neighbor_container = wrap_v(v_ode, neighbor_container_index, neighbor_container,
                                      semi)

        container_coords = current_coordinates(u, container)
        neighbor_coords = current_coordinates(u_neighbor_container, neighbor_container)

        neighborhood_search = neighborhood_searches[container_index][neighbor_container_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        for_particle_neighbor(container, neighbor_container, container_coords,
                              neighbor_coords,
                              neighborhood_search) do particle, neighbor, pos_diff, distance
            rho_b = particle_density(v_neighbor_container, neighbor_container, neighbor)
            m_b = hydrodynamic_mass(neighbor_container, neighbor)
            volume = m_b / rho_b

            kernel_correction_coefficient[particle] += volume *
                                                       smoothing_kernel(container, distance)
            if distance > sqrt(eps())
                dw_gamma[:, particle] += volume *
                                         smoothing_kernel_grad(container, pos_diff,
                                                               distance)
            end
        end
    end

    for particle in eachparticle(container)
        dw_gamma[:, particle] ./= kernel_correction_coefficient[particle]
    end
end
