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

# also referred to as 0th order scalar correction (cheapest)
struct KernelCorrection end

# also referred to as 0th order correction of scalars and gradients (cheapest)
struct KernelGradientCorrection end

function cw(container, particle)
    cw(container, particle, container.correction)
end

function cw(container, particle, ::Any)
    #skip
end

function cw(container, particle, ::Union{KernelCorrection, KernelGradientCorrection})
    return container.cache.cw[particle]
end

function kernel_correct_value(container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack cache = container
    @unpack cw = cache

    cw .= zero(eltype(cw))

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

            cw[particle] += volume * smoothing_kernel(container, distance)
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
    return container.cache.dw_gamma[:, particle]
end

function kernel_gradient_correct_value(container, container_index, v, u, v_ode, u_ode, semi)
    @unpack particle_containers, neighborhood_searches = semi
    @unpack cache = container
    @unpack cw, dw_gamma = cache

    cw .= zero(eltype(cw))
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

            cw[particle] += volume * smoothing_kernel(container, distance)
            if distance > sqrt(eps())
                dw_gamma[:, particle] += volume *
                                         smoothing_kernel_grad(container, pos_diff,
                                                               distance)
            end
        end
    end

    for particle in eachparticle(container)
        dw_gamma[:, particle] ./= cw[particle]
    end
end

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
