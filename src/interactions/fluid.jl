# Fluid-fluid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, smoothing_kernel, smoothing_length, surface_tension, surface_normal, a_surface_tension, a_viscosity = particle_container

    if !isnan(a_surface_tension[1, 1])
        a_surface_tension .= 0
        a_viscosity .= 0
    end

    # some surface tension models require the surface normal
    if need_normal(surface_tension)
        surface_normal .= 0.0
        calc_normal_akinci(u_particle_container, u_neighbor_container, neighborhood_search, particle_container, neighbor_container)
    end

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)

                v_diff = get_particle_vel(particle, u_particle_container, particle_container) -
                get_particle_vel(neighbor, u_neighbor_container, neighbor_container)

                calc_dv!(du, u_particle_container, u_neighbor_container,
                         particle, neighbor, pos_diff, v_diff, distance,
                         particle_container, neighbor_container)

                continuity_equation!(du, density_calculator,
                                     u_particle_container, u_neighbor_container,
                                     particle, neighbor, pos_diff, v_diff, distance,
                                     particle_container, neighbor_container)
            end
        end
    end

    return du
end

# section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: most of the time this only leads to an approximation of the surface normal
@inline function calc_normal_akinci(u_particle_container, u_neighbor_container, neighborhood_search, particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length, surface_normal = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)
            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            # correctness strongly depends on this leading to a symmetric distribution of points!
            if sqrt(eps()) < distance <= smoothing_length
                m_b = neighbor_container.mass[neighbor]
                density_neighbor = get_particle_density(neighbor, u_neighbor_container,
                                                        neighbor_container)
                surface_normal[:, particle] .+= m_b / density_neighbor *
                                                kernel_deriv(smoothing_kernel, distance,
                                                             smoothing_length) *
                                                pos_diff / distance
            end
        end
        surface_normal[:, particle] .*= smoothing_length
    end
end

@inline function need_normal(surface_tension_model)
    if surface_tension_model isa SurfaceTensionAkinci
        return true
    end
    return false
end

# calculate the dv term
@inline function calc_dv!(du, u_particle_container, u_neighbor_container,
                          particle, neighbor, pos_diff, v_diff, distance,
                          particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length, state_equation, viscosity, surface_tension, ref_density, radius, a_surface_tension, a_viscosity = particle_container

    # per convention neigbor values are indicated by 'b' and local values with 'a'
    rho_a = get_particle_density(particle, u_particle_container, particle_container)
    rho_b = get_particle_density(neighbor, u_neighbor_container, neighbor_container)
    rho_mean = (rho_a + rho_b) / 2

    m_b = neighbor_container.mass[neighbor]

    p_a = particle_container.pressure[particle]
    p_b = neighbor_container.pressure[neighbor]


    # calculate the 0-order correction term
    #correction_term = m_b/density_neighbor * kernel(smoothing_kernel, distance, smoothing_length)
    #println("correction term", correction_term)
    grad_kernel = calc_grad_kernel(smoothing_kernel, distance, smoothing_length, pos_diff)

    # equation 4 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
    # correction term for free surfaces
    k_ij = ref_density / rho_mean

    dv_viscosity = k_ij * calc_visc_term(particle_container, m_b, v_diff, pos_diff, distance, rho_mean, smoothing_length, grad_kernel)
    dv_pressure = calc_momentum_eq(m_b, p_a, p_b, rho_a, rho_b, grad_kernel)

    # surface tension
    dv_surface_tension = k_ij * surface_tension(smoothing_length, m_b,
                                         get_normal(particle, particle_container,
                                                    surface_tension,
                                                    ndims(particle_container)),
                                         get_normal(neighbor, particle_container,
                                                    surface_tension,
                                                    ndims(particle_container)), pos_diff,
                                         distance)

    # save acceleration term if vector are allocated
    if !isnan(a_surface_tension[1, 1])
        a_viscosity[:, particle] .+= dv_viscosity
        a_surface_tension[:, particle] .+= dv_surface_tension
    end

    dv = dv_pressure + dv_viscosity + dv_surface_tension

    for i in 1:ndims(particle_container)
        du[ndims(particle_container) + i, particle] += dv[i]
    end

    return du
end

@inline function calc_grad_kernel(smoothing_kernel, distance, smoothing_length, pos_diff)
    return kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff /
    distance
end

@inline function calc_momentum_eq(m_b, p_a, p_b, rho_a, rho_b, grad_kernel)
    return -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * grad_kernel
end

@inline function calc_visc_term(particle_container, m_b, v_diff, pos_diff, distance, density_mean, smoothing_length, grad_kernel)
    @unpack state_equation, viscosity = particle_container

    return m_b * viscosity(state_equation.sound_speed, v_diff, pos_diff,
    distance, density_mean, smoothing_length) * grad_kernel
end

# update density in case of ContinuityDensity is used
@inline function continuity_equation!(du, density_calculator::ContinuityDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, v_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container)
    @unpack smoothing_kernel, smoothing_length = particle_container

    # density change added at the end of du
    du[2 * ndims(particle_container) + 1, particle] += sum(neighbor_container.mass[neighbor] *
    v_diff *
                                                           kernel_deriv(smoothing_kernel,
                                                                        distance,
                                                                        smoothing_length) .*
                                                           pos_diff) / distance

    return du
end

# skip
@inline function continuity_equation!(du, density_calculator::SummationDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, v_diff, distance,
                                      particle_container, neighbor_container)
    return du
end

# Fluid-boundary and fluid-solid interaction
function interact!(du, u_particle_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::Union{BoundaryParticleContainer,
                                             SolidParticleContainer})
    @unpack density_calculator, state_equation, viscosity, smoothing_kernel, smoothing_length = particle_container
    @unpack sound_speed = state_equation

    @threaded for particle in each_moving_particle(particle_container)
        density_a = get_particle_density(particle, u_particle_container, particle_container)
        v_a = get_particle_vel(particle, u_particle_container, particle_container)

        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                m_b = neighbor_container.mass[neighbor]

                continuity_equation!(du, density_calculator,
                                     u_particle_container, u_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)

                pi_ab = viscosity(sound_speed, v_a, pos_diff, distance, density_a,
                                  smoothing_length)
                dv_viscosity = m_b * pi_ab *
                               kernel_deriv(smoothing_kernel, distance, smoothing_length) *
                               pos_diff / distance

                dv_boundary = boundary_particle_impact(particle, neighbor,
                                                       u_particle_container,
                                                       u_neighbor_container,
                                                       particle_container,
                                                       neighbor_container,
                                                       pos_diff, distance, m_b)

                dv = dv_boundary + dv_viscosity

                for i in 1:ndims(particle_container)
                    du[ndims(particle_container) + i, particle] += dv[i]
                end
            end
        end
    end

    return du
end
