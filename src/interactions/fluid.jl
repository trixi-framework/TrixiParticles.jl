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

        @threaded for particle in each_moving_particle(particle_container)
            particle_coords = get_current_coords(particle, u_particle_container,
                                                 particle_container)

            # section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
            # Note: most of the time this only leads to an approximation of the surface normal
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

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                calc_dv!(du, u_particle_container, u_neighbor_container,
                         particle, neighbor, pos_diff, distance,
                         particle_container, neighbor_container)

                continuity_equation!(du, density_calculator,
                                     u_particle_container, u_neighbor_container,
                                     particle, neighbor, pos_diff, distance,
                                     particle_container, neighbor_container)
            end
        end
    end

    return du
end

@inline function need_normal(surface_tension_model)
    if surface_tension_model isa SurfaceTensionAkinci
        return true
    end
    return false
end

# calculate the dv term
@inline function calc_dv!(du, u_particle_container, u_neighbor_container,
                          particle, neighbor, pos_diff, distance,
                          particle_container, neighbor_container)
    @unpack smoothing_kernel, smoothing_length, state_equation, viscosity, surface_tension, ref_density, radius, a_surface_tension, a_viscosity = particle_container

    density_particle = get_particle_density(particle, u_particle_container,
                                            particle_container)
    density_neighbor = get_particle_density(neighbor, u_neighbor_container,
                                            neighbor_container)

    # Viscosity
    v_diff = get_particle_vel(particle, u_particle_container, particle_container) -
             get_particle_vel(neighbor, u_neighbor_container, neighbor_container)
    density_mean = (density_particle + density_neighbor) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, density_mean, smoothing_length)

    m_b = neighbor_container.mass[neighbor]

    # equation 4 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
    k_ij = ref_density / density_mean

    # calculate the 0-order correction term
    correction_term = m_b/density_neighbor * kernel(smoothing_kernel, distance, smoothing_length)
    println("correction term", correction_term)
    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff /
                  distance


    # momentum equation
    dv_pressure = -m_b *
                  (particle_container.pressure[particle] / density_particle^2 +
                   neighbor_container.pressure[neighbor] / density_neighbor^2) * grad_kernel

    dv_viscosity = k_ij * m_b * pi_ab * grad_kernel

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

# update density in case of ContinuityDensity is used
@inline function continuity_equation!(du, density_calculator::ContinuityDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container)
    @unpack smoothing_kernel, smoothing_length = particle_container

    vdiff = get_particle_vel(particle, u_particle_container, particle_container) -
            get_particle_vel(neighbor, u_neighbor_container, neighbor_container)

    # density change added at the end of du
    du[2 * ndims(particle_container) + 1, particle] += sum(neighbor_container.mass[neighbor] *
                                                           vdiff *
                                                           kernel_deriv(smoothing_kernel,
                                                                        distance,
                                                                        smoothing_length) .*
                                                           pos_diff) / distance

    return du
end

# skip
@inline function continuity_equation!(du, density_calculator::SummationDensity,
                                      u_particle_container, u_neighbor_container,
                                      particle, neighbor, pos_diff, distance,
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
