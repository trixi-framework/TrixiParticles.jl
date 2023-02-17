# Fluid-fluid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, smoothing_kernel, smoothing_length, surface_tension, surface_normal, a_surface_tension, a_viscosity, ref_density = particle_container

    # no major performance impact
    if !isnan(a_surface_tension[1, 1])
        fill!(a_surface_tension, 0.0)
        fill!(a_viscosity, 0.0)
    end

    # some surface tension models require the surface normal
    if need_normal(surface_tension) # is deactivated by default
        calc_normal_akinci(v_particle_container, u_particle_container, v_neighbor_container,
                           u_neighbor_container, neighborhood_search,
                           particle_container, neighbor_container)
    end

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        #Note: don't move outside of @threaded block causes a bug (@2/2023)
        NDIMS = ndims(particle_container)

        correction_term = 0
        dv_viscosity = zero(SVector{NDIMS, eltype(particle_container)})
        dv_pressure = zero(SVector{NDIMS, eltype(particle_container)})
        dv_surface_tension = zero(SVector{NDIMS, eltype(particle_container)})

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                v_diff = get_particle_vel(particle, v_particle_container,
                                          particle_container) -
                         get_particle_vel(neighbor, v_neighbor_container,
                                          neighbor_container)

                # per convention neigbor values are indicated by 'b' and local values with 'a'
                rho_a = get_particle_density(particle, v_particle_container,
                                             particle_container)
                rho_b = get_particle_density(neighbor, v_neighbor_container,
                                             neighbor_container)
                rho_mean = (rho_a + rho_b) / 2

                m_b = neighbor_container.mass[neighbor]

                grad_kernel = calc_grad_kernel(smoothing_kernel, distance, smoothing_length,
                                               pos_diff)


                # equation 4 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
                # correction term for free surfaces
                k_ij = 1.0
                if surface_tension isa AkinciTypeSurfaceTension
                    k_ij = ref_density / rho_mean
                end

                dv_viscosity += k_ij *
                                calc_visc_term(particle_container, m_b, v_diff, pos_diff,
                                               distance,
                                               rho_mean, smoothing_length, grad_kernel)

                p_a = particle_container.pressure[particle]
                p_b = neighbor_container.pressure[neighbor]
                dv_pressure += calc_momentum_eq(m_b, p_a, p_b, rho_a, rho_b, grad_kernel)

                dv_surface_tension += k_ij *
                                      calc_surface_tension(particle, neighbor, pos_diff,
                                                           distance,
                                                           particle_container,
                                                           neighbor_container,
                                                           surface_tension)

                # save acceleration term if vector is allocated
                # if !isnan(a_surface_tension[1, 1])
                #     for i in 1:NDIMS
                #         a_surface_tension[i, particle] += dv_surface_tension[i]
                #     end
                # end

                # if !isnan(a_viscosity[1, 1])
                #     for i in 1:NDIMS
                #         a_viscosity[i, particle] += dv_viscosity[i]
                #     end
                # end

                continuity_equation!(dv, density_calculator,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, v_diff, distance,
                                     particle_container, neighbor_container, grad_kernel)
            end
        end

        for i in 1:NDIMS
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i] +
                               dv_surface_tension[i]
        end
    end

    return dv
end

# section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: most of the time this only leads to an approximation of the surface normal
function calc_normal_akinci(v_particle_container, u_particle_container,
                            v_neighbor_container, u_neighbor_container,
                            neighborhood_search, particle_container,
                            neighbor_container)
    @unpack smoothing_kernel, smoothing_length, surface_normal = particle_container

    @threaded for particle in each_moving_particle(particle_container)
        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)
        for i in 1:ndims(particle_container)
            surface_normal[i, particle] = 0.0
        end

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)
            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            # correctness strongly depends on this leading to a symmetric distribution of points!
            if sqrt(eps()) < distance <= smoothing_length
                m_b = neighbor_container.mass[neighbor]
                density_neighbor = get_particle_density(neighbor, v_neighbor_container,
                                                        neighbor_container)
                surface_normal[:, particle] .+= m_b / density_neighbor *
                                                kernel_deriv(smoothing_kernel, distance,
                                                             smoothing_length) *
                                                pos_diff / distance
            end
        end
        for i in 1:ndims(particle_container)
            surface_normal[i, particle] *= smoothing_length
        end
    end
end

@inline function need_normal(surface_tension_model)
    if surface_tension_model isa SurfaceTensionAkinci
        return true
    end
    return false
end

@inline function calc_surface_tension(particle, neighbor, pos_diff, distance,
                                      particle_container, neighbor_container,
                                      surface_tension::CohesionForceAkinci)
    @unpack smoothing_length = particle_container

    # per convention neigbor values are indicated by 'b' and local values with 'a'
    m_b = neighbor_container.mass[neighbor]

    return surface_tension(smoothing_length, m_b, pos_diff, distance)
end

@inline function calc_surface_tension(particle, neighbor, pos_diff, distance,
                                      particle_container, neighbor_container,
                                      surface_tension::SurfaceTensionAkinci)
    @unpack smoothing_length = particle_container

    # per convention neigbor values are indicated by 'b' and local values with 'a'
    m_b = neighbor_container.mass[neighbor]

    return surface_tension(smoothing_length, m_b,
                           get_normal(particle, particle_container,
                                      surface_tension),
                           get_normal(neighbor, neighbor_container,
                                      surface_tension), pos_diff,
                           distance)
end

@inline function calc_surface_tension(particle, neighbor, pos_diff, distance,
                                      particle_container::ParticleContainer,
                                      neighbor_container,
                                      surface_tension::NoSurfaceTension)
    return zeros(SVector{ndims(particle_container), eltype(particle_container)})
end

@inline function calc_grad_kernel(smoothing_kernel, distance, smoothing_length, pos_diff)
    return kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff /
           distance
end

@inline function calc_momentum_eq(m_b, p_a, p_b, rho_a, rho_b, grad_kernel)
    return -m_b * (p_a / rho_a^2 + p_b / rho_b^2) * grad_kernel
end

@inline function calc_visc_term(particle_container, m_b, v_diff, pos_diff, distance,
                                density_mean, smoothing_length, grad_kernel)
    @unpack state_equation, viscosity = particle_container

    return m_b *
           viscosity(state_equation.sound_speed, v_diff, pos_diff,
                     distance, density_mean, smoothing_length) * grad_kernel
end

# update density in case of ContinuityDensity is used
@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, v_diff, distance,
                                      particle_container::FluidParticleContainer,
                                      neighbor_container, grad_kernel)
    @unpack smoothing_kernel, smoothing_length = particle_container

    # density change added at the end of du
    NDIMS = ndims(particle_container)
    dv[NDIMS + 1, particle] += sum(neighbor_container.mass[neighbor] *
                                  v_diff .* grad_kernel)

    return dv
end

# skip
@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_container, v_neighbor_container,
                                      particle, neighbor, pos_diff, v_diff, distance,
                                      particle_container, neighbor_container, grad_kernel)
    return dv
end

# Fluid-boundary and fluid-solid interaction
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::Union{BoundaryParticleContainer,
                                             SolidParticleContainer})
    @unpack density_calculator, state_equation, viscosity,
    smoothing_kernel, smoothing_length = particle_container
    @unpack sound_speed = state_equation

    @threaded for particle in each_moving_particle(particle_container)
        density_a = get_particle_density(particle, v_particle_container, particle_container)
        v_a = get_particle_vel(particle, v_particle_container, particle_container)

        particle_coords = get_current_coords(particle, u_particle_container,
                                             particle_container)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = get_current_coords(neighbor, u_neighbor_container,
                                                 neighbor_container)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)

            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                m_b = neighbor_container.mass[neighbor]

                v_diff = get_particle_vel(particle, v_particle_container,
                                          particle_container) -
                         get_particle_vel(neighbor, v_neighbor_container,
                                          neighbor_container)

                grad_kernel = calc_grad_kernel(smoothing_kernel, distance, smoothing_length,
                                               pos_diff)

                continuity_equation!(dv, density_calculator,
                                     v_particle_container, v_neighbor_container,
                                     particle, neighbor, pos_diff, v_diff, distance,
                                     particle_container, neighbor_container, grad_kernel)

                dv_viscosity = m_b *
                               viscosity(sound_speed, v_a, pos_diff, distance, density_a,
                                         smoothing_length) * grad_kernel

                dv_boundary = boundary_particle_impact(particle, neighbor,
                                                       v_particle_container,
                                                       v_neighbor_container,
                                                       particle_container,
                                                       neighbor_container,
                                                       grad_kernel, pos_diff, distance, m_b)

                for i in 1:ndims(particle_container)
                    dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
                end
            end
        end
    end

    return dv
end
