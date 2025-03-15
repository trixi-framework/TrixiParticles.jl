# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system)
    (; sound_speed, density_calculator, correction) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords,
                           neighborhood_search;
                           points=each_moving_particle(particle_system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

        p_a = particle_pressure(v_particle_system, particle_system, particle)
        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        # This technique is for a more robust `pressure_acceleration` but only with TVF.
        # It results only in significant improvement for EDAC and not for WCSPH.
        # See Ramachandran (2019) p. 582
        # Note that the return value is zero when not using EDAC with TVF.
        p_avg = average_pressure(particle_system, particle)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        dv_pressure = pressure_acceleration(particle_system, neighbor_system,
                                            particle, neighbor,
                                            m_a, m_b, p_a - p_avg, p_b - p_avg, rho_a,
                                            rho_b, pos_diff, distance, grad_kernel,
                                            correction)

        dv_viscosity_ = dv_viscosity(particle_system, neighbor_system,
                                     v_particle_system, v_neighbor_system,
                                     particle, neighbor, pos_diff, distance,
                                     sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)

        # Add convection term when using `TransportVelocityAdami`
        dv_convection = momentum_convection(particle_system, neighbor_system,
                                            pos_diff, distance,
                                            v_particle_system, v_neighbor_system,
                                            rho_a, rho_b, m_a, m_b,
                                            particle, neighbor, grad_kernel)

        dv_surface_tension = surface_tension_force(surface_tension_a, surface_tension_b,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor, pos_diff, distance)

        dv_adhesion = adhesion_force(surface_tension_a, particle_system, neighbor_system,
                                     particle, neighbor, pos_diff, distance)

        dv_velocity_correction = velocity_correction(particle_system, neighbor_system,
                                                     pos_diff, distance,
                                                     v_particle_system, v_neighbor_system,
                                                     rho_a, rho_b, m_a, m_b,
                                                     particle, neighbor, grad_kernel)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity_[i] + dv_convection[i] +
                               dv_surface_tension[i] + dv_adhesion[i] +
                               dv_velocity_correction[i]
        end

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        pressure_evolution!(dv, particle_system, neighbor_system,
                            v_particle_system, v_neighbor_system, v_diff, grad_kernel,
                            particle, neighbor, pos_diff, distance, sound_speed,
                            m_a, m_b, p_a, p_b, rho_a, rho_b)

        transport_velocity!(dv, particle_system, rho_a, rho_b, m_a, m_b,
                            grad_kernel, particle)

        continuity_equation!(dv, density_calculator, v_diff, particle, m_b, rho_a, rho_b,
                             particle_system, grad_kernel)
    end

    return dv
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system,
                                     v_particle_system, v_neighbor_system,
                                     v_diff, grad_kernel, particle, neighbor,
                                     pos_diff, distance, sound_speed,
                                     m_a, m_b, p_a, p_b, rho_a, rho_b)
    (; particle_refinement) = particle_system

    # This is basically the continuity equation times `sound_speed^2`
    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel)

    beta_inv_a = beta_correction(particle_system, particle)
    beta_inv_b = beta_correction(neighbor_system, neighbor)

    dv[end, particle] += beta_inv_a * artificial_eos +
                         pressure_damping_term(particle_system, neighbor_system,
                                               particle_refinement, particle, neighbor,
                                               pos_diff, distance, beta_inv_a, m_a, m_b,
                                               p_a, p_b, rho_b, rho_b, grad_kernel) +
                         pressure_reduction(particle_system, neighbor_system,
                                            particle_refinement,
                                            v_particle_system, v_neighbor_system,
                                            particle, neighbor, pos_diff, distance, m_b,
                                            p_a, p_b, rho_a, rho_b, beta_inv_a, beta_inv_b)

    return dv
end
@inline beta_correction(system, particle) = one(eltype(system))

@inline function beta_correction(system::FluidSystem, particle)
    beta_correction(system, system.particle_refinement, particle)
end

@inline beta_correction(particle_system, ::Nothing, particle) = one(eltype(particle_system))

@inline function beta_correction(particle_system, refinement, particle)
    return inv(particle_system.cache.beta[particle])
end

function pressure_damping_term(particle_system, neighbor_system, ::Nothing,
                               particle, neighbor, pos_diff, distance, beta_inv_a,
                               m_a, m_b, p_a, p_b, rho_a, rho_b, grad_kernel)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    eta_a = rho_a * particle_system.nu_edac
    eta_b = rho_b * particle_system.nu_edac
    eta_tilde = 2 * eta_a * eta_b / (eta_a + eta_b)

    smoothing_length_average = 0.5 * (smoothing_length(particle_system, particle) +
                                smoothing_length(particle_system, particle))
    tmp = eta_tilde / (distance^2 + smoothing_length_average^2 / 100)

    # This formulation was introduced by Hu and Adams (2006). https://doi.org/10.1016/j.jcp.2005.09.001
    # They argued that the formulation is more flexible because of the possibility to formulate
    # different inter-particle averages or to assume different inter-particle distributions.
    # Ramachandran (2019) and Adami (2012) use this formulation also for the pressure acceleration.
    #
    # TODO: Is there a better formulation to discretize the Laplace operator?
    # Because when using this formulation for the pressure acceleration, it is not
    # energy conserving.
    # See issue: https://github.com/trixi-framework/TrixiParticles.jl/issues/394
    #
    # This is similar to density diffusion in WCSPH
    return volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)
end

function pressure_damping_term(particle_system, neighbor_system, refinement,
                               particle, neighbor, pos_diff, distance, beta_inv_a,
                               m_a, m_b, p_a, p_b, rho_a, rho_b, grad_kernel_a)
    (; sound_speed) = particle_system

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    # Haftu et al. (2022) uses `alpha_edac = 1.5` in all their simulations
    alpha_edac = 1.5

    # TODO: Haftu et al. (2022) use `8` but I think it depeneds on the dimension (see Monaghan, 2005)
    tmp = 2 * ndims(particle_system) + 4

    nu_edac_a = alpha_edac * sound_speed * smoothing_length(particle_system, particle) / tmp
    nu_edac_b = alpha_edac * sound_speed * smoothing_length(neighbor_system, neighbor) / tmp

    nu_edac_ab = 4 * (nu_edac_a * nu_edac_b) / (nu_edac_a + nu_edac_b)

    grad_kernel_b = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

    grad_W_avg = 0.5 * (grad_kernel_a + grad_kernel_b)

    return beta_inv_a * nu_edac_ab * pressure_diff * dot(pos_diff, grad_W_avg) * m_b / rho_b
end

function pressure_reduction(particle_system, neighbor_system, ::Nothing,
                            v_particle_system, v_neighbor_system,
                            particle, neighbor, pos_diff, distance, m_b,
                            p_a, p_b, rho_a, rho_b, beta_a, beta_b)
    return zero(eltype(particle_system))
end

function pressure_reduction(particle_system, neighbor_system, refinement,
                            v_particle_system, v_neighbor_system,
                            particle, neighbor, pos_diff, distance, m_b,
                            p_a, p_b, rho_a, rho_b, beta_a, beta_b)
    beta_inv_a = beta_correction(particle_system, particle)
    beta_inv_b = beta_correction(neighbor_system, neighbor)

    p_a_avg = average_pressure(particle_system, particle)
    p_b_avg = average_pressure(neighbor_system, neighbor)

    P_a = (p_a - p_a_avg) / (rho_a^2 * beta_inv_a)
    P_b = (p_b - p_b_avg) / (rho_b^2 * beta_inv_b)

    grad_kernel_a = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)
    grad_kernel_b = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

    v_diff = advection_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_particle_system, particle_system, particle)

    return m_b * (dot(v_diff, P_a * grad_kernel_a + P_b * grad_kernel_b))
end

@inline function velocity_correction(particle_system, neighbor_system,
                                     pos_diff, distance,
                                     v_particle_system, v_neighbor_system,
                                     rho_a, rho_b, m_a, m_b,
                                     particle, neighbor, grad_kernel)
    return zero(pos_diff)
end

@inline function velocity_correction(particle_system,
                                     neighbor_system::EntropicallyDampedSPHSystem,
                                     pos_diff, distance,
                                     v_particle_system, v_neighbor_system,
                                     rho_a, rho_b, m_a, m_b,
                                     particle, neighbor, grad_kernel)
    velocity_correction(particle_system, neighbor_system,
                        particle_system.particle_refinement,
                        pos_diff, distance, v_particle_system, v_neighbor_system,
                        rho_a, rho_b, m_a, m_b, particle, neighbor, grad_kernel)
end

@inline function velocity_correction(particle_system, neighbor_system, ::Nothing,
                                     pos_diff, distance,
                                     v_particle_system, v_neighbor_system,
                                     rho_a, rho_b, m_a, m_b,
                                     particle, neighbor, grad_kernel)
    return zero(pos_diff)
end

@inline function velocity_correction(particle_system, neighbor_system,
                                     particle_refinement, pos_diff, distance,
                                     v_particle_system, v_neighbor_system,
                                     rho_a, rho_b, m_a, m_b, particle, neighbor,
                                     grad_kernel)
    momentum_velocity_a = current_velocity(v_particle_system, particle_system, particle)
    advection_velocity_a = advection_velocity(v_particle_system, particle_system, particle)

    momentum_velocity_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    advection_velocity_b = advection_velocity(v_neighbor_system, neighbor_system, neighbor)

    v_diff = momentum_velocity_a - momentum_velocity_b
    v_diff_tilde = advection_velocity_a - advection_velocity_b

    beta_inv_a = beta_correction(particle_system, particle)

    return -m_b * beta_inv_a *
           (dot(v_diff_tilde - v_diff, grad_kernel) * momentum_velocity_a) / rho_b
end

# We need a separate method for EDAC since the density is stored in `v[end-1,:]`.
@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      vdiff, particle, m_b, rho_a, rho_b,
                                      particle_system::EntropicallyDampedSPHSystem,
                                      grad_kernel)
    dv[end - 1, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)

    return dv
end

@inline function continuity_equation!(dv, density_calculator,
                                      vdiff, particle, m_b, rho_a, rho_b,
                                      particle_system::EntropicallyDampedSPHSystem,
                                      grad_kernel)
    return dv
end

# Formulation using symmetric gradient formulation for corrections not depending on local neighborhood.
@inline function pressure_acceleration(particle_system::EntropicallyDampedSPHSystem,
                                       neighbor_system, particle, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, correction)
    return pressure_acceleration(particle_system, particle_system.particle_refinement,
                                 neighbor_system, particle, neighbor,
                                 m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff, distance, W_a,
                                 correction)
end

# Formulation using symmetric gradient formulation for corrections not depending on local neighborhood.
@inline function pressure_acceleration(particle_system::EntropicallyDampedSPHSystem,
                                       ::Nothing, particle, neighbor_system, neighbor,
                                       m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                       distance, W_a, correction)
    (; pressure_acceleration_formulation) = particle_system
    return pressure_acceleration_formulation(m_a, m_b, rho_a, rho_b, p_a, p_b, W_a)
end

@inline function pressure_acceleration(particle_system::EntropicallyDampedSPHSystem,
                                       particle_refinement, neighbor_system, particle,
                                       neighbor, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                       pos_diff, distance, W_a, correction)
    p_a_avg = average_pressure(particle_system, particle)
    p_b_avg = average_pressure(neighbor_system, neighbor)

    P_a = beta_correction(particle_system, particle) * (p_a - p_a_avg) / rho_a^2
    P_b = beta_correction(neighbor_system, neighbor) * (p_b - p_b_avg) / rho_b^2

    grad_kernel_a = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)
    grad_kernel_b = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

    return -m_b * (P_a * grad_kernel_a + P_b * grad_kernel_b)
end
