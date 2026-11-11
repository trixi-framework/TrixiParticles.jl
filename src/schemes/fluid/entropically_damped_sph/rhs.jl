# Fluid-fluid and fluid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::EntropicallyDampedSPHSystem,
                   neighbor_system, semi)
    (; sound_speed, density_calculator, correction, nu_edac) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    h = initial_smoothing_length(particle_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_symmetric_point_neighbor(particle_system, neighbor_system,
                                     system_coords, neighbor_coords, semi;
                                     points=each_integrated_particle(particle_system)) do particle,
                                                                                          neighbor,
                                                                                          pos_diff,
                                                                                          distance
        # Skip neighbors with (almost) the same position because the kernel gradient
        # is zero, but computing it would divide by zero (see `almostzero`).
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(particle_system) && distance < almostzero(h) && return

        grad_kernel = symmetric_kernel_grad(particle_system, neighbor_system,
                                            pos_diff, distance, particle, neighbor)

        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)
        v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        # Propagate `@inbounds` to the momentum equation, which accesses particle data
        dv_particle = @inbounds add_momentum_equation(zero(v_a), particle_system,
                                                      neighbor_system,
                                                      v_particle_system, v_neighbor_system,
                                                      particle, neighbor,
                                                      pos_diff, distance, grad_kernel,
                                                      sound_speed, m_a, m_b, p_a, p_b,
                                                      rho_a, rho_b, v_a, v_b)

        # Extra terms in the momentum equation when using a shifting technique.
        # These are not included in `add_momentum_equation` because they must not be
        # applied to structures (see `interact_structure_fluid!`).
        dv_particle = @inbounds add_dv_shifting(dv_particle,
                                                shifting_technique(particle_system),
                                                particle_system, neighbor_system,
                                                v_particle_system, v_neighbor_system,
                                                particle, neighbor, m_a, m_b, rho_a, rho_b,
                                                v_a, v_b,
                                                pos_diff, distance, grad_kernel, correction)

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        v_diff = v_a - v_b

        pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                            particle, neighbor, pos_diff, distance,
                            sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, nu_edac)

        drho_particle = zero(rho_a)

        # Propagate `@inbounds` to the continuity equation, which accesses particle data
        drho_particle = @inbounds add_continuity_equation(drho_particle, density_calculator,
                                                          particle_system, neighbor_system,
                                                          particle, neighbor, pos_diff,
                                                          distance, m_b, rho_a, rho_b,
                                                          v_a, v_b, grad_kernel)

        @inbounds write_drho_particle!(dv, density_calculator, drho_particle, particle)
    end

    return dv
end

# Add the acceleration of `particle` due to `neighbor` to `dv_particle`.
# `particle` must be in `particle_system` and `neighbor` must be in `neighbor_system`.
# This includes pressure, viscosity, surface tension and adhesion, but not the extra terms
# from shifting techniques (see `interact_structure_fluid!`).
# Note that this function is also used for the structure-fluid interaction to compute
# the exact opposite pair force. When adding new terms here, make sure that they are
# also valid for structure neighbors.
@propagate_inbounds function add_momentum_equation(dv_particle,
                                                   particle_system::EntropicallyDampedSPHSystem,
                                                   neighbor_system,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   grad_kernel, sound_speed, m_a, m_b,
                                                   p_a, p_b, rho_a, rho_b, v_a, v_b)
    (; correction) = particle_system

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    # This technique by Basa et al. 2017 (10.1002/fld.1927) aims to reduce numerical
    # errors due to large pressures by subtracting the average pressure of neighboring
    # particles.
    # It results in significant improvement for EDAC, especially with TVF,
    # but not for WCSPH, according to Ramachandran & Puri (2019), Section 3.2.
    # Note that the return value is zero when not using average pressure reduction.
    p_avg = average_pressure(particle_system, particle)

    # With variable smoothing lengths, each pressure is divided by the `beta` of its own
    # particle and multiplied by the kernel gradient of its own particle. With
    # `pressure_acceleration_summation_density`, this is Eq. 8 in Haftu et al.
    # Without variable smoothing lengths, `beta` is one and this is the usual pressure
    # acceleration.
    beta_a = get_beta(particle_system, particle)
    beta_b = get_beta(neighbor_system, neighbor)
    dv_particle += pressure_acceleration_variable_h(particle_system, neighbor_system,
                                                    particle, neighbor, m_a, m_b,
                                                    (p_a - p_avg) / beta_a,
                                                    (p_b - p_avg) / beta_b,
                                                    rho_a, rho_b, pos_diff, distance,
                                                    grad_kernel, correction)

    dv_particle = add_dv_viscosity(dv_particle, particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system,
                                   particle, neighbor, pos_diff, distance,
                                   sound_speed, m_a, m_b, rho_a, rho_b,
                                   v_a, v_b, grad_kernel)

    dv_particle = add_dv_surface_tension(dv_particle, surface_tension_a, surface_tension_b,
                                         particle_system, neighbor_system,
                                         particle, neighbor, pos_diff, distance,
                                         rho_a, rho_b, grad_kernel, 1)

    dv_particle = add_dv_adhesion(dv_particle, surface_tension_a,
                                  particle_system, neighbor_system,
                                  particle, neighbor, pos_diff, distance)

    return dv_particle
end

@propagate_inbounds function add_continuity_equation(drho_particle,
                                                     ::ContinuityDensity,
                                                     particle_system::EntropicallyDampedSPHSystem,
                                                     neighbor_system,
                                                     particle, neighbor, pos_diff,
                                                     distance, m_b, rho_a, rho_b, v_a, v_b,
                                                     grad_kernel)
    return edac_continuity_equation(Val(has_variable_smoothing_length(particle_system)),
                                    drho_particle, particle_system, neighbor_system,
                                    particle, neighbor, pos_diff, distance,
                                    m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
end

@propagate_inbounds function edac_continuity_equation(::Val{false}, drho_particle,
                                                      particle_system, neighbor_system,
                                                      particle, neighbor, pos_diff,
                                                      distance, m_b, rho_a, rho_b,
                                                      v_a, v_b, grad_kernel)
    return add_continuity_equation(drho_particle, particle_system, neighbor_system,
                                   particle, neighbor, pos_diff, distance,
                                   m_b, rho_a, rho_b, v_a, v_b, grad_kernel)
end

# `ContinuityEquation` in the code of Haftu et al. This uses the kernel gradient of
# `particle` only, like the gather formulation of the summation density.
@propagate_inbounds function edac_continuity_equation(::Val{true}, drho_particle,
                                                      particle_system, neighbor_system,
                                                      particle, neighbor, pos_diff,
                                                      distance, m_b, rho_a, rho_b,
                                                      v_a, v_b, grad_kernel)
    grad_kernel_a,
    _ = pair_kernel_grads(particle_system, neighbor_system,
                          pos_diff, distance, particle, neighbor)

    v_diff = continuity_equation_shifting_term(v_a - v_b,
                                               shifting_technique(particle_system),
                                               particle_system, neighbor_system,
                                               particle, neighbor, rho_a, rho_b)

    return drho_particle + m_b * dot(v_diff, grad_kernel_a)
end

@inline function pressure_evolution!(dv, particle_system, neighbor_system, v_diff,
                                     grad_kernel, particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac)
    return pressure_evolution!(Val(has_variable_smoothing_length(particle_system)),
                               dv, particle_system, neighbor_system, v_diff,
                               grad_kernel, particle, neighbor,
                               pos_diff, distance, sound_speed, m_a, m_b,
                               p_a, p_b, rho_a, rho_b, nu_edac)
end

# Variable smoothing length formulation of the EDAC equation. As in the code of Haftu et al.,
# it depends on the pressure acceleration formulation: `inter_particle_averaged_pressure`
# corresponds to `EDACEquation`, all other formulations to Eq. 6 (`EDACEquation2`).
@inline function pressure_evolution!(::Val{true}, dv, particle_system, neighbor_system,
                                     v_diff, grad_kernel, particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac)
    (; reference_density) = get_refinement(particle_system)
    formulation = pressure_acceleration_formulation(particle_system)

    grad_kernel_a,
    grad_kernel_b = pair_kernel_grads(particle_system, neighbor_system,
                                      pos_diff, distance, particle, neighbor)

    beta_a = get_beta(particle_system, particle)

    artificial_eos = reference_density * sound_speed^2 * (m_b / rho_b) *
                     dot(v_diff, grad_kernel_a) / beta_a

    grad_kernel_damping,
    beta_damping = damping_kernel_grad_and_beta(neighbor_system, grad_kernel,
                                                grad_kernel_a, beta_a)
    damping_term = edac_damping_term(formulation, particle_system, neighbor_system,
                                     particle, neighbor, grad_kernel_damping, pos_diff,
                                     distance, m_a, m_b, p_a, p_b, rho_a, rho_b,
                                     beta_damping, nu_edac)

    # Correction term due to shifting, `(û - u) ⋅ ∇p` in the code of Haftu et al.,
    # where `û - u` is the shifting velocity `δv`.
    # As in their code, both `p_a` and `p_b` are reduced by the average pressure of
    # `particle`, and the `beta` of `particle` follows the same rule as in the damping term.
    p_avg = average_pressure(particle_system, particle)
    grad_pressure = edac_shifting_pressure_gradient(formulation, m_a, m_b,
                                                    p_a - p_avg, p_b - p_avg,
                                                    rho_a, rho_b, beta_damping,
                                                    get_beta(neighbor_system, neighbor),
                                                    grad_kernel_a, grad_kernel_b)
    shifting_term = dot(delta_v(particle_system, particle), grad_pressure)

    # Pressure is stored in `v` right after the velocity
    dv[ndims(particle_system) + 1,
       particle] += artificial_eos + damping_term +
                    shifting_term

    return dv
end

# Pressure gradient in the shifting term of `EDACEquation2` and `EDACEquationSolids2`
# in the code of Haftu et al. This is the negative of
# `pressure_acceleration_summation_density` for all formulations except
# `inter_particle_averaged_pressure`.
# The asymmetric version of the formulation expects `W_b` to be evaluated at `-pos_diff`.
@inline function edac_shifting_pressure_gradient(formulation, m_a, m_b, p_a, p_b,
                                                 rho_a, rho_b, beta_a, beta_b,
                                                 grad_kernel_a, grad_kernel_b)
    return -pressure_acceleration_summation_density(m_a, m_b, rho_a, rho_b,
                                                    p_a / beta_a, p_b / beta_b,
                                                    grad_kernel_a, -grad_kernel_b)
end

# Pressure gradient in the shifting term of `EDACEquation` and `EDACEquationSolids`,
# which has no `beta`
@inline function edac_shifting_pressure_gradient(::typeof(inter_particle_averaged_pressure),
                                                 m_a, m_b, p_a, p_b, rho_a, rho_b,
                                                 beta_a, beta_b,
                                                 grad_kernel_a, grad_kernel_b)
    return -inter_particle_averaged_pressure(m_a, m_b, rho_a, rho_b, p_a, p_b,
                                             grad_kernel_a, -grad_kernel_b)
end

# Fluid neighbors use the averaged kernel gradient and the `beta` of `particle` in the
# damping term (`EDACEquation2` in the code of Haftu et al.). All other neighbors use the
# kernel gradient of `particle` and no `beta` (`EDACEquationSolids2`).
# The damping term of `inter_particle_averaged_pressure` (`EDACEquation` and
# `EDACEquationSolids`) has no `beta` for any neighbor.
@inline function damping_kernel_grad_and_beta(::AbstractFluidSystem, grad_kernel,
                                              grad_kernel_a, beta_a)
    return grad_kernel, beta_a
end

@inline function damping_kernel_grad_and_beta(neighbor_system, grad_kernel,
                                              grad_kernel_a, beta_a)
    return grad_kernel_a, one(beta_a)
end

# Damping term of Eq. 6 in Haftu et al.
@inline function edac_damping_term(formulation, particle_system, neighbor_system,
                                   particle, neighbor, grad_kernel, pos_diff, distance,
                                   m_a, m_b, p_a, p_b, rho_a, rho_b, beta_a, nu_edac)
    nu_a,
    nu_b,
    denominator = edac_damping_coefficients(particle_system, neighbor_system,
                                            particle, neighbor, distance, nu_edac)

    # The harmonic mean is the formulation of Cleary (1999) (Eq. 10, Haftu et al.)
    nu_ab = 4 * nu_a * nu_b / (nu_a + nu_b)

    return (m_b / rho_b) * nu_ab * (p_a - p_b) * dot(grad_kernel, pos_diff) /
           (beta_a * denominator)
end

# Damping term of `EDACEquation` in the code of Haftu et al.
@inline function edac_damping_term(::typeof(inter_particle_averaged_pressure),
                                   particle_system, neighbor_system,
                                   particle, neighbor, grad_kernel, pos_diff, distance,
                                   m_a, m_b, p_a, p_b, rho_a, rho_b, beta_a, nu_edac)
    nu_a,
    nu_b,
    denominator = edac_damping_coefficients(particle_system, neighbor_system,
                                            particle, neighbor, distance, nu_edac)

    eta_a = rho_a * nu_a
    eta_b = rho_b * nu_b
    eta_ab = 2 * eta_a * eta_b / (eta_a + eta_b)

    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    return volume_term * eta_ab * (p_a - p_b) * dot(grad_kernel, pos_diff) / denominator
end

@inline function edac_damping_coefficients(particle_system, neighbor_system,
                                           particle, neighbor, distance, nu_edac)
    h_a, h_b = pair_smoothing_lengths(particle_system, neighbor_system, particle, neighbor)
    h_reference = initial_smoothing_length(particle_system)

    # The EDAC viscosity is proportional to the smoothing length (Eq. 11, Haftu et al.)
    nu_a = nu_edac * h_a / h_reference
    nu_b = nu_edac * h_b / h_reference

    # This denominator is missing in Eq. 6 in Haftu et al., but it is part of their code
    h_ab = (h_a + h_b) / 2
    denominator = distance^2 + h_ab^2 / 100

    return nu_a, nu_b, denominator
end

@inline function pressure_evolution!(::Val{false}, dv, particle_system, neighbor_system,
                                     v_diff, grad_kernel, particle, neighbor,
                                     pos_diff, distance, sound_speed, m_a, m_b,
                                     p_a, p_b, rho_a, rho_b, nu_edac)
    volume_a = m_a / rho_a
    volume_b = m_b / rho_b
    volume_term = (volume_a^2 + volume_b^2) / m_a

    # EDAC pressure evolution
    pressure_diff = p_a - p_b

    # This is basically the continuity equation times `sound_speed^2`
    artificial_eos = m_b * rho_a / rho_b * sound_speed^2 * dot(v_diff, grad_kernel)

    eta_a = rho_a * nu_edac
    eta_b = rho_b * nu_edac
    eta_tilde = 2 * eta_a * eta_b / (eta_a + eta_b)

    smoothing_length_average = (smoothing_length(particle_system, particle) +
                                smoothing_length(neighbor_system, neighbor)) / 2
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
    damping_term = volume_term * tmp * pressure_diff * dot(grad_kernel, pos_diff)

    # Pressure is stored in `v` right after the velocity
    dv[ndims(particle_system) + 1, particle] += artificial_eos + damping_term

    return dv
end
