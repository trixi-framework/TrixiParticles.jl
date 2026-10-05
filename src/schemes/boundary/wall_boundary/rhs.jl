# Interaction of boundary with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::Union{AbstractBoundarySystem, OpenBoundarySystem},
                   neighbor_system, semi)
    # TODO Solids and moving boundaries should be considered in the continuity equation
    return dv
end

# For dummy particles with `ContinuityDensity`, solve the continuity equation
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WallBoundarySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                   neighbor_system::Union{AbstractFluidSystem,
                                          OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang}},
                   semi)
    (; boundary_model) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Some unsafe kernel gradients divide by `distance`, so effectively coincident particles
    # are treated as numerically zero to avoid division by zero and handle rounding errors.
    # This is a numerical convention; not every kernel has a zero-gradient limit.
    # Scale the cutoff by the smoothing length `h`: `distance^2 < eps(typeof(h)) * h^2`.
    # Comparing squared distances directly avoids computing a square root.
    # Note that `sqrt(eps(typeof(h))) * h != eps(h)`.
    h = initial_smoothing_length(particle_system)
    zero_distance_squared = eps(typeof(h)) * h^2

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # Skip numerically zero separations only when the correction permits it.
        # Note that `return` only exits the closure, i.e., skips the current neighbor.
        skip_zero_distance(particle_system) && distance^2 < zero_distance_squared && return

        # The correction-dependent check makes it safe to call the unsafe kernel gradient
        # without repeating its zero-distance check.
        grad_kernel = smoothing_kernel_grad_unsafe(particle_system, pos_diff,
                                                   distance, particle)

        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)

        drho_particle = add_continuity_equation(zero(rho_a),
                                                density_calculator(neighbor_system),
                                                m_b, rho_a, rho_b, v_a, v_b,
                                                grad_kernel, particle)

        dv[end, particle] += drho_particle
    end

    return dv
end

# This is the derivative of the density summation, which is compatible with the
# `SummationDensity` pressure acceleration.
# Energy preservation tests will fail with the other formulation.
@propagate_inbounds function add_continuity_equation(drho_particle,
                                                     ::SummationDensity,
                                                     m_b, rho_a, rho_b, v_a, v_b,
                                                     grad_kernel, particle)
    return drho_particle + m_b * dot(v_a - v_b, grad_kernel)
end

# This is identical to the continuity equation of the fluid
@propagate_inbounds function add_continuity_equation(drho_particle,
                                                     ::ContinuityDensity,
                                                     m_b, rho_a, rho_b, v_a, v_b,
                                                     grad_kernel, particle)
    return drho_particle + rho_a / rho_b * m_b * dot(v_a - v_b, grad_kernel)
end
