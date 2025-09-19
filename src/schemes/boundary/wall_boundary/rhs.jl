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

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(boundary_model, pos_diff, distance, particle)

        continuity_equation!(dv, density_calculator(neighbor_system),
                             m_b, rho_a, rho_b, v_diff, grad_kernel, particle)
    end

    return dv
end

# This is the derivative of the density summation, which is compatible with the
# `SummationDensity` pressure acceleration.
# Energy preservation tests will fail with the other formulation.
function continuity_equation!(dv, fluid_density_calculator::SummationDensity,
                              m_b, rho_a, rho_b, v_diff, grad_kernel, particle)
    dv[end, particle] += m_b * dot(v_diff, grad_kernel)
end

# This is identical to the continuity equation of the fluid
function continuity_equation!(dv, fluid_density_calculator::ContinuityDensity,
                              m_b, rho_a, rho_b, v_diff, grad_kernel, particle)
    dv[end, particle] += rho_a / rho_b * m_b * dot(v_diff, grad_kernel)
end
