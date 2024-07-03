# Interaction of boundary with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::Union{BoundarySystem, OpenBoundarySPHSystem},
                   neighbor_system)
    # TODO Solids and moving boundaries should be considered in the continuity equation
    return dv
end

# Interaction of outlet open boundary with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::OpenBoundarySPHSystem{<:OutFlow},
                   neighbor_system::FluidSystem)
    (; fluid_system, sound_speed) = particle_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

        p_a = particle_pressure(v_particle_system, particle_system, particle)
        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(fluid_system, pos_diff, distance)

        dv_pressure = pressure_acceleration(fluid_system, fluid_system, neighbor,
                                            m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                            distance, grad_kernel, nothing)

        dv_viscosity_ = dv_viscosity(particle_system, neighbor_system,
                                     v_particle_system, v_neighbor_system,
                                     particle, neighbor, pos_diff, distance,
                                     sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity_[i]
        end
    end

    return dv
end

# For dummy particles with `ContinuityDensity`, solve the continuity equation
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::BoundarySPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                   neighbor_system::FluidSystem)
    (; boundary_model) = particle_system
    fluid_density_calculator = neighbor_system.density_calculator

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(boundary_model, pos_diff, distance)

        continuity_equation!(dv, fluid_density_calculator, m_b, rho_a, rho_b, v_diff,
                             grad_kernel, particle)
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
