"""
    interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
              u_neighbor_system, neighborhood_search, particle_system,
              neighbor_system)

Update the velocity differentials `dv` for interacting particles in a WeaklyCompressibleSPHSystem.

This function computes the interactions between particles and their neighbors within the kernel cutoff
and updates the `dv` array accordingly. It takes into account pressure forces, viscosity, and
for 'ContinuityDensity' updates the density using the continuity equation.

# Arguments
- `dv`: Array of velocity differentials to be updated.
- `particle_system`: A WeaklyCompressibleSPHSystem object.
- `neighbor_system`: A WeaklyCompressibleSPHSystem object including the same system.

# Return
- Updated `dv` array.

"""
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system::WeaklyCompressibleSPHSystem)
    @unpack density_calculator, state_equation, viscosity, smoothing_length,
    correction = particle_system
    @unpack sound_speed = state_equation

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
        rho_mean = (rho_a + rho_b) / 2

        # Determine correction values
        viscosity_correction, pressure_correction = free_surface_correction(correction,
                                                                            particle_system,
                                                                            rho_mean)

        # Pressure forces
        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance,
                                            correction=correction,
                                            kernel_correction_coefficient=
                                            kernel_correction_coefficient(particle_system,
                                                                          particle),
                                            dw_gamma=dw_gamma(particle_system, particle))

        m_a = particle_system.mass[particle]
        m_b = neighbor_system.mass[neighbor]

        dv_pressure = pressure_correction * (-m_b *
                       (particle_system.pressure[particle] / rho_a^2 +
                        neighbor_system.pressure[neighbor] / rho_b^2) * grad_kernel)

        dv_viscosity = viscosity_correction * viscosity(particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system,
                                 particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
        end

        continuity_equation!(dv, density_calculator,
                             v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             particle_system, neighbor_system, grad_kernel)
    end

    return dv
end

"""
    continuity_equation!(dv, density_calculator, v_particle_system, v_neighbor_system,
                         particle, neighbor, pos_diff, distance, particle_system, neighbor_system)

Use the continuity equation to update the density.

# Arguments
- `dv`: Array of velocity differentials to be updated with the density saved at NDIMS+1.
- `particle`: Index of the current particle.
- `neighbor`: Index of the neighbor particle.
- `pos_diff`: Difference in positions between the target particle and the neighbor particle.
- `distance`: Distance between the current particle and the neighbor particle.

# Return
- Updated `dv` array.

"""
@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::WeaklyCompressibleSPHSystem,
                                      neighbor_system, grad_kernel)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)
    NDIMS = ndims(particle_system)
    dv[NDIMS + 1, particle] += m_b * dot(vdiff, grad_kernel)

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system, neighbor_system, grad_kernel)
    return dv
end

# Fluid-boundary and fluid-solid interaction

"""
    interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
              u_neighbor_system, neighborhood_search, particle_system,
              neighbor_system)

Update the velocity differentials `dv` for interacting particles in a WeaklyCompressibleSPHSystem
and either a BoundarySPHSystem or TotalLagrangianSPHSystem.

This function computes the interactions between particles and their neighbors within the kernel
cutoff and updates the `dv` array accordingly. It takes into account viscosity and boundary
forces as well as for 'ContinuityDensity' updates the density using the continuity equation.

# Arguments
- `dv`: Array of velocity differentials to be updated.
- `particle_system`: A WeaklyCompressibleSPHSystem object.
- `neighbor_system`: Either a BoundarySPHSystem or TotalLagrangianSPHSystem.

# Return
- Updated `dv` array.

"""
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system::Union{BoundarySPHSystem, TotalLagrangianSPHSystem})
    @unpack density_calculator, state_equation, smoothing_length = particle_system
    @unpack sound_speed = state_equation
    @unpack boundary_model = neighbor_system
    @unpack viscosity = boundary_model

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)
        dv_viscosity = viscosity(particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system, particle,
                                 neighbor, pos_diff, distance, sound_speed, m_a, m_b)

        # Boundary forces
        dv_boundary = boundary_particle_impact(particle, neighbor, boundary_model,
                                               v_particle_system, v_neighbor_system,
                                               particle_system, neighbor_system,
                                               pos_diff, distance, m_b)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
        end

        continuity_equation!(dv, density_calculator,
                             v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             particle_system, neighbor_system, grad_kernel)
    end

    return dv
end
