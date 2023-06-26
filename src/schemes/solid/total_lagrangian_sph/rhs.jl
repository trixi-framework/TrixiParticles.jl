# Solid-solid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem)
    interact_solid_solid!(dv, neighborhood_search, particle_system, neighbor_system)
end

# Function barrier without dispatch for unit testing
@inline function interact_solid_solid!(dv, neighborhood_search, particle_system,
                                       neighbor_system)
    @unpack penalty_force = particle_system

    # Different solids do not interact with each other (yet)
    if particle_system !== neighbor_system
        return dv
    end

    # Everything here is done in the initial coordinates
    system_coords = initial_coordinates(particle_system)
    neighbor_coords = initial_coordinates(neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    # For solid-solid interaction, this has to happen in the initial coordinates.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, initial_pos_diff,
                                                  initial_distance
        # Only consider particles with a distance > 0.
        initial_distance < sqrt(eps()) && return

        rho_a = particle_system.material_density[particle]
        rho_b = neighbor_system.material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(particle_system, initial_pos_diff,
                                            initial_distance)

        m_b = neighbor_system.mass[neighbor]

        dv_particle = m_b *
                      (pk1_corrected(particle_system, particle) / rho_a^2 +
                       pk1_corrected(neighbor_system, neighbor) / rho_b^2) *
                      grad_kernel

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_particle[i]
        end

        calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                            initial_distance, particle_system, penalty_force)

        # TODO continuity equation?
    end

    return dv
end

# Solid-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::WeaklyCompressibleSPHSystem)
    @unpack boundary_model = particle_system
    @unpack state_equation, viscosity, smoothing_length = neighbor_system
    @unpack sound_speed = state_equation

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Apply the same force to the solid particle
        # that the fluid particle experiences due to the soild particle.
        # Note that the same arguments are passed here as in fluid-solid interact!,
        # except that pos_diff has a flipped sign.
        #
        # In fluid-solid interaction, use the "hydrodynamic mass" of the solid particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # use `m_a` to get the same viscosity as for the fluid-solid direction.
        dv_viscosity = viscosity(neighbor_system, particle_system,
                                 v_neighbor_system, v_particle_system,
                                 neighbor, particle,
                                 pos_diff, distance, sound_speed, m_b, m_a)

        # Boundary forces
        dv_boundary = boundary_particle_impact(neighbor, particle, boundary_model,
                                               v_neighbor_system, v_particle_system,
                                               neighbor_system, particle_system,
                                               pos_diff, distance, m_a)
        dv_particle = dv_boundary + dv_viscosity

        for i in 1:ndims(particle_system)
            # Multiply `dv` (acceleration on fluid particle b) by the mass of
            # particle b to obtain the force.
            # Divide by the material mass of particle a to obtain the acceleration
            # of solid particle a.
            dv[i, particle] += dv_particle[i] * neighbor_system.mass[neighbor] /
                               particle_system.mass[particle]
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             particle_system, neighbor_system)
    end

    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::TotalLagrangianSPHSystem,
                                      neighbor_system::WeaklyCompressibleSPHSystem)
    return dv
end

@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::TotalLagrangianSPHSystem{
                                                                                <:BoundaryModelDummyParticles
                                                                                },
                                      neighbor_system::WeaklyCompressibleSPHSystem)
    @unpack density_calculator = particle_system.boundary_model

    continuity_equation!(dv, density_calculator,
                         v_particle_system, v_neighbor_system,
                         particle, neighbor, pos_diff, distance,
                         particle_system, neighbor_system)
end

@inline function continuity_equation!(dv, density_calculator,
                                      u_particle_system, u_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::TotalLagrangianSPHSystem,
                                      neighbor_system::WeaklyCompressibleSPHSystem)
    return dv
end

@inline function continuity_equation!(dv, ::ContinuityDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::TotalLagrangianSPHSystem,
                                      neighbor_system::WeaklyCompressibleSPHSystem)
    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)

    dv[end, particle] += sum(neighbor_system.mass[neighbor] * vdiff .*
                             smoothing_kernel_grad(neighbor_system, pos_diff,
                                                   distance))

    return dv
end

# Solid-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::BoundarySPHSystem)
    # TODO continuity equation?
    return dv
end
