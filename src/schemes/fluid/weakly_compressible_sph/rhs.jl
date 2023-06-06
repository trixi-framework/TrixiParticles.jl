# Fluid-fluid interaction
<<<<<<< HEAD:src/interactions/fluid.jl
function interact!(dv, v_particle_container, u_particle_container,
                   v_neighbor_container, u_neighbor_container, neighborhood_search,
                   particle_container::FluidParticleContainer,
                   neighbor_container::FluidParticleContainer)
    @unpack density_calculator, state_equation, viscosity, smoothing_length,
    correction = particle_container
=======
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system::WeaklyCompressibleSPHSystem)
    @unpack density_calculator, state_equation, viscosity,
    smoothing_length = particle_system
>>>>>>> kernel_correction_summation_density:src/schemes/fluid/weakly_compressible_sph/rhs.jl

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0.
        distance < sqrt(eps()) && return

        # Viscosity
        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = (rho_a + rho_b) / 2

        # determine correction values
        viscosity_correction, pressure_correction = fluid_corrections(correction,
                                                                      particle_container,
                                                                      rho_mean)

        pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                          distance, rho_mean, smoothing_length)

        # Pressure forces
        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance)
        m_b = neighbor_system.mass[neighbor]
        dv_pressure = -m_b *
<<<<<<< HEAD:src/interactions/fluid.jl
                      (particle_container.pressure[particle] / rho_a^2 +
                       neighbor_container.pressure[neighbor] / rho_b^2) * grad_kernel *
                      pressure_correction
        dv_viscosity = -m_b * pi_ab * grad_kernel * viscosity_correction
=======
                      (particle_system.pressure[particle] / rho_a^2 +
                       neighbor_system.pressure[neighbor] / rho_b^2) * grad_kernel
        dv_viscosity = -m_b * pi_ab * grad_kernel
>>>>>>> kernel_correction_summation_density:src/schemes/fluid/weakly_compressible_sph/rhs.jl

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
        end

        continuity_equation!(dv, density_calculator,
                             v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             particle_system, neighbor_system)
    end

    return dv
end

@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system::WeaklyCompressibleSPHSystem,
                                      neighbor_system)
    mass = hydrodynamic_mass(neighbor_system, neighbor)
    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)
    NDIMS = ndims(particle_system)
    dv[NDIMS + 1, particle] += sum(mass * vdiff .*
                                   smoothing_kernel_grad(particle_system, pos_diff,
                                                         distance))

    return dv
end

@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      particle_system, neighbor_system)
    return dv
end

# Fluid-boundary and fluid-solid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system::Union{BoundarySPHSystem, TotalLagrangianSPHSystem})
    @unpack density_calculator, state_equation, viscosity,
    smoothing_length = particle_system
    @unpack sound_speed = state_equation

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
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # Viscosity
        v_a = current_velocity(v_particle_system, particle_system, particle)
        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        v_diff = v_a - v_b

        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = (rho_a + rho_b) / 2

        pi_ab = viscosity(sound_speed, v_diff, pos_diff, distance, rho_mean,
                          smoothing_length)
        dv_viscosity = -m_b * pi_ab *
                       smoothing_kernel_grad(particle_system, pos_diff, distance)

        # Boundary forces
        dv_boundary = boundary_particle_impact(particle, neighbor,
                                               v_particle_system, v_neighbor_system,
                                               particle_system, neighbor_system,
                                               pos_diff, distance, m_b)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_boundary[i] + dv_viscosity[i]
        end

        continuity_equation!(dv, density_calculator,
                             v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             particle_system, neighbor_system)
    end

    return dv
end
