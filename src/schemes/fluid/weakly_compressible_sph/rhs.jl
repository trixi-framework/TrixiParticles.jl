# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates the density
# using the continuity equation.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::WeaklyCompressibleSPHSystem, neighbor_system, semi)
    (; density_calculator, state_equation, correction) = particle_system

    sound_speed = system_sound_speed(particle_system)

    surface_tension_a = surface_tension_model(particle_system)
    surface_tension_b = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # In order to visualize quantities like pressure forces or viscosity forces, uncomment
    # the following code and the two other lines below that are marked as "debug example".
    # debug_array = zeros(ndims(particle_system), nparticles(particle_system))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_moving_particle(particle_system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = (rho_a + rho_b) / 2

        # Determine correction factors.
        # This can be ignored, as these are all 1 when no correction is used.
        (viscosity_correction, pressure_correction,
         surface_tension_correction) = free_surface_correction(correction, particle_system,
                                                               rho_mean)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        # The following call is equivalent to
        #     `p_a = current_pressure(v_particle_system, particle_system, particle)`
        #     `p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)`
        # Only when the neighbor system is a `BoundarySPHSystem` or a `TotalLagrangianSPHSystem`
        # with the boundary model `PressureMirroring`, this will return `p_b = p_a`, which is
        # the pressure of the fluid particle.
        p_a,
        p_b = @inbounds particle_neighbor_pressure(v_particle_system,
                                                   v_neighbor_system,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor)

        dv_pressure = pressure_correction *
                      pressure_acceleration(particle_system, neighbor_system,
                                            particle, neighbor,
                                            m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                            distance, grad_kernel, correction)

        # Propagate `@inbounds` to the viscosity function, which accesses particle data
        dv_viscosity_ = viscosity_correction *
                        @inbounds dv_viscosity(particle_system, neighbor_system,
                                               v_particle_system, v_neighbor_system,
                                               particle, neighbor, pos_diff, distance,
                                               sound_speed, m_a, m_b, rho_a, rho_b,
                                               grad_kernel)

        # Add convection term (only when using `TransportVelocityAdami`)
        dv_convection = dv_transport_velocity(transport_velocity(particle_system),
                                              particle_system, neighbor_system,
                                              particle, neighbor,
                                              v_particle_system, v_neighbor_system,
                                              m_a, m_b, rho_a, rho_b, pos_diff, distance,
                                              grad_kernel, correction)

        dv_surface_tension = surface_tension_correction *
                             surface_tension_force(surface_tension_a, surface_tension_b,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor, pos_diff, distance,
                                                   rho_a, rho_b, grad_kernel)

        dv_adhesion = adhesion_force(surface_tension_a, particle_system, neighbor_system,
                                     particle, neighbor, pos_diff, distance)

        for i in 1:ndims(particle_system)
            @inbounds dv[i,
                         particle] += dv_pressure[i] + dv_viscosity_[i] +
                                      dv_convection[i] + dv_surface_tension[i] +
                                      dv_adhesion[i]
            # Debug example
            # debug_array[i, particle] += dv_pressure[i]
        end

        # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
        # Propagate `@inbounds` to the continuity equation, which accesses particle data
        @inbounds continuity_equation!(dv, density_calculator, v_particle_system,
                                       v_neighbor_system, particle, neighbor,
                                       pos_diff, distance, m_b, rho_a, rho_b,
                                       particle_system, neighbor_system, grad_kernel)
    end
    # Debug example
    # periodic_box = neighborhood_search.periodic_box
    # Note: this saves a file in every stage of the integrator
    # if !@isdefined iter; iter = 0; end
    # TODO: This call should use public API. This requires some additional changes to simplify the calls.
    # trixi2vtk(v_particle_system, u_particle_system, -1.0, particle_system, periodic_box, debug=debug_array, prefix="debug", iter=iter += 1)

    return dv
end

# With 'SummationDensity', density is calculated in wcsph/system.jl:compute_density!
@inline function continuity_equation!(dv, density_calculator::SummationDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system, neighbor_system, grad_kernel)
    return dv
end

# This formulation was chosen to be consistent with the used pressure_acceleration formulations.
@propagate_inbounds function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                                  v_particle_system, v_neighbor_system,
                                                  particle, neighbor, pos_diff, distance,
                                                  m_b, rho_a, rho_b,
                                                  particle_system::WeaklyCompressibleSPHSystem,
                                                  neighbor_system, grad_kernel)
    (; density_diffusion) = particle_system

    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)

    dv[end, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)

    # Artificial density diffusion should only be applied to system(s) representing a fluid
    # with the same physical properties i.e. density and viscosity.
    # TODO: shouldn't be applied to particles on the interface (depends on PR #539)
    if particle_system === neighbor_system
        density_diffusion!(dv, density_diffusion, v_particle_system, particle, neighbor,
                           pos_diff, distance, m_b, rho_a, rho_b, particle_system,
                           grad_kernel)
    end
end

@propagate_inbounds function particle_neighbor_pressure(v_particle_system,
                                                        v_neighbor_system,
                                                        particle_system, neighbor_system,
                                                        particle, neighbor)
    p_a = current_pressure(v_particle_system, particle_system, particle)
    p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

    return p_a, p_b
end

@inline function particle_neighbor_pressure(v_particle_system, v_neighbor_system,
                                            particle_system,
                                            neighbor_system::BoundarySPHSystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                            particle, neighbor)
    p_a = current_pressure(v_particle_system, particle_system, particle)

    return p_a, p_a
end
