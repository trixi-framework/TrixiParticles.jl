# Computes the forces that particles in `particle_system` experience from particles
# in `neighbor_system` and updates `dv` accordingly.
# It takes into account pressure forces, viscosity, and for `ContinuityDensity` updates the density
# using the continuity equation.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system, neighborhood_search,
                   particle_system::WeaklyCompressibleSPHSystem,
                   neighbor_system)
    (; density_calculator, state_equation, correction) = particle_system
    (; sound_speed) = state_equation

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # In order to visualize quantities like pressure forces or viscosity forces, uncomment
    # the following code and the two other lines below that are marked as "debug example".
    # debug_array = zeros(ndims(particle_system), nparticles(particle_system))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_system_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        rho_a = particle_density(v_particle_system, particle_system, particle)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_mean = 0.5 * (rho_a + rho_b)

        # Determine correction values
        viscosity_correction, pressure_correction = free_surface_correction(correction,
                                                                            particle_system,
                                                                            rho_mean)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        # The following call is equivalent to
        #     `p_a = particle_pressure(v_particle_system, particle_system, particle)`
        #     `p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)`
        # Only when the neighbor system is a `BoundarySPHSystem` or a `TotalLagrangianSPHSystem`
        # with the boundary model `PressureMirroring`, this will return `p_b = p_a`, which is
        # the pressure of the fluid particle.
        p_a, p_b = particle_neighbor_pressure(v_particle_system, v_neighbor_system,
                                              particle_system, neighbor_system,
                                              particle, neighbor)

        dv_pressure = pressure_correction *
                      pressure_acceleration(particle_system, neighbor_system, neighbor,
                                            m_a, m_b, p_a, p_b, rho_a, rho_b, pos_diff,
                                            distance, grad_kernel, correction)

        dv_viscosity_ = viscosity_correction *
                        dv_viscosity(particle_system, neighbor_system,
                                     v_particle_system, v_neighbor_system,
                                     particle, neighbor, pos_diff, distance,
                                     sound_speed, m_a, m_b, rho_mean)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity_[i]
            # Debug example
            # debug_array[i, particle] += dv_pressure[i]
        end

        # TODO If variable smoothing_length is used, this should use the neighbor smoothing length
        continuity_equation!(dv, density_calculator, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance, m_b, rho_a, rho_b,
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
@inline function continuity_equation!(dv, density_calculator::ContinuityDensity,
                                      v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::WeaklyCompressibleSPHSystem,
                                      neighbor_system, grad_kernel)
    (; density_diffusion) = particle_system

    vdiff = current_velocity(v_particle_system, particle_system, particle) -
            current_velocity(v_neighbor_system, neighbor_system, neighbor)

    dv[end, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)

    density_diffusion!(dv, density_diffusion, v_particle_system, v_neighbor_system,
                       particle, neighbor, pos_diff, distance, m_b, rho_a, rho_b,
                       particle_system, neighbor_system, grad_kernel)
end

@inline function density_diffusion!(dv, density_diffusion::DensityDiffusion,
                                    v_particle_system, v_neighbor_system,
                                    particle, neighbor, pos_diff, distance,
                                    m_b, rho_a, rho_b,
                                    particle_system::WeaklyCompressibleSPHSystem,
                                    neighbor_system::WeaklyCompressibleSPHSystem,
                                    grad_kernel)
    # Density diffusion terms are all zero for distance zero
    distance < sqrt(eps()) && return

    (; delta) = density_diffusion
    (; smoothing_length, state_equation) = particle_system
    (; sound_speed) = state_equation

    volume_b = m_b / rho_b

    psi = density_diffusion_psi(density_diffusion, rho_a, rho_b, pos_diff, distance,
                                particle_system, particle, neighbor)
    density_diffusion_term = dot(psi, grad_kernel) * volume_b

    dv[end, particle] += delta * smoothing_length * sound_speed * density_diffusion_term
end

# Density diffusion `nothing` or interaction other than fluid-fluid
@inline function density_diffusion!(dv, density_diffusion,
                                    v_particle_system, v_neighbor_system,
                                    particle, neighbor, pos_diff, distance,
                                    m_b, rho_a, rho_b,
                                    particle_system, neighbor_system, grad_kernel)
    return dv
end

@inline function particle_neighbor_pressure(v_particle_system, v_neighbor_system,
                                            particle_system, neighbor_system,
                                            particle, neighbor)
    p_a = particle_pressure(v_particle_system, particle_system, particle)
    p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)

    return p_a, p_b
end

@inline function particle_neighbor_pressure(v_particle_system, v_neighbor_system,
                                            particle_system,
                                            neighbor_system::BoundarySPHSystem{<:BoundaryModelDummyParticles{PressureMirroring}},
                                            particle, neighbor)
    p_a = particle_pressure(v_particle_system, particle_system, particle)

    return p_a, p_a
end

# Fluid particles collide with a boundary particle at distance =< 0 to the surface boundary
# particle to prevent them going through the first layer of boundary particles
@inline function collision_interact!(dv, v_particle_system, u_particle_system,
                                     v_neighbor_system, u_neighbor_system,
                                     neighborhood_search, particle_system::FluidSystem,
                                     neighbor_system::Union{BoundarySPHSystem,
                                                            RigidSPHSystem,
                                                            TotalLagrangianSPHSystem})
    # (particle_spacing) = neighbor_system

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_system_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        if distance <= sqrt(eps())
            # todo: not correct
            normal = pos_diff / distance

            part_v = extract_svector(v_particle_system, particle_system, particle)
            nghbr_v = extract_svector(v_neighbor_system, neighbor_system, neighbor)
            # Relative velocity
            rel_vel_normal = dot(part_v - nghbr_v, normal)

            collision_damping_coefficient = 0.1
            force_magnitude = collision_damping_coefficient * rel_vel_normal
            force = force_magnitude * normal

            @inbounds for i in 1:ndims(particle_system)
                dv[i, particle] += force[i] / mass[particle]
            end
        end
    end
    return dv
end
