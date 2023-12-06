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

    viscosity = viscosity_model(neighbor_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # In order to visualize quantities like pressure forces or viscosity forces, uncomment
    # the following code and the two other lines below that are marked as "debug example".
    # debug_array = zeros(ndims(particle_system), nparticles(particle_system))

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    for_particle_neighbor(particle_system, neighbor_system,
                          system_coords, neighbor_system_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        # Only consider particles with a distance > 0
        distance < sqrt(eps()) && return

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

        dv_pressure = pressure_acceleration(pressure_correction, m_b, particle, neighbor,
                                            particle_system, neighbor_system,
                                            rho_a, rho_b, pos_diff,
                                            distance, grad_kernel, density_calculator)

        dv_viscosity = viscosity_correction *
                       viscosity(particle_system, neighbor_system,
                                 v_particle_system, v_neighbor_system,
                                 particle, neighbor, pos_diff, distance,
                                 sound_speed, m_a, m_b, rho_mean)

        for i in 1:ndims(particle_system)
            dv[i, particle] += dv_pressure[i] + dv_viscosity[i]
            # Debug example
            # debug_array[i, particle] += dv_pressure[i]
        end

        # if variable smoothing_length is used this uses the neighbor smoothing length
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

# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `ContinuityDensity` with the formulation `\rho_a * \sum m_b / \rho_b ...`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
@inline function pressure_acceleration(pressure_correction, m_b, particle, neighbor,
                                       particle_system,
                                       neighbor_system::WeaklyCompressibleSPHSystem,
                                       rho_a, rho_b, pos_diff, distance,
                                       grad_kernel, ::ContinuityDensity)
    return (-m_b *
            (particle_system.pressure[particle] + neighbor_system.pressure[neighbor]) /
            (rho_a * rho_b) * grad_kernel) *
           pressure_correction
end

# As shown in "Variational and momentum preservation aspects of Smooth Particle Hydrodynamic
# formulations" by Bonet and Lok (1999), for a consistent formulation this form has to be
# used with `SummationDensity`.
# This can also be seen in the tests for total energy conservation, which fail with the
# other `pressure_acceleration` form.
@inline function pressure_acceleration(pressure_correction, m_b, particle, neighbor,
                                       particle_system,
                                       neighbor_system::WeaklyCompressibleSPHSystem,
                                       rho_a, rho_b, pos_diff, distance,
                                       grad_kernel, ::SummationDensity)
    return (-m_b *
            (particle_system.pressure[particle] / rho_a^2 +
             neighbor_system.pressure[neighbor] / rho_b^2) * grad_kernel) *
           pressure_correction
end

@inline function pressure_acceleration(pressure_correction, m_b, particle, neighbor,
                                       particle_system,
                                       neighbor_system::Union{BoundarySPHSystem,
                                                              TotalLagrangianSPHSystem},
                                       rho_a, rho_b, pos_diff, distance,
                                       grad_kernel, density_calculator)
    (; boundary_model) = neighbor_system

    return pressure_acceleration(pressure_correction, m_b, particle, neighbor,
                                 particle_system, neighbor_system,
                                 boundary_model, rho_a, rho_b, pos_diff,
                                 distance, grad_kernel, density_calculator)
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
