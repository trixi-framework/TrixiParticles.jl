
# Full interaction for open boundaries using `BoundaryModelDynamicalPressureZhang`
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                   neighbor_system, semi)
    (; fluid_system, cache, boundary_model) = particle_system

    sound_speed = system_sound_speed(fluid_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # `foreach_point_neighbor` makes sure that `particle` and `neighbor` are
        # in bounds of the respective system. For performance reasons, we use `@inbounds`
        # in this hot loop to avoid bounds checking when extracting particle quantities.
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # Zhan et al. (2025):
        # "To avoid the lack of support near the buffer surface entirely, one may use the
        # angular momentum conservative form."
        dv_pressure = inter_particle_averaged_pressure(m_a, m_b, rho_a, rho_b,
                                                       p_a, p_b, grad_kernel)

        # This vanishes for particles with full kernel support
        p_boundary = cache.pressure_boundary[particle]
        dv_pressure_boundary = 2 * p_boundary * (m_b / (rho_a * rho_b)) * grad_kernel

        # Propagate `@inbounds` to the viscosity function, which accesses particle data
        dv_viscosity_ = @inbounds dv_viscosity(viscosity_model(fluid_system,
                                                               neighbor_system),
                                               particle_system, neighbor_system,
                                               v_particle_system, v_neighbor_system,
                                               particle, neighbor, pos_diff, distance,
                                               sound_speed, m_a, m_b, rho_a, rho_b,
                                               grad_kernel)

        # Extra terms in the momentum equation when using a shifting technique
        # TODO: Do we need this?
        # dv_tvf = dv_shifting(shifting_technique(fluid_system),
        #                      particle_system, neighbor_system, particle, neighbor,
        #                      v_particle_system, v_neighbor_system,
        #                      m_a, m_b, rho_a, rho_b, pos_diff, distance,
        #                      grad_kernel, correction)

        dv_particle = dv_pressure + dv_viscosity_ + dv_pressure_boundary # + dv_tvf

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        v_diff = relative_velocity(particle_system, neighbor_system,
                                   v_particle_system, v_neighbor_system, particle, neighbor)

        # Continuity equation
        @inbounds dv[end, particle] += rho_a / rho_b * m_b * dot(v_diff, grad_kernel)

        # TODO: Add density diffusion to `OpenBoundarySystem` instead of accessing it from the fluid system
        density_diffusion!(dv, density_diffusion(particle_system),
                           v_particle_system, particle, neighbor,
                           pos_diff, distance, m_b, rho_a, rho_b,
                           particle_system, grad_kernel)

        pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                            particle, neighbor, pos_diff, distance,
                            sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, fluid_system)
    end

    # This ensures that, even during stages, the velocity remains aligned with the boundary zone
    @threaded semi for particle in each_integrated_particle(particle_system)
        boundary_zone = current_boundary_zone(particle_system, particle)

        project_velocity_on_plane_normal!(dv, particle_system, particle, boundary_zone,
                                          boundary_model)
    end

    return dv
end

function pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                             particle, neighbor, pos_diff, distance,
                             sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b,
                             fluid_system::WeaklyCompressibleSPHSystem)
    return dv
end

function pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                             particle, neighbor, pos_diff, distance,
                             sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b,
                             fluid_system::EntropicallyDampedSPHSystem)
    pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                        particle, neighbor, pos_diff, distance,
                        sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, fluid_system.nu_edac)
end

function relative_velocity(particle_system, neighbor_system,
                           v_particle_system, v_neighbor_system, particle, neighbor)
    return current_velocity(v_particle_system, particle_system, particle) -
           current_velocity(v_neighbor_system, neighbor_system, neighbor)
end

# TODO: Verify the following
# For open boundaries, only the velocity component orthogonal to the boundary should affect the density.
# The tangential component (parallel to the wall) is filtered out, so only the orthogonal component remains.
# This prevents the density from being affected by tangential relative motion along the boundary wall.
# Contrary:
# only projected velocities:
# "Actually, such constraint helps the present viscous term in Eq. (13) to handle
# the lack of support along the same direction, since the normal component of the stress
# to the buffer surface has been canceled out."
# function relative_velocity(particle_system, neighbor_system::WallBoundarySystem,
#                            v_particle_system, v_neighbor_system, particle, neighbor)
#     boundary_zone = current_boundary_zone(particle_system, particle)

#     v_diff = current_velocity(v_particle_system, particle_system, particle) -
#              current_velocity(v_neighbor_system, neighbor_system, neighbor)

#     return v_diff - dot(v_diff, boundary_zone.plane_normal) * boundary_zone.plane_normal

#     # return zero(SVector{ndims(particle_system), eltype(particle_system)})
# end
