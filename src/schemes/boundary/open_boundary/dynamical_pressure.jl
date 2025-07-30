struct BoundaryModelZhangDynamicalPressure end

@inline function v_nvariables(system::OpenBoundarySPHSystem,
                              boundary_model::BoundaryModelZhangDynamicalPressure)
    return ndims(system) * factor_tvf(system.fluid_system) + 1
end

@inline function add_velocity!(du, v, particle,
                               system::OpenBoundarySPHSystem{<:BoundaryModelZhangDynamicalPressure})
    add_velocity!(du, v, particle, system, system.fluid_system.transport_velocity)
end

@inline function current_density(v, ::BoundaryModelZhangDynamicalPressure,
                                 system::OpenBoundarySPHSystem)
    # When using `ContinuityDensity`, the density is stored in the last row of `v`
    return view(v, size(v, 1), :)
end

function write_v0!(v0, system::OpenBoundarySPHSystem, ::BoundaryModelZhangDynamicalPressure)
    v0[end, :] = system.initial_condition.density

    return v0
end

function write_v0!(v0, system::OpenBoundarySPHSystem, ::TransportVelocityAdami)
    (; initial_condition) = system

    for particle in eachparticle(system)
        # Write particle velocities
        for dim in 1:ndims(system)
            v0[ndims(system) + dim, particle] = initial_condition.velocity[dim, particle]
        end
    end

    return v0
end

@inline function update_transport_velocity!(system::OpenBoundarySPHSystem{<:BoundaryModelZhangDynamicalPressure},
                                            v_ode, semi)
    update_transport_velocity!(system, v_ode, semi, system.fluid_system.transport_velocity)
end

@inline impose_new_density!(v, u, system, particle, boundary_model, t) = v

function impose_new_density!(v, u, system, particle,
                             boundary_model::BoundaryModelZhangDynamicalPressure, t)
    boundary_zone = current_boundary_zone(system, particle)
    (; prescribed_density, prescribed_pressure) = boundary_zone
    (; eos_reference_density, state_equation) = system.cache

    density = current_density(v, boundary_model, system)

    if prescribed_pressure
        particle_coords = current_coords(u, system, particle)
        p_boundary = apply_reference_pressure(system, particle, particle_coords, t)

        # The density of the newly populated (actually recycled) particles in the bidirectional
        # in-/outflow buffer is obtained following the boundary pressure and EoS
        rho_0 = prescribed_density ?
                apply_reference_density(system, particle, particle_coords, t) :
                eos_reference_density

        @inbounds density[particle] = inverse_state_equation(state_equation, p_boundary,
                                                             rho_0)

    else
        @inbounds density[particle] = eos_reference_density
    end
    return v
end

function update_final!(system, boundary_model::BoundaryModelZhangDynamicalPressure, v, u,
                       v_ode, u_ode, semi, t)
    (; boundary_pressure, state_equation, eos_reference_density) = system.cache

    @threaded semi for particle in eachparticle(system)
        boundary_zone = current_boundary_zone(system, particle)
        (; prescribed_density, prescribed_pressure) = boundary_zone
        if prescribed_pressure
            pos = current_coords(u, system, particle)
            boundary_pressure[particle] = apply_reference_pressure(system, particle, pos, t)
        else
            rho = current_density(v, system, particle)
            rho_0 = prescribed_density ? apply_reference_density(system, particle, pos, t) :
                    eos_reference_density
            system.pressure[particle] = state_equation(rho, rho_0)
        end
    end

    return system
end

# Called from update callback via `update_open_boundary_eachstep!`
function update_boundary_quantities!(system,
                                     boundary_model::BoundaryModelZhangDynamicalPressure,
                                     v, u, v_ode, u_ode, semi, t)
    prescribe_reference_values!(v, u, system, semi, t)

    return system
end

function project_velocity_on_plane_normal!(v, system, particle, boundary_zone,
                                           boundary_model::BoundaryModelZhangDynamicalPressure)
    # Project `vel` on the normal direction of the boundary zone
    # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
    # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
    # only this component of interpolated velocity is kept [...]"
    v_particle = current_velocity(v, system, particle)
    v_particle_projected = dot(v_particle, boundary_zone.plane_normal) *
                           boundary_zone.plane_normal

    for dim in eachindex(v_particle)
        @inbounds v[dim, particle] = v_particle_projected[dim]
    end

    return v
end

# Interaction of boundary with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::OpenBoundarySPHSystem{<:BoundaryModelZhangDynamicalPressure},
                   neighbor_system, semi)
    (; fluid_system, cache) = particle_system
    sound_speed = system_sound_speed(fluid_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

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

        grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        p_a = current_pressure(v_particle_system, particle_system, particle)
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # "To avoid the lack of support near the buffer surface entirely, one may use the
        # angular momentum conservative form."
        dv_pressure = inter_particle_averaged_pressure(m_a, m_b, rho_a, rho_b,
                                                       p_a, p_b, grad_kernel)

        # This vanishes for particles with full kernel support
        p_boundary = cache.boundary_pressure[particle]
        dv_pressure_missing = 2 * p_boundary * (m_b / (rho_a * rho_b)) * grad_kernel

        # Propagate `@inbounds` to the viscosity function, which accesses particle data
        dv_viscosity_ = @inbounds dv_viscosity(viscosity_model(fluid_system,
                                                               neighbor_system),
                                               particle_system, neighbor_system,
                                               v_particle_system, v_neighbor_system,
                                               particle, neighbor, pos_diff, distance,
                                               sound_speed, m_a, m_b, rho_a, rho_b,
                                               grad_kernel)

        for i in 1:ndims(particle_system)
            @inbounds dv[i,
                         particle] += dv_pressure[i] + dv_viscosity_[i] +
                                      dv_pressure_missing[i]
        end

        # TODO
        # particle_coords = current_coords(u_particle_system, particle_system, particle)

        # dist_to_transition = dot(particle_coords - boundary_zone.zone_origin,
        #                          -boundary_zone.plane_normal)
        # max_dist_to_transition = boundary_zone.zone_width -
        #                         compact_support(fluid_system, fluid_system)

        # if dist_to_transition < max_dist_to_transition

        #     # Apply transport velocity if enabled.
        #     # According to Zhang et al. (2017), explicit momentum convection (see `momentum_convection`) is not strictly required:
        #     # "The extra-stress term A is not present in the momentum equation for solid dynamics as our numerical tests show that
        #     # its influence is negligible due to the well resolved velocity field. This is consistent with the observation that, for flows up
        #     # to moderate Reynolds numbers (O(10^2)), the influence of this term is negligible"
        #     transport_velocity!(dv, v_particle_system, particle_system, neighbor_system,
        #                         particle_system.fluid_system.transport_velocity,
        #                         particle, neighbor, m_a, m_b, distance, pos_diff)
        # end

        # Continuity equation
        vdiff = current_velocity(v_particle_system, particle_system, particle) -
                current_velocity(v_neighbor_system, neighbor_system, neighbor)

        @inbounds dv[end, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)
    end

    return dv
end
