struct BoundaryModelZhang{EOS, BP}
    state_equation    :: EOS
    boundary_pressure :: BP
end

function BoundaryModelZhang(; state_equation, boundary_pressure)

    # Theoretically, spatially non-uniform boundary pressure could also be
    # imposed since Eq. (12) does not require pb to be constant along the
    # channel cross section, although it is generally not necessary.
    return BoundaryModelZhang(state_equation, boundary_pressure)
end

@inline function v_nvariables(system, boundary_model::BoundaryModelZhang)
    return ndims(system) + 1
end

@inline function current_density(v, ::BoundaryModelZhang, system::OpenBoundarySPHSystem)
    # When using `ContinuityDensity`, the density is stored in the last row of `v`
    return view(v, size(v, 1), :)
end

function write_v0!(v0, system::OpenBoundarySPHSystem, ::BoundaryModelZhang)
    (; reference_density, initial_condition) = system

    coords_svector = reinterpret(reshape, SVector{ndims(system), eltype(system)},
                                 initial_condition.coordinates)

    # Note that `.=` is very slightly faster, but not GPU-compatible
    v0[end,
       :] = copy(reference_value.(reference_density, initial_condition.density,
                                  coords_svector, 0))

    return v0
end

@inline impose_new_density!(v, u, system, boundary_model, particle, t) = v

function impose_new_density!(v, u, system, boundary_model::BoundaryModelZhang, particle, t)
    (; boundary_pressure, boundary_pressure_function, boundary_pressure) = system.cache
    (; state_equation) = boundary_model

    density = current_density(v, boundary_model, system)

    particle_coords = current_coords(u, system, particle)
    p_current = boundary_pressure[particle]

    p_boundary = reference_value(boundary_pressure_function, p_current, particle_coords, t)

    # the density of the newly populated (actually recycled) particles in the bidirectional
    # in-/outflow buffer is obtained following the boundary pressure and EoS
    @inbounds density[particle] = inverse_state_equation(state_equation, p_boundary)

    return v
end

function update_final!(system, boundary_model::BoundaryModelZhang, v, u, v_ode, u_ode,
                       semi, t)
    (; boundary_pressure, boundary_pressure_function, prescribed_pressure) = system.cache

    @threaded semi for particle in eachparticle(system)
        particle_coords = current_coords(u, system, particle)
        p_current = boundary_pressure[particle]

        boundary_pressure[particle] = reference_value(boundary_pressure_function, p_current,
                                                      particle_coords, t)

        if !(prescribed_pressure)
            rho = current_density(v, system, particle)
            system.pressure[particle] = boundary_model.state_equation(rho)
        end
    end

    return system
end

function update_boundary_quantities!(system, boundary_model::BoundaryModelZhang, v, u,
                                     v_ode, u_ode, semi, t)
    (; cache, pressure, boundary_zone, reference_density,
     reference_velocity, reference_pressure) = system
    (; prescribed_density, prescribed_pressure, prescribed_velocity) = cache

    density = current_density(v, boundary_model, system)

    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)
        if prescribed_pressure
            p_current = current_pressure(v, system, particle)
            pressure[particle] = reference_value(reference_pressure, p_current,
                                                 particle_coords, t)
        end

        if prescribed_velocity
            v_current = current_velocity(v, system, particle)
            v_ref = reference_value(reference_velocity, v_current, particle_coords, t)
            @inbounds for dim in eachindex(v_ref)
                v[dim, particle] = v_ref[dim]
            end
        end

        if prescribed_density
            rho_current = current_density(v, system, particle)
            density[particle] = reference_value(reference_density, rho_current,
                                                particle_coords, t)
        end

        project_velocity_on_plane_normal!(v, system, particle, boundary_zone,
                                          boundary_model)
    end

    return system
end

function project_velocity_on_plane_normal!(v, system, particle, boundary_zone,
                                           boundary_model)
    return v
end

function project_velocity_on_plane_normal!(v, system, particle, boundary_zone,
                                           boundary_model::BoundaryModelZhang)
    # Project `vel` on the normal direction of the boundary zone
    # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
    # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
    # only this component of interpolated velocity is kept [...]"
    vel = current_velocity(v, system, particle)
    vel_ = dot(vel, boundary_zone.plane_normal) * boundary_zone.plane_normal

    @inbounds for dim in eachindex(vel)
        v[dim, particle] = vel_[dim]
    end

    return v
end

# Interaction of boundary with other systems
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::OpenBoundarySPHSystem{<:BoundaryModelZhang},
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

        # Continuity equation
        vdiff = current_velocity(v_particle_system, particle_system, particle) -
                current_velocity(v_neighbor_system, neighbor_system, neighbor)

        @inbounds dv[end, particle] += rho_a / rho_b * m_b * dot(vdiff, grad_kernel)
    end

    return dv
end
