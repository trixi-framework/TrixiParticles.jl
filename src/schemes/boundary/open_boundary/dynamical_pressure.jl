# TODO: Should we ensure that the solid boundary extends beyond the `BoundaryZone` to guarantee complete kernel support for all particles?
struct BoundaryModelDynamicalPressureZhang end

@inline function v_nvariables(system::OpenBoundarySPHSystem,
                              boundary_model::BoundaryModelDynamicalPressureZhang)
    v_nvariables(boundary_model, system, system.fluid_system)
end

@inline function v_nvariables(::BoundaryModelDynamicalPressureZhang,
                              system::OpenBoundarySPHSystem, ::WeaklyCompressibleSPHSystem)
    # Velocity and density is integrated
    return ndims(system) + 1
end

@inline function v_nvariables(::BoundaryModelDynamicalPressureZhang,
                              system::OpenBoundarySPHSystem, ::EntropicallyDampedSPHSystem)
    # Velocity, density and pressure is integrated
    return ndims(system) + 2
end

@inline function current_density(v, ::BoundaryModelDynamicalPressureZhang,
                                 system::OpenBoundarySPHSystem)
    return view(v, ndims(system) + 1, :)
end

@inline function current_pressure(v, boundary_model::BoundaryModelDynamicalPressureZhang,
                                  system::OpenBoundarySPHSystem)
    current_pressure(v, boundary_model, system, system.fluid_system)
end

@inline function current_pressure(v, ::BoundaryModelDynamicalPressureZhang,
                                  system::OpenBoundarySPHSystem,
                                  ::WeaklyCompressibleSPHSystem)
    return system.cache.pressure
end

@inline function current_pressure(v, ::BoundaryModelDynamicalPressureZhang,
                                  system::OpenBoundarySPHSystem,
                                  ::EntropicallyDampedSPHSystem)
    # When using `EntropicallyDampedSPHSystem`, the pressure is stored in the last row of `v`
    return view(v, size(v, 1), :)
end

@inline impose_rest_density!(v, system, particle, boundary_model) = v

@inline function impose_rest_density!(v, system, particle,
                                      boundary_model::BoundaryModelDynamicalPressureZhang)
    (; density_rest, pressure_boundary) = system.cache

    # Density of recycled buffer particles is obtained following the EoS (Eq. 15, Zhang et al. 2015)
    density = density_rest +
              pressure_boundary[particle] / system_sound_speed(system.fluid_system)^2

    set_particle_density!(v, system, particle, density)
end

@inline impose_rest_pressure!(v, system, particle, boundary_model) = v

@inline function impose_rest_pressure!(v, system, particle,
                                       boundary_model::BoundaryModelDynamicalPressureZhang)
    set_particle_pressure!(v, system, particle, system.cache.pressure_rest)
end

function write_v0!(v0, system::OpenBoundarySPHSystem, ::BoundaryModelDynamicalPressureZhang)
    write_v0!(v0, system, system.boundary_model, system.fluid_system)
end

function write_v0!(v0, system::OpenBoundarySPHSystem, ::BoundaryModelDynamicalPressureZhang,
                   ::EntropicallyDampedSPHSystem)
    v0[size(v0, 1) - 1, :] = system.initial_condition.density
    v0[size(v0, 1), :] = system.initial_condition.pressure

    return v0
end

function write_v0!(v0, system::OpenBoundarySPHSystem, ::BoundaryModelDynamicalPressureZhang,
                   ::WeaklyCompressibleSPHSystem)
    v0[size(v0, 1), :] = system.initial_condition.density

    return v0
end

function update_boundary_model!(system, boundary_model::BoundaryModelDynamicalPressureZhang,
                                v, u, v_ode, u_ode, semi, t)
    (; pressure_boundary, pressure_rest) = system.cache

    compute_pressure!(system, system.fluid_system, v, semi)

    @threaded semi for particle in each_moving_particle(system)
        boundary_zone = current_boundary_zone(system, particle)

        particle_coords = current_coords(u, system, particle)

        # TODO: Clarify the author's intention here
        # From Zhang et al. (2025):
        #   "When the bidirectional in-/outlet buffer works with the velocity
        #   in-/outflow boundary condition, such as in PIVO (Pressurized Inlet,
        #   Velocity Outlet) and VIPO (Velocity Inlet, Pressure Outlet) flows, the
        #   pressure boundary condition should also be imposed at the velocity
        #   in-/outlet to eliminate the truncated error in approximating pressure gradient,
        #   but the corresponding pb in Eq. (13) is given as pi.
        #   Meanwhile, both the density and pressure of newly populated particles remain unchanged."
        if boundary_zone.prescribed_pressure
            # TODO: How do we prescibe zero pressure? Do we have to also update the momentum pressure?
            pressure_boundary[particle] = reference_pressure(boundary_zone, v, system,
                                                             particle, particle_coords, t)
        else
            pressure_boundary[particle] = pressure_rest
        end
    end

    return system
end

function compute_pressure!(system::OpenBoundarySPHSystem,
                           fluid_system::EntropicallyDampedSPHSystem, v, semi)
    return system
end

function compute_pressure!(system::OpenBoundarySPHSystem,
                           fluid_system::WeaklyCompressibleSPHSystem, v, semi)
    @threaded semi for particle in eachparticle(system)
        apply_state_equation!(system, fluid_system, current_density(v, system, particle),
                              particle)
    end

    return system
end

# Use this function to avoid passing closures to Polyester.jl with `@batch` (`@threaded`).
# Otherwise, `@threaded` does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function apply_state_equation!(system::OpenBoundarySPHSystem,
                                       fluid_system::WeaklyCompressibleSPHSystem, density,
                                       particle)
    system.cache.pressure[particle] = fluid_system.state_equation(density)
end

# Called from update callback via `update_open_boundary_eachstep!`
function update_boundary_quantities!(system,
                                     boundary_model::BoundaryModelDynamicalPressureZhang,
                                     v, u, v_ode, u_ode, semi, t)
    (; pressure_boundary, pressure_rest) = system.cache

    @threaded semi for particle in each_moving_particle(system)
        boundary_zone = current_boundary_zone(system, particle)
        (; prescribed_density, prescribed_pressure, prescribed_velocity) = boundary_zone

        particle_coords = current_coords(u, system, particle)

        if prescribed_pressure
            # TODO: How do we prescibe zero pressure? Do we have to also update the momentum pressure?
            pressure_boundary[particle] = reference_pressure(boundary_zone, v, system,
                                                             particle, particle_coords, t)
        else
            pressure_boundary[particle] = pressure_rest
        end

        if prescribed_density
            rho_ref = reference_density(boundary_zone, v, system, particle,
                                        particle_coords, t)
            set_particle_density!(v, system, particle, rho_ref)
        end

        if prescribed_velocity
            v_ref = reference_velocity(boundary_zone, v, system, particle,
                                       particle_coords, t)

            for dim in eachindex(v_ref)
                @inbounds v[dim, particle] = v_ref[dim]
            end
        end

        # Project the velocity on the normal direction of the boundary zone
        # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
        # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
        # only this component of interpolated velocity is kept [...]"
        project_velocity_on_plane_normal!(v, system, particle, boundary_zone,
                                          boundary_model)
    end

    return system
end

function project_velocity_on_plane_normal!(v, system, particle, boundary_zone,
                                           boundary_model::BoundaryModelDynamicalPressureZhang)
    # Project the velocity on the normal direction of the boundary zone
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

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::OpenBoundarySPHSystem{<:BoundaryModelDynamicalPressureZhang},
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

        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # "To avoid the lack of support near the buffer surface entirely, one may use the
        # angular momentum conservative form."
        dv_pressure = inter_particle_averaged_pressure(m_a, m_b, rho_a, rho_b,
                                                       p_a, p_b, grad_kernel)

        # This is zero if not prescribed or vanishes for particles with full kernel support
        p_p = cache.pressure_boundary[particle]
        dv_pressure_boundary = 2 * p_p * (m_b / (rho_a * rho_b)) * grad_kernel

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

        for i in 1:ndims(particle_system)
            @inbounds dv[i,
                         particle] += dv_pressure[i] + dv_viscosity_[i] + #dv_tvf[i] +
                                      dv_pressure_boundary[i]
        end

        v_diff = current_velocity(v_particle_system, particle_system, particle) -
                 current_velocity(v_neighbor_system, neighbor_system, neighbor)

        # Continuity equation
        @inbounds dv[ndims(particle_system) + 1,
                     particle] += rho_a / rho_b * m_b * dot(v_diff, grad_kernel)

        pressure_evolution!(dv, particle_system, neighbor_system, v_diff, grad_kernel,
                            particle, neighbor, pos_diff, distance,
                            sound_speed, m_a, m_b, p_a, p_b, rho_a, rho_b, fluid_system)
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
