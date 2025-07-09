@doc raw"""
    BoundaryModelTafuni()

Boundary model for the `OpenBoundarySPHSystem`.
This model implements the method of [Tafuni et al. (2018)](@cite Tafuni2018) to extrapolate the properties from the fluid domain
to the buffer zones (inflow and outflow) using ghost nodes.
The position of the ghost nodes is obtained by mirroring the boundary particles
into the fluid along a direction that is normal to the open boundary.
"""
struct BoundaryModelTafuni{MM}
    mirror_method::MM
end

struct FirstOrderMirroring{ELTYPE}
    firstorder_tolerance::ELTYPE
    function FirstOrderMirroring(; firstorder_tolerance::ELTYPE=1e-3) where {ELTYPE}
        return new{typeof(firstorder_tolerance)}(firstorder_tolerance)
    end
end

struct SimpleMirroring{ELTYPE}
    firstorder_tolerance::ELTYPE
    function SimpleMirroring(; firstorder_tolerance::Real=1e-3)
        return new{typeof(firstorder_tolerance)}(firstorder_tolerance)
    end
end

struct ZerothOrderMirroring end

function BoundaryModelTafuni(;
                             mirror_method=FirstOrderMirroring(; firstorder_tolerance=1e-3))
    return BoundaryModelTafuni(mirror_method)
end

function update_boundary_quantities!(system, boundary_model::BoundaryModelTafuni,
                                     v, u, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "extrapolate and correct values" begin
        fluid_system = corresponding_fluid_system(system, semi)

        v_open_boundary = wrap_v(v_ode, system, semi)
        v_fluid = wrap_v(v_ode, fluid_system, semi)
        u_open_boundary = wrap_u(u_ode, system, semi)
        u_fluid = wrap_u(u_ode, fluid_system, semi)

        extrapolate_values!(system, boundary_model.mirror_method, v_open_boundary, v_fluid,
                            u_open_boundary, u_fluid, semi, t; system.cache...)
    end
end

update_final!(system, ::BoundaryModelTafuni, v, u, v_ode, u_ode, semi, t) = system

function extrapolate_values!(system,
                             mirror_method::Union{FirstOrderMirroring, SimpleMirroring},
                             v_open_boundary, v_fluid, u_open_boundary, u_fluid,
                             semi, t; prescribed_density=false,
                             prescribed_pressure=false, prescribed_velocity=false)
    (; pressure, density, boundary_zone, reference_density,
     reference_velocity, reference_pressure) = system

    fluid_system = corresponding_fluid_system(system, semi)

    # Static indices to avoid allocations
    two_to_end = SVector{ndims(system)}(2:(ndims(system) + 1))

    # Use the fluid-fluid nhs, since the boundary particles are mirrored into the fluid domain
    nhs = get_neighborhood_search(fluid_system, fluid_system, semi)

    fluid_coords = current_coordinates(u_fluid, fluid_system)

    # We cannot use `foreach_point_neighbor` here because we are looking for neighbors
    # of the ghost node positions of each particle.
    # We can do this because we require the neighborhood search to support querying neighbors
    # of arbitrary positions (see `PointNeighbors.requires_update`).
    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u_open_boundary, system, particle)
        ghost_node_position = mirror_position(particle_coords, boundary_zone)

        # Use `Ref` to ensure the variables are accessible and mutable within the closure below
        # (see https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured).
        correction_matrix = Ref(zero(SMatrix{ndims(system) + 1, ndims(system) + 1,
                                             eltype(system)}))

        extrapolated_density_correction = Ref(zero(SVector{ndims(system) + 1,
                                                           eltype(system)}))

        extrapolated_pressure_correction = Ref(zero(SVector{ndims(system) + 1,
                                                            eltype(system)}))

        extrapolated_velocity_correction = Ref(zero(SMatrix{ndims(system),
                                                            ndims(system) + 1,
                                                            eltype(system)}))

        # TODO: Not public API
        PointNeighbors.foreach_neighbor(fluid_coords, nhs, particle, ghost_node_position,
                                        nhs.search_radius) do particle, neighbor, pos_diff,
                                                              distance
            m_b = hydrodynamic_mass(fluid_system, neighbor)
            rho_b = current_density(v_fluid, fluid_system, neighbor)
            pressure_b = current_pressure(v_fluid, fluid_system, neighbor)
            v_b = current_velocity(v_fluid, fluid_system, neighbor)

            kernel_value = smoothing_kernel(fluid_system, distance, particle)
            grad_kernel = smoothing_kernel_grad(fluid_system, pos_diff, distance,
                                                particle)

            # `pos_diff` corresponds to `x_{kl} = x_k - x_l` in the paper (Tafuni et al., 2018),
            # where `x_k` is the position of the ghost node and `x_l` is the position of the neighbor particle
            L, R = correction_arrays(kernel_value, grad_kernel, pos_diff, rho_b, m_b)

            correction_matrix[] += L

            if !prescribed_pressure
                extrapolated_pressure_correction[] += pressure_b * R
            end

            if !prescribed_velocity
                extrapolated_velocity_correction[] += v_b * R'
            end

            if !prescribed_density
                extrapolated_density_correction[] += rho_b * R
            end
        end

        pos_diff = particle_coords - ghost_node_position

        # Mirror back the ghost node values to the boundary particles
        if abs(det(correction_matrix[])) >= mirror_method.firstorder_tolerance
            L_inv = inv(correction_matrix[])

            # pressure
            if prescribed_pressure
                pressure[particle] = reference_value(reference_pressure, pressure[particle],
                                                     particle_coords, t)
            else
                f_p = L_inv * extrapolated_pressure_correction[]
                df_p = f_p[two_to_end] # f_p[2:end] as SVector

                gradient_part = mirror_method isa SimpleMirroring ? 0 : dot(pos_diff, df_p)

                pressure[particle] = f_p[1] + gradient_part
            end

            # density
            if prescribed_density
                density[particle] = reference_value(reference_density, density[particle],
                                                    particle_coords, t)
            else
                f_d = L_inv * extrapolated_density_correction[]
                df_d = f_d[two_to_end] # f_d[2:end] as SVector

                gradient_part = mirror_method isa SimpleMirroring ? 0 : dot(pos_diff, df_d)

                density[particle] = f_d[1] + gradient_part
            end

            # velocity
            if prescribed_velocity
                v_particle = current_velocity(v_open_boundary, system, particle)
                v_ref = reference_value(reference_velocity, v_particle, particle_coords, t)
                @inbounds for dim in eachindex(v_ref)
                    v_open_boundary[dim, particle] = v_ref[dim]
                end
            else
                @inbounds for dim in eachindex(pos_diff)
                    f_v = L_inv * extrapolated_velocity_correction[][dim, :]
                    df_v = f_v[two_to_end] # f_v[2:end] as SVector

                    gradient_part = mirror_method isa SimpleMirroring ? 0 :
                                    dot(pos_diff, df_v)

                    v_open_boundary[dim, particle] = f_v[1] + gradient_part
                end

                # Project the velocity on the normal direction of the boundary zone (only for inflow boundaries).
                # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
                # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
                # only this component of interpolated velocity is kept [...]"
                project_velocity_on_plane_normal!(v_open_boundary, system, particle,
                                                  boundary_zone)
            end

        elseif correction_matrix[][1, 1] > eps()
            # Determinant is small, fallback to zero-th order mirroring
            shepard_coefficient = correction_matrix[][1, 1]

            # pressure
            if prescribed_pressure
                pressure[particle] = reference_value(reference_pressure, pressure[particle],
                                                     particle_coords, t)
            else
                pressure[particle] = first(extrapolated_pressure_correction[]) /
                                     shepard_coefficient
            end

            # density
            if prescribed_density
                density[particle] = reference_value(reference_density, density[particle],
                                                    particle_coords, t)
            else
                density[particle] = first(extrapolated_density_correction[]) /
                                    shepard_coefficient
            end

            # velocity
            if prescribed_velocity
                v_particle = current_velocity(v_open_boundary, system, particle)
                v_ref = reference_value(reference_velocity, v_particle, particle_coords, t)
                @inbounds for dim in eachindex(v_ref)
                    v_open_boundary[dim, particle] = v_ref[dim]
                end
            else
                velocity_interpolated = extrapolated_velocity_correction[][:, 1] /
                                        shepard_coefficient

                @inbounds for dim in eachindex(velocity_interpolated)
                    v_open_boundary[dim, particle] = velocity_interpolated[dim]
                end

                # Project the velocity on the normal direction of the boundary zone (only for inflow boundaries).
                # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
                # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
                # only this component of interpolated velocity is kept [...]"
                project_velocity_on_plane_normal!(v_open_boundary, system, particle,
                                                  boundary_zone)
            end
        end
    end

    if !(prescribed_velocity) && boundary_zone.average_inflow_velocity
        # When no velocity is prescribed at the inflow, the velocity is extrapolated from the fluid domain.
        # Thus, turbulent flows near the inflow can lead to non-uniform buffer particles distribution,
        # resulting in a potential numerical instability. Averaging mitigates these effects.
        average_velocity!(v_open_boundary, u_open_boundary, system, boundary_zone, semi)
    end

    return system
end

function extrapolate_values!(system, ::ZerothOrderMirroring,
                             v_open_boundary, v_fluid, u_open_boundary, u_fluid,
                             semi, t; prescribed_density=false,
                             prescribed_pressure=false, prescribed_velocity=false)
    (; pressure, density, boundary_zone, reference_density,
     reference_velocity, reference_pressure) = system

    fluid_system = corresponding_fluid_system(system, semi)

    # Use the fluid-fluid nhs, since the boundary particles are mirrored into the fluid domain
    nhs = get_neighborhood_search(fluid_system, fluid_system, semi)

    fluid_coords = current_coordinates(u_fluid, fluid_system)

    # We cannot use `foreach_point_neighbor` here because we are looking for neighbors
    # of the ghost node positions of each particle.
    # We can do this because we require the neighborhood search to support querying neighbors
    # of arbitrary positions (see `PointNeighbors.requires_update`).
    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u_open_boundary, system, particle)
        ghost_node_position = mirror_position(particle_coords, boundary_zone)

        # Use `Ref` to ensure the variables are accessible and mutable within the closure below
        # (see https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured).
        shepard_coefficient = Ref(zero(eltype(system)))
        interpolated_density = Ref(zero(eltype(system)))
        interpolated_pressure = Ref(zero(eltype(system)))
        interpolated_velocity = Ref(zero(particle_coords))

        # TODO: Not public API
        PointNeighbors.foreach_neighbor(fluid_coords, nhs, particle, ghost_node_position,
                                        nhs.search_radius) do particle, neighbor, pos_diff,
                                                              distance
            m_b = hydrodynamic_mass(fluid_system, neighbor)
            rho_b = current_density(v_fluid, fluid_system, neighbor)
            volume_b = m_b / rho_b
            pressure_b = current_pressure(v_fluid, fluid_system, neighbor)
            vel_b = current_velocity(v_fluid, fluid_system, neighbor)

            W_ab = smoothing_kernel(fluid_system, distance, particle)

            shepard_coefficient[] += volume_b * W_ab

            if !prescribed_pressure
                interpolated_pressure[] += pressure_b * volume_b * W_ab
            end

            if !prescribed_velocity
                interpolated_velocity[] += vel_b * volume_b * W_ab
            end

            if !prescribed_density
                interpolated_density[] += rho_b * volume_b * W_ab
            end
        end

        if shepard_coefficient[] > sqrt(eps())
            interpolated_density[] /= shepard_coefficient[]
            interpolated_pressure[] /= shepard_coefficient[]
            interpolated_velocity[] /= shepard_coefficient[]
        else
            interpolated_density[] = current_density(v_open_boundary, system, particle)
            interpolated_pressure[] = current_pressure(v_open_boundary, system, particle)
            interpolated_velocity[] = current_velocity(v_open_boundary, system, particle)
        end

        pos_diff = particle_coords - ghost_node_position

        if prescribed_velocity
            v_particle = current_velocity(v_open_boundary, system, particle)
            v_ref = reference_value(reference_velocity, v_particle, particle_coords, t)
            @inbounds for dim in eachindex(v_ref)
                v_open_boundary[dim, particle] = v_ref[dim]
            end
        else
            @inbounds for dim in eachindex(pos_diff)
                v_open_boundary[dim, particle] = interpolated_velocity[][dim]
            end

            # Project the velocity on the normal direction of the boundary zone (only for inflow boundaries).
            # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
            # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
            # only this component of interpolated velocity is kept [...]"
            project_velocity_on_plane_normal!(v_open_boundary, system, particle,
                                              boundary_zone)
        end

        if prescribed_density
            density[particle] = reference_value(reference_density, density[particle],
                                                particle_coords, t)
        else
            density[particle] = interpolated_density[]
        end

        if prescribed_pressure
            pressure[particle] = reference_value(reference_pressure, pressure[particle],
                                                 particle_coords, t)
        else
            pressure[particle] = interpolated_pressure[]
        end
    end

    if !(prescribed_velocity) && boundary_zone.average_inflow_velocity
        # When no velocity is prescribed at the inflow, the velocity is extrapolated from the fluid domain.
        # Thus, turbulent flows near the inflow can lead to non-uniform buffer particles distribution,
        # resulting in a potential numerical instability. Averaging mitigates these effects.
        average_velocity!(v_open_boundary, u_open_boundary, system, boundary_zone, semi)
    end

    return system
end

function correction_arrays(W_ab, grad_W_ab, pos_diff::SVector{3}, rho_b, m_b)
    # `pos_diff` corresponds to `x_{kl} = x_k - x_l` in the paper (Tafuni et al., 2018),
    # where `x_k` is the position of the ghost node and `x_l` is the position of the neighbor particle
    # Note that in eq. (16) and (17) the indices are swapped, i.e. `x_{lk}` is used instead of `x_{kl}`.
    x_ba = -pos_diff[1]
    y_ba = -pos_diff[2]
    z_ba = -pos_diff[3]

    grad_W_ab_x = grad_W_ab[1]
    grad_W_ab_y = grad_W_ab[2]
    grad_W_ab_z = grad_W_ab[3]

    V_b = m_b / rho_b

    M = @SMatrix [W_ab W_ab*x_ba W_ab*y_ba W_ab*z_ba;
                  grad_W_ab_x grad_W_ab_x*x_ba grad_W_ab_x*y_ba grad_W_ab_x*z_ba;
                  grad_W_ab_y grad_W_ab_y*x_ba grad_W_ab_y*y_ba grad_W_ab_y*z_ba;
                  grad_W_ab_z grad_W_ab_z*x_ba grad_W_ab_z*y_ba grad_W_ab_z*z_ba]

    L = V_b * M

    R = V_b * SVector(W_ab, grad_W_ab_x, grad_W_ab_y, grad_W_ab_z)

    return L, R
end

function correction_arrays(W_ab, grad_W_ab, pos_diff::SVector{2}, rho_b, m_b)
    # `pos_diff` corresponds to `x_{kl} = x_k - x_l` in the paper (Tafuni et al., 2018),
    # where `x_k` is the position of the ghost node and `x_l` is the position of the neighbor particle
    # Note that in eq. (16) and (17) the indices are swapped, i.e. `x_{lk}` is used instead of `x_{kl}`.
    x_ba = -pos_diff[1]
    y_ba = -pos_diff[2]

    grad_W_ab_x = grad_W_ab[1]
    grad_W_ab_y = grad_W_ab[2]

    V_b = m_b / rho_b

    M = @SMatrix [W_ab W_ab*x_ba W_ab*y_ba;
                  grad_W_ab_x grad_W_ab_x*x_ba grad_W_ab_x*y_ba;
                  grad_W_ab_y grad_W_ab_y*x_ba grad_W_ab_y*y_ba]
    L = V_b * M

    R = V_b * SVector(W_ab, grad_W_ab_x, grad_W_ab_y)

    return L, R
end

function mirror_position(particle_coords, boundary_zone)
    particle_position = particle_coords - boundary_zone.zone_origin
    dist = dot(particle_position, boundary_zone.plane_normal)

    return particle_coords - 2 * dist * boundary_zone.plane_normal
end

average_velocity!(v, u, system, boundary_zone, semi) = v

function average_velocity!(v, u, system, boundary_zone::BoundaryZone{InFlow}, semi)
    (; plane_normal, zone_origin, initial_condition) = boundary_zone

    # We only use the extrapolated velocity in the vicinity of the transition region.
    # Otherwise, if the boundary zone is too large, averaging would be excessively influenced
    # by the fluid velocity further away from the boundary.
    max_dist = initial_condition.particle_spacing * 110 / 100

    candidates = findall(x -> dot(x - zone_origin, -plane_normal) <= max_dist,
                         reinterpret(reshape, SVector{ndims(system), eltype(u)},
                                     active_coordinates(u, system)))

    avg_velocity = sum(candidates) do particle
        return current_velocity(v, system, particle)
    end

    avg_velocity /= length(candidates)

    @threaded semi for particle in each_moving_particle(system)
        # Set the velocity of the ghost node to the average velocity of the fluid domain
        @inbounds for dim in eachindex(avg_velocity)
            v[dim, particle] = avg_velocity[dim]
        end
    end

    return v
end

project_velocity_on_plane_normal!(v, system, particle, boundary_zone) = v

function project_velocity_on_plane_normal!(v, system, particle,
                                           boundary_zone::BoundaryZone{InFlow})
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
