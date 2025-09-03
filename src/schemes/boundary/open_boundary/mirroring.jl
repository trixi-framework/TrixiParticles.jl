"""
    FirstOrderMirroring(; firstorder_tolerance=1f-3)

Fluid properties are extrapolated onto ghost nodes using the method proposed by [Liu and Liu (2006)](@cite Liu2006),
to extend the gradient into the boundary zone.

# Keywords
- `firstorder_tolerance=1f-3`: If the determinant of the correction matrix is smaller than this value,
                               the method falls back to [`ZerothOrderMirroring`](@ref).
"""
struct FirstOrderMirroring{ELTYPE}
    firstorder_tolerance::ELTYPE

    function FirstOrderMirroring(; firstorder_tolerance::Real=1.0f-3)
        return new{typeof(firstorder_tolerance)}(firstorder_tolerance)
    end
end

"""
    SimpleMirroring(; firstorder_tolerance=1f-3))

This method is similar to [`FirstOrderMirroring`](@ref), but does not use
the corrected gradient as proposed by [Negi et al. (2022)](@cite Negi2022).

# Keywords
- `firstorder_tolerance=1f-3`: If the determinant of the correction matrix is smaller than this value,
                               the method falls back to [`ZerothOrderMirroring`](@ref).
"""
struct SimpleMirroring{ELTYPE}
    firstorder_tolerance::ELTYPE

    function SimpleMirroring(; firstorder_tolerance::Real=1.0f-3)
        return new{typeof(firstorder_tolerance)}(firstorder_tolerance)
    end
end

"""
    ZerothOrderMirroring()

Fluid properties are interpolated onto ghost nodes using Shepard interpolation.
(See slide 6 from the 4th DualSPHysics Users Workshop:
[Tafuni, Lisbon 2018](https://dual.sphysics.org/4thusersworkshop/data/_uploaded/PDF_Talks_4thWorkshop/Tafuni_Lisbon2018.pdf)).
The position of the ghost nodes is obtained by mirroring the boundary particles
into the fluid along a direction that is normal to the open boundary.
The interpolated values at the ghost nodes are then assigned to the corresponding boundary particles.
"""
struct ZerothOrderMirroring end

@doc raw"""
    BoundaryModelTafuni(; mirror_method=FirstOrderMirroring())

Boundary model for the `OpenBoundarySPHSystem`.
This model implements the method of [Tafuni et al. (2018)](@cite Tafuni2018) to extrapolate the properties from the fluid domain
to the buffer zones (inflow and outflow) using ghost nodes.
The position of the ghost nodes is obtained by mirroring the boundary particles
into the fluid along a direction that is normal to the open boundary.
Fluid properties are then interpolated at these ghost node positions using surrounding fluid particles.
The values are then mirrored back to the boundary particles.
We provide three different mirroring methods:
    - [`ZerothOrderMirroring`](@ref): Uses a Shepard interpolation to interpolate the values.
    - [`FirstOrderMirroring`](@ref): Uses a first order correction based on the gradient of the interpolated values .
    - [`SimpleMirroring`](@ref): Similar to the first order mirroring, but does not use the gradient of the interpolated values.
"""
struct BoundaryModelTafuni{MM}
    mirror_method::MM
end

function BoundaryModelTafuni(; mirror_method=FirstOrderMirroring())
    return BoundaryModelTafuni(mirror_method)
end

function update_boundary_quantities!(system, boundary_model::BoundaryModelTafuni,
                                     v, u, v_ode, u_ode, semi, t)
    (; reference_pressure, reference_density, reference_velocity, boundary_zone,
     pressure, density, cache) = system
    (; prescribed_pressure, prescribed_density, prescribed_velocity) = cache

    @trixi_timeit timer() "extrapolate and correct values" begin
        fluid_system = corresponding_fluid_system(system, semi)

        v_fluid = wrap_v(v_ode, fluid_system, semi)
        u_fluid = wrap_u(u_ode, fluid_system, semi)

        extrapolate_values!(system, boundary_model.mirror_method, v, v_fluid,
                            u, u_fluid, semi, t; prescribed_pressure,
                            prescribed_density, prescribed_velocity)
    end

    if !prescribed_velocity && boundary_zone.average_inflow_velocity
        # When no velocity is prescribed at the inflow, the velocity is extrapolated from the fluid domain.
        # Thus, turbulent flows near the inflow can lead to a non-uniform buffer particle distribution,
        # resulting in a potential numerical instability. Averaging mitigates these effects.
        average_velocity!(v, u, system, boundary_zone, semi)
    end

    if prescribed_pressure
        @threaded semi for particle in each_moving_particle(system)
            particle_coords = current_coords(u, system, particle)

            pressure[particle] = reference_value(reference_pressure, pressure[particle],
                                                 particle_coords, t)
        end
    end

    if prescribed_density
        @threaded semi for particle in each_moving_particle(system)
            particle_coords = current_coords(u, system, particle)

            density[particle] = reference_value(reference_density, density[particle],
                                                particle_coords, t)
        end
    end

    if prescribed_velocity
        @threaded semi for particle in each_moving_particle(system)
            particle_coords = current_coords(u, system, particle)
            v_particle = current_velocity(v, system, particle)

            v_ref = reference_value(reference_velocity, v_particle, particle_coords, t)

            for dim in eachindex(v_ref)
                @inbounds v[dim, particle] = v_ref[dim]
            end
        end
    end
end

update_boundary_model!(system, ::BoundaryModelTafuni, v, u, v_ode, u_ode, semi, t) = system

function extrapolate_values!(system,
                             mirror_method::Union{FirstOrderMirroring, SimpleMirroring},
                             v_open_boundary, v_fluid, u_open_boundary, u_fluid,
                             semi, t; prescribed_density=false,
                             prescribed_pressure=false, prescribed_velocity=false)
    (; pressure, density, boundary_zone) = system

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

        interpolated_density_correction = Ref(zero(SVector{ndims(system) + 1,
                                                           eltype(system)}))

        interpolated_pressure_correction = Ref(zero(SVector{ndims(system) + 1,
                                                            eltype(system)}))

        interpolated_velocity_correction = Ref(zero(SMatrix{ndims(system),
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
                interpolated_pressure_correction[] += pressure_b * R
            end

            if !prescribed_density
                interpolated_density_correction[] += rho_b * R
            end

            if !prescribed_velocity
                interpolated_velocity_correction[] += v_b * R'
            end
        end

        pos_diff = particle_coords - ghost_node_position

        # Mirror back the ghost node values to the boundary particles
        if abs(det(correction_matrix[])) >= mirror_method.firstorder_tolerance
            L_inv = inv(correction_matrix[])

            # Pressure
            if !prescribed_pressure
                first_order_scalar_extrapolation!(pressure, particle, L_inv,
                                                  interpolated_pressure_correction[],
                                                  two_to_end, pos_diff, mirror_method)
            end

            # Density
            if !prescribed_density
                first_order_scalar_extrapolation!(density, particle, L_inv,
                                                  interpolated_density_correction[],
                                                  two_to_end, pos_diff, mirror_method)
            end

            # Velocity
            if !prescribed_velocity
                first_order_velocity_extrapolation!(v_open_boundary, particle, L_inv,
                                                    interpolated_velocity_correction[],
                                                    two_to_end, pos_diff, mirror_method)

                # Project the velocity on the normal direction of the boundary zone (only for inflow boundaries).
                # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
                # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
                # only this component of interpolated velocity is kept [...]"
                project_velocity_on_plane_normal!(v_open_boundary, system, particle,
                                                  boundary_zone)
            end

            # No else: `correction_matrix[][1, 1] <= eps()` means no fluid neighbors
            # and thus no reliable interpolation, so boundary values remain at their current state.
        elseif correction_matrix[][1, 1] > eps()
            # Determinant is small, fallback to zero-th order mirroring
            shepard_coefficient = correction_matrix[][1, 1]

            # Pressure
            if !prescribed_pressure
                # Only the first entry is used, as the subsequent entries represent gradient
                # components that are not required for zeroth-order interpolation.
                interpolated_pressure = first(interpolated_pressure_correction[])
                zeroth_order_scalar_extrapolation!(pressure, particle, shepard_coefficient,
                                                   interpolated_pressure)
            end

            # Density
            if !prescribed_density
                # Only the first entry is used, as the subsequent entries represent gradient
                # components that are not required for zeroth-order interpolation.
                interpolated_density = first(interpolated_density_correction[])
                zeroth_order_scalar_extrapolation!(density, particle, shepard_coefficient,
                                                   interpolated_density)
            end

            # Velocity
            if !prescribed_velocity
                # Only the first column is used, as the subsequent entries represent gradient
                # components that are not required for zeroth-order interpolation.
                interpolated_velocity = interpolated_velocity_correction[][:, 1]
                zeroth_order_velocity_interpolation!(v_open_boundary, particle,
                                                     shepard_coefficient,
                                                     interpolated_velocity)

                # Project the velocity on the normal direction of the boundary zone (only for inflow boundaries).
                # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
                # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
                # only this component of interpolated velocity is kept [...]"
                project_velocity_on_plane_normal!(v_open_boundary, system, particle,
                                                  boundary_zone)
            end
        end
    end

    return system
end

function extrapolate_values!(system, mirror_method::ZerothOrderMirroring,
                             v_open_boundary, v_fluid, u_open_boundary, u_fluid, semi, t;
                             prescribed_density=false, prescribed_pressure=false,
                             prescribed_velocity=false)
    (; pressure, density, boundary_zone) = system

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
            v_b = current_velocity(v_fluid, fluid_system, neighbor)

            W_ab = smoothing_kernel(fluid_system, distance, particle)

            shepard_coefficient[] += volume_b * W_ab

            if !prescribed_pressure
                interpolated_pressure[] += pressure_b * volume_b * W_ab
            end

            if !prescribed_density
                interpolated_density[] += rho_b * volume_b * W_ab
            end

            if !prescribed_velocity
                interpolated_velocity[] += v_b * volume_b * W_ab
            end
        end

        if shepard_coefficient[] > eps()
            pos_diff = particle_coords - ghost_node_position

            if !prescribed_pressure
                zeroth_order_scalar_extrapolation!(pressure, particle,
                                                   shepard_coefficient[],
                                                   interpolated_pressure[])
            end

            if !prescribed_density
                zeroth_order_scalar_extrapolation!(density, particle,
                                                   shepard_coefficient[],
                                                   interpolated_density[])
            end

            if !prescribed_velocity
                zeroth_order_velocity_interpolation!(v_open_boundary, particle,
                                                     shepard_coefficient[],
                                                     interpolated_velocity[])

                # Project the velocity on the normal direction of the boundary zone (only for inflow boundaries).
                # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
                # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
                # only this component of interpolated velocity is kept [...]"
                project_velocity_on_plane_normal!(v_open_boundary, system, particle,
                                                  boundary_zone)
            end
        end
    end

    return system
end

function zeroth_order_scalar_extrapolation!(values, particle, shepard_coefficient,
                                            extrapolated_value)
    values[particle] = extrapolated_value / shepard_coefficient

    return values
end

function zeroth_order_velocity_interpolation!(v, particle, shepard_coefficient,
                                              extrapolated_velocity)
    velocity_interpolated = extrapolated_velocity / shepard_coefficient

    for dim in eachindex(velocity_interpolated)
        @inbounds v[dim, particle] = velocity_interpolated[dim]
    end

    return v
end

function first_order_scalar_extrapolation!(values, particle, L_inv,
                                           interpolated_values_correction,
                                           two_to_end, pos_diff, ::FirstOrderMirroring)
    f_s = L_inv * interpolated_values_correction
    df_s = f_s[two_to_end] # `f_s[2:end]` as SVector

    values[particle] = f_s[1] + dot(pos_diff, df_s)

    return values
end

function first_order_scalar_extrapolation!(values, particle, L_inv,
                                           interpolated_values_correction,
                                           two_to_end, pos_diff, ::SimpleMirroring)
    f_s = L_inv * interpolated_values_correction

    values[particle] = f_s[1]

    return values
end

function first_order_velocity_extrapolation!(v, particle, L_inv,
                                             interpolated_velocity_correction,
                                             two_to_end, pos_diff, ::FirstOrderMirroring)
    for dim in eachindex(pos_diff)
        @inbounds f_v = L_inv * interpolated_velocity_correction[dim, :]
        df_v = f_v[two_to_end] # `f_v[2:end]` as SVector

        @inbounds v[dim, particle] = f_v[1] + dot(pos_diff, df_v)
    end

    return v
end

function first_order_velocity_extrapolation!(v, particle, L_inv,
                                             interpolated_velocity_correction,
                                             two_to_end, pos_diff, ::SimpleMirroring)
    for dim in eachindex(pos_diff)
        @inbounds f_v = L_inv * interpolated_velocity_correction[dim, :]

        @inbounds v[dim, particle] = f_v[1]
    end

    return v
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

    # Division inside the `sum` closure to maintain GPU compatibility
    avg_velocity = sum(candidates) do particle
        return current_velocity(v, system, particle) / length(candidates)
    end

    @threaded semi for particle in each_moving_particle(system)
        # Set the velocity of the ghost node to the average velocity of the fluid domain
        for dim in eachindex(avg_velocity)
            @inbounds v[dim, particle] = avg_velocity[dim]
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
    v_particle = current_velocity(v, system, particle)
    v_particle_projected = dot(v_particle, boundary_zone.plane_normal) *
                           boundary_zone.plane_normal

    for dim in eachindex(v_particle)
        @inbounds v[dim, particle] = v_particle_projected[dim]
    end

    return v
end
