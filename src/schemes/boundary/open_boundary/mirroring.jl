@doc raw"""
    BoundaryModelTafuni()

Boundary model for the `OpenBoundarySPHSystem`.
This model implements the method of [Tafuni et al. (2018)](@cite Tafuni2018) to extrapolate the properties from the fluid domain
to the buffer zones (inflow and outflow) using ghost nodes.
The position of the ghost nodes is obtained by mirroring the boundary particles
into the fluid along a direction that is normal to the open boundary.
"""
struct BoundaryModelTafuni end

function update_boundary_quantities!(system, ::BoundaryModelTafuni, v, u, v_ode, u_ode,
                                     semi, t)
    @trixi_timeit timer() "extrapolate and correct values" begin
        v_open_boundary = wrap_v(v_ode, system, semi)
        v_fluid = wrap_v(v_ode, system.fluid_system, semi)
        u_open_boundary = wrap_u(u_ode, system, semi)
        u_fluid = wrap_u(u_ode, system.fluid_system, semi)

        extrapolate_values!(system, v_open_boundary, v_fluid, u_open_boundary, u_fluid,
                            semi, t; system.cache...)
    end
end

update_final!(system, ::BoundaryModelTafuni, v, u, v_ode, u_ode, semi, t) = system

function extrapolate_values!(system, v_open_boundary, v_fluid, u_open_boundary, u_fluid,
                             semi, t; prescribed_density=false,
                             prescribed_pressure=false, prescribed_velocity=false)
    (; pressure, density, fluid_system, boundary_zone, reference_density,
     reference_velocity, reference_pressure) = system

    state_equation = system_state_equation(system.fluid_system)

    # Static indices to avoid allocations
    two_to_end = SVector{ndims(system)}(2:(ndims(system) + 1))

    # Use the fluid-fluid nhs, since the boundary particles are mirrored into the fluid domain
    neighborhood_search = get_neighborhood_search(fluid_system, fluid_system, semi)

    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u_open_boundary, system, particle)
        ghost_node_position = mirror_position(particle_coords, boundary_zone)

        # Set zero
        correction_matrix = zero(SMatrix{ndims(system) + 1, ndims(system) + 1,
                                         eltype(system)})
        extrapolated_pressure_correction = zero(SVector{ndims(system) + 1, eltype(system)})

        extrapolated_velocity_correction = zero(SMatrix{ndims(system), ndims(system) + 1,
                                                        eltype(system)})

        for neighbor in PointNeighbors.eachneighbor(ghost_node_position,
                                        neighborhood_search)
            neighbor_coords = current_coords(u_fluid, fluid_system, neighbor)
            pos_diff = ghost_node_position - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if distance2 <= neighborhood_search.search_radius^2
                distance = sqrt(distance2)

                m_b = hydrodynamic_mass(fluid_system, neighbor)
                rho_b = current_density(v_fluid, fluid_system, neighbor)
                pressure_b = current_pressure(v_fluid, fluid_system, neighbor)
                v_b = current_velocity(v_fluid, fluid_system, neighbor)

                # Project `v_b` on the normal direction of the boundary zone
                # See https://doi.org/10.1016/j.jcp.2020.110029 Section 3.3.:
                # "Because ﬂow from the inlet interface occurs perpendicular to the boundary,
                # only this component of interpolated velocity is kept [...]"
                v_b = dot(v_b, boundary_zone.plane_normal) * boundary_zone.plane_normal

                kernel_value = smoothing_kernel(fluid_system, distance, particle)
                grad_kernel = smoothing_kernel_grad(fluid_system, pos_diff, distance,
                                                    particle)

                L, R = correction_arrays(kernel_value, grad_kernel, pos_diff, rho_b, m_b)

                correction_matrix += L

                if !prescribed_pressure
                    extrapolated_pressure_correction += pressure_b * R
                end

                if !prescribed_velocity
                    extrapolated_velocity_correction += v_b * R'
                end
            end
        end

        # See also `correction_matrix_inversion_step!` for an explanation
        if abs(det(correction_matrix)) < 1.0f-9
            L_inv = I
        else
            L_inv = inv(correction_matrix)
        end

        pos_diff = particle_coords - ghost_node_position

        # In Negi et al. (2020) https://doi.org/10.1016/j.cma.2020.113119,
        # they have modified the equation for extrapolation to
        #
        #           f_0 = f_k - (r_0 - r_k) ⋅ ∇f_k
        #
        # in order to get zero gradient at the outlet interface.
        # Note: This modification is mentioned here for reference only and is NOT applied in this implementation.
        if prescribed_pressure
            pressure[particle] = reference_value(reference_pressure, pressure[particle],
                                                 particle_coords, t)
        else
            f_p = L_inv * extrapolated_pressure_correction
            df_p = f_p[two_to_end]

            pressure[particle] = f_p[1] + dot(pos_diff, df_p)
        end

        if prescribed_velocity
            v_particle = current_velocity(v_open_boundary, system, particle)
            v_ref = reference_value(reference_velocity, v_particle, particle_coords, t)
            @inbounds for dim in eachindex(v_ref)
                v_open_boundary[dim, particle] = v_ref[dim]
            end
        else
            @inbounds for dim in eachindex(pos_diff)
                f_v = L_inv * extrapolated_velocity_correction[dim, :]
                df_v = f_v[two_to_end]

                v_open_boundary[dim, particle] = f_v[1] + dot(pos_diff, df_v)
            end
        end

        # Unlike Tafuni et al. (2018), we calculate the density using the inverse state-equation
        if prescribed_density
            density[particle] = reference_value(reference_density, density[particle],
                                                particle_coords, t)
        else
            inverse_state_equation!(density, state_equation, pressure, particle)
        end
    end

    return system
end

function correction_arrays(W_ab, grad_W_ab, pos_diff::SVector{3}, rho_b, m_b)
    x_ab = pos_diff[1]
    y_ab = pos_diff[2]
    z_ab = pos_diff[3]

    grad_W_ab_x = grad_W_ab[1]
    grad_W_ab_y = grad_W_ab[2]
    grad_W_ab_z = grad_W_ab[3]

    V_b = m_b / rho_b

    M = @SMatrix [W_ab W_ab*x_ab W_ab*y_ab W_ab*z_ab;
                  grad_W_ab_x grad_W_ab_x*x_ab grad_W_ab_x*y_ab grad_W_ab_x*z_ab;
                  grad_W_ab_y grad_W_ab_y*x_ab grad_W_ab_y*y_ab grad_W_ab_y*z_ab;
                  grad_W_ab_z grad_W_ab_z*x_ab grad_W_ab_z*y_ab grad_W_ab_z*z_ab]

    L = V_b * M

    R = V_b * SVector(W_ab, grad_W_ab_x, grad_W_ab_y, grad_W_ab_z)

    return L, R
end

function correction_arrays(W_ab, grad_W_ab, pos_diff::SVector{2}, rho_b, m_b)
    x_ab = pos_diff[1]
    y_ab = pos_diff[2]

    grad_W_ab_x = grad_W_ab[1]
    grad_W_ab_y = grad_W_ab[2]

    V_b = m_b / rho_b

    M = @SMatrix [W_ab W_ab*x_ab W_ab*y_ab;
                  grad_W_ab_x grad_W_ab_x*x_ab grad_W_ab_x*y_ab;
                  grad_W_ab_y grad_W_ab_y*x_ab grad_W_ab_y*y_ab]
    L = V_b * M

    R = V_b * SVector(W_ab, grad_W_ab_x, grad_W_ab_y)

    return L, R
end

function mirror_position(particle_coords, boundary_zone)
    particle_position = particle_coords - boundary_zone.zone_origin
    dist = dot(particle_position, boundary_zone.plane_normal)

    return particle_coords - 2 * dist * boundary_zone.plane_normal
end
