# Two ways of providing the information to an open boundary are considered:
# - physical quantities are either assigned a priori
# - or extrapolated from the fluid domain to the buffer zones (inflow and outflow) using ghost nodes

# The position of the ghost nodes is obtained by mirroring the boundary particles
# into the fluid along a direction that is normal to the open boundary.

# In order to calculate fluid quantities at the ghost nodes, a standard particle interpolation
# would not be consistent due to the proximity of these points to an open boundary,
# which translates into the lack of a full kernel support.
struct BoundaryModelTafuni end

function update_quantities!(system, ::BoundaryModelTafuni, v, u, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "interpolate and correct values" begin
        interpolate_values!(system, v_ode, u_ode, semi, t; system.cache...)
    end
end

update_final!(system, ::BoundaryModelTafuni, v, u, v_ode, u_ode, semi, t) = system

function interpolate_values!(system, v_ode, u_ode, semi, t; prescribed_density=false,
                             prescribed_pressure=false, prescribed_velocity=false)
    (; pressure, density, fluid_system, boundary_zone,
    reference_velocity, reference_pressure) = system

    state_equation = system_state_equation(system.fluid_system)

    svector_(f, system) = SVector(ntuple(@inline(dim->f[dim + 1]), ndims(system)))

    zero_svector(system) = SVector(ntuple(@inline(dim->zero(eltype(system))),
                                          ndims(system) + 1))
    zero_smatrix(system) = SMatrix{ndims(system) + 1,
                                   ndims(system) + 1}(ntuple(@inline(i->zero(eltype(system))),
                                                             Val((ndims(system) + 1)^2)))

    v_open_boundary = wrap_v(v_ode, system, semi)
    v_fluid = wrap_v(v_ode, fluid_system, semi)
    u_open_boundary = wrap_u(u_ode, system, semi)
    u_fluid = wrap_u(u_ode, fluid_system, semi)

    neighborhood_search = get_neighborhood_search(system, fluid_system, semi)

    interpolated_velocity = [zero_svector(system) for _ in 1:ndims(system)]

    for particle in each_moving_particle(system)
        particle_coords = current_coords(u_open_boundary, system, particle)
        ghost_node_position = mirror_position(particle_coords, boundary_zone)

        # Set zero
        correction_matrix = zero_smatrix(system)
        interpolated_pressure = zero_svector(system)
        @inbounds for dim in 1:ndims(system)
            interpolated_velocity[dim] = zero_svector(system)
        end

        for neighbor in PointNeighbors.eachneighbor(ghost_node_position,
                                                    neighborhood_search)
            neighbor_coords = current_coords(u_fluid, fluid_system, neighbor)
            pos_diff = ghost_node_position - neighbor_coords
            distance2 = dot(pos_diff, pos_diff)

            if distance2 <= neighborhood_search.search_radius^2
                distance = sqrt(distance2)

                m_b = hydrodynamic_mass(fluid_system, neighbor)
                rho_b = particle_density(v_fluid, fluid_system, neighbor)
                pressure_b = particle_pressure(v_fluid, fluid_system, neighbor)
                v_b = current_velocity(v_fluid, fluid_system, neighbor)

                kernel_value = smoothing_kernel(fluid_system, distance)
                grad_kernel = smoothing_kernel_grad(fluid_system, pos_diff, distance)

                L, R = correction_arrays(kernel_value, grad_kernel, pos_diff, rho_b, m_b)

                correction_matrix += L

                !(prescribed_pressure) && (interpolated_pressure += pressure_b * R)

                !(prescribed_velocity) && @inbounds for dim in 1:ndims(system)
                    interpolated_velocity[dim] += v_b[dim] * R
                end
            end
        end

        if abs(det(correction_matrix)) < 1e-9
            L_inv = I
        else
            L_inv = inv(correction_matrix)
        end

        pos_diff = particle_coords - ghost_node_position

        if prescribed_pressure
            pressure[particle] = reference_value(reference_pressure, pressure, system,
                                                 particle, particle_coords, t)
        else
            f_p = L_inv * interpolated_pressure
            df_p = svector_(f_p, system)

            pressure[particle] = f_p[1] + dot(pos_diff, df_p)
        end

        if prescribed_velocity
            v_ref = reference_value(reference_velocity, v_open_boundary, system,
                                    particle, particle_coords, t)
            @inbounds for dim in 1:ndims(system)
                v_open_boundary[dim, particle] = v_ref[dim]
            end
        else
            @inbounds for dim in 1:ndims(system)
                f_v = L_inv * interpolated_velocity[dim]
                df_v = svector_(f_v, system)

                v_open_boundary[dim, particle] = f_v[1] + dot(pos_diff, df_v)
            end
        end

        if prescribed_density
            density[particle] = reference_value(reference_density, density, system,
                                                particle, particle_coords, t)
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

    L = V_b * SMatrix{4, 4}([W_ab W_ab*x_ab W_ab*y_ab W_ab*z_ab;
                       grad_W_ab_x grad_W_ab_x*x_ab grad_W_ab_x*y_ab grad_W_ab_x*z_ab;
                       grad_W_ab_y grad_W_ab_y*x_ab grad_W_ab_y*y_ab grad_W_ab_y*z_ab;
                       grad_W_ab_z grad_W_ab_z*x_ab grad_W_ab_z*y_ab grad_W_ab_z*z_ab])

    R = V_b * SVector(W_ab, grad_W_ab_x, grad_W_ab_y, grad_W_ab_z)

    return L, R
end

function correction_arrays(W_ab, grad_W_ab, pos_diff::SVector{2}, rho_b, m_b)
    x_ab = pos_diff[1]
    y_ab = pos_diff[2]

    grad_W_ab_x = grad_W_ab[1]
    grad_W_ab_y = grad_W_ab[2]

    V_b = m_b / rho_b

    L = V_b * SMatrix{3, 3}([W_ab W_ab*x_ab W_ab*y_ab;
                       grad_W_ab_x grad_W_ab_x*x_ab grad_W_ab_x*y_ab;
                       grad_W_ab_y grad_W_ab_y*x_ab grad_W_ab_y*y_ab])

    R = V_b * SVector(W_ab, grad_W_ab_x, grad_W_ab_y)

    return L, R
end

function mirror_position(particle_coords, boundary_zone)
    particle_position = particle_coords - boundary_zone.zone_origin
    dist = dot(particle_position, boundary_zone.flow_direction)

    return particle_coords - 2 * dist * boundary_zone.flow_direction
end
