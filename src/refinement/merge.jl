function merge_particles!(semi, v_ode, u_ode, v_tmp, u_tmp)
    foreach_system(semi) do system
        (; delete_candidates) = system.particle_refinement
        delete_candidates .= false

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        for _ in 1:3
            merge_particles!(system, v, u)
        end
    end

    deleteat!(semi, v_ode, u_ode, v_tmp, u_tmp)

    return semi
end

@inline merge_particles!(system, v, u) = System

@inline function merge_particles!(system::FluidSystem, v, u)
    return merge_particles!(system, system.particle_refinement, v, u)
end

@inline merge_particles!(system::FluidSystem, ::Nothing, v, u) = system

@inline function merge_particles!(system::FluidSystem, particle_refinement, v, u)
    (; smoothing_kernel, cache) = system
    (; mass_ref, max_spacing_ratio, merge_candidates, delete_candidates) = particle_refinement

    set_zero!(merge_candidates)

    system_coords = current_coordinates(u, particle_system)
    neighborhood_search = get_neighborhood_search(system, semi)

    # Collect merge candidates
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           neighborhood_search) do particle, neighbor, pos_diff, distance
        particle == neighbor && return
        delete_candidates[particle] && return

        m_a = hydrodynamic_mass(system, particle)
        m_b = hydrodynamic_mass(system, neighbor)

        m_max = max_spacing_ratio * mass_ref[particle]

        if m_a <= m_max
            m_merge = m_a + m_b
            m_max_min = min(m_max, max_spacing_ratio * mass_ref[neighbor])
            if m_merge < m_max_min
                if merge_candidates[particle] == 0
                    merge_candidates[particle] = neighbor
                else
                    stored_neighbor = current_coords(u, system, merge_candidates[particle])
                    pos_diff_stored = stored_neighbor - current_coords(u, system, particle)

                    if distance < norm(pos_diff_stored)
                        merge_candidates[particle] = neighbor
                    end
                end
            end
        end
    end

    # Merge and delete particles
    for particle in merge_candidates
        iszero(merge_candidates[particle]) && continue

        candidate = merge_candidates[particle]

        if particle == merge_candidates[candidate]
            if particle < candidate
                m_a = hydrodynamic_mass(system, particle)
                m_b = hydrodynamic_mass(system, candidate)

                m_merge = m_a + m_b

                pos_a = current_coords(u, system, particle)
                pos_b = current_coords(u, system, candidate)

                vel_a = current_velocity(v, system, particle)
                vel_b = current_velocity(v, system, candidate)

                pos_merge = (m_a * pos_a + m_b * pos_b) / m_merge
                vel_merge = (m_a * vel_a + m_b * vel_b) / m_merge

                for dim in 1:ndims(system)
                    u[dim, particle] = pos_merge[dim]
                    v[dim, particle] = vel_merge[dim]
                end

                # update smoothing length
                h_a = smoothing_length(system, particle)
                h_b = smoothing_length(system, candidate)

                # TODO:
                # Check normalization_factor = smoothing_kernel(zero(pos_diff), one(h_a))
                tmp_m = m_merge * normalization_factor(smoothing_kernel, h_a)
                tmp_a = m_a * smoothing_kernel(pos_merge - pos_a, h_a)
                tmp_b = m_b * smoothing_kernel(pos_merge - pos_b, h_b)

                cache.smoothing_length[particle] = (tmp_m / (tmp_a + tmp_b))^(1 /
                                                                              ndims(system))

                system.mass[particl] = m_merge
            else
                delete_candidates[candidate] = true
            end
        end
    end

    return system
end
