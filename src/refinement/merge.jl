function merge_particles!(semi, v_ode, u_ode, v_tmp, u_tmp)
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        merge_particles!(system, semi, v, u)
    end

    deleteat!(semi, v_ode, u_ode, v_tmp, u_tmp)

    return semi
end

@inline merge_particles!(system, semi, v, u) = system

@inline function merge_particles!(system::FluidSystem, semi, v, u)
    return merge_particles!(system, system.particle_refinement, semi, v, u)
end

@inline merge_particles!(system::FluidSystem, ::Nothing, semi, v, u) = system

@inline function merge_particles!(system::FluidSystem, particle_refinement, semi, v, u)
    (; delete_candidates) = system.cache

    resize!(delete_candidates, nparticles(system))
    delete_candidates .= false

    resize_refinement!(system, particle_refinement)

    # Merge particles iteratively
    for _ in 1:3
        merge_particles_inner!(system, particle_refinement, semi, v, u)
    end

    return system
end

function merge_particles_inner!(system, particle_refinement, semi, v, u)
    (; smoothing_kernel, cache) = system
    (; mass_ref, max_spacing_ratio, merge_candidates) = particle_refinement

    set_zero!(merge_candidates)

    system_coords = current_coordinates(u, system)

    # Collect merge candidates
    foreach_point_neighbor(system, system, system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        particle == neighbor && return
        cache.delete_candidates[particle] && return

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
    for particle in findall(!iszero, merge_candidates)
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

                tmp_a = m_a * kernel(smoothing_kernel, norm(pos_merge - pos_a), h_a)
                tmp_b = m_b * kernel(smoothing_kernel, norm(pos_merge - pos_b), h_b)

                cache.smoothing_length[particle] = (tmp_m / (tmp_a + tmp_b))^(1 /
                                                                              ndims(system))

                system.mass[particle] = m_merge
            else
                cache.delete_candidates[candidate] = true
            end
        end
    end

    return system
end
