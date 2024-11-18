function split_particles!(semi, v_ode, u_ode, v_ode_tmp, u_ode_tmp)
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        collect_split_candidates!(system, v, u)
    end

    resize!(semi, v_ode, u_ode, v_ode_tmp, u_ode_tmp)

    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        split_particles!(system, v, u)
    end

    return semi
end

@inline collect_split_candidates!(system, v, u) = System

@inline function collect_split_candidates!(system::FluidSystem, v, u)
    return collect_split_candidates!(system, system.particle_refinement, v, u)
end

@inline collect_split_candidates!(system::FluidSystem, ::Nothing, v, u) = system

@inline function collect_split_candidates!(system::FluidSystem, particle_refinement, v, u)
    (; mass_ref, max_spacing_ratio, refinement_pattern) = particle_refinement

    n_split_candidates = 0

    for particle in eachparticle(system)
        m_a = hydrodynamic_mass(system, particle)
        m_max = max_spacing_ratio * mass_ref[particle]

        if m_a > m_max
            n_split_candidates += 1
        end
    end

    n_childs_exclude_center = nchilds(system, refinement_pattern) - 1

    particle_refinement.n_new_particles[] = n_split_candidates * n_childs_exclude_center

    return system
end

@inline split_particles!(system, v, u) = System

@inline function split_particles!(system::FluidSystem, v, u)
    return split_particles!(system, system.particle_refinement, v, u)
end

@inline split_particles!(system::FluidSystem, ::Nothing, v, u) = system

@inline function split_particles!(system::FluidSystem, particle_refinement, v, u)
    (; smoothing_length) = system.cache
    (; mass_ref, max_spacing_ratio, refinement_pattern, n_particles_before_resize) = particle_refinement
    (; alpha, relative_position) = refinement_pattern

    for particle in eachparticle(system)
        m_a = hydrodynamic_mass(system, particle)
        m_max = max_spacing_ratio * mass_ref[particle]

        if m_a > m_max
            smoothing_length_old = smoothing_length[particle]
            mass_old = system.mass[particle]

            system.mass[particle] = mass_old / nchilds(system, refinement_pattern)

            set_particle_pressure!(v, system, partice, pressure)

            set_particle_density!(v, system, partice, density)

            smoothing_length[particle] = alpha * smoothing_length_old

            pos_center = current_coords(system, u, particle)
            vel_center = current_velocity(system, v, particle)

            for child_id_local in 1:(nchilds(system, refinement_pattern) - 1)
                child = n_particles_before_resize + child_id_local

                system.mass[child] = mass_old / nchilds(system, refinement_pattern)

                set_particle_pressure!(v, system, child, pressure)

                set_particle_density!(v, system, child, density)

                smoothing_length[child] = alpha * smoothing_length_old

                rel_pos = smoothing_length_old * relative_position[child]
                new_pos = pos_center + rel_pos

                for dim in 1:ndims(system)
                    u[dim, child] = new_pos[dim]
                    v[dim, child] = vel_center[dim]
                end
            end
        end
    end

    return system
end
