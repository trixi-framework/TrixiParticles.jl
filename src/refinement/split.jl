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
    (; mass_ref, max_spacing_ratio, split_candidates,
     refinement_pattern) = particle_refinement
    (; n_children, center_particle) = refinement_pattern

    resize!(split_candidates, 0)

    for particle in eachparticle(system)
        m_a = hydrodynamic_mass(system, particle)
        m_max = max_spacing_ratio * mass_ref[particle]

        if m_a > m_max
            push!(split_candidates, particle)
        end
    end

    particle_refinement.n_new_particles[] = length(split_candidates) *
                                            (n_children - center_particle)

    # Set capacity for resizing
    system.cache.additional_capacity[] = particle_refinement.n_new_particles[]

    return system
end

@inline split_particles!(system, v, u) = System

@inline function split_particles!(system::FluidSystem, v, u)
    return split_particles!(system, system.particle_refinement, v, u)
end

@inline split_particles!(system::FluidSystem, ::Nothing, v, u) = system

@inline function split_particles!(system::FluidSystem, particle_refinement, v, u)
    (; smoothing_length) = system.cache
    (; split_candidates, refinement_pattern, n_new_particles) = particle_refinement
    (; alpha, relative_position, n_children, center_particle) = refinement_pattern

    child_id_global = nparticles(system) - n_new_particles[]
    for particle in split_candidates
        smoothing_length_old = smoothing_length[particle]
        mass_old = system.mass[particle]

        system.mass[particle] = mass_old / n_children

        p_a = particle_pressure(v, system, particle)
        rho_a = particle_density(v, system, particle)

        smoothing_length[particle] = alpha * smoothing_length_old

        pos_center = current_coords(u, system, particle)
        vel_center = current_velocity(v, system, particle)

        for child_id_local in 1:(n_children - center_particle)
            child_id_global += 1

            system.mass[child_id_global] = mass_old / n_children

            set_particle_pressure!(v, system, child_id_global, p_a)

            set_particle_density!(v, system, child_id_global, rho_a)

            smoothing_length[child_id_global] = alpha * smoothing_length_old

            rel_pos = smoothing_length_old * relative_position[child_id_local]
            new_pos = pos_center + rel_pos

            for dim in 1:ndims(system)
                u[dim, child_id_global] = new_pos[dim]
                v[dim, child_id_global] = vel_center[dim]
            end
        end
    end

    return system
end
