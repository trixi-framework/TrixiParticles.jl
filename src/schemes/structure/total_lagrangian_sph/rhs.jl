# Structure-structure interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::TotalLagrangianSPHSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    # Different structures do not interact with each other (yet)
    particle_system === neighbor_system || return dv

    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    interact_structure_structure!(dv, v_particle_system, particle_system, semi)
end

# Function barrier without dispatch for unit testing
@inline function interact_structure_structure!(dv, v_system, system, semi)
    (; penalty_force) = system

    # Everything here is done in the initial coordinates
    system_coords = initial_coordinates(system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    # For structure-structure interaction, this has to happen in the initial coordinates.
    foreach_point_neighbor(system, system, system_coords, system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       initial_pos_diff,
                                                                       initial_distance
        # Only consider particles with a distance > 0.
        # See `src/general/smoothing_kernels.jl` for more details.
        initial_distance^2 < eps(initial_smoothing_length(system)^2) && return

        rho_a = @inbounds system.material_density[particle]
        rho_b = @inbounds system.material_density[neighbor]

        grad_kernel = smoothing_kernel_grad(system, initial_pos_diff,
                                            initial_distance, particle)

        m_a = @inbounds system.mass[particle]
        m_b = @inbounds system.mass[neighbor]

        # PK1 / rho^2
        pk1_rho2_a = @inbounds pk1_rho2(system, particle)
        pk1_rho2_b = @inbounds pk1_rho2(system, neighbor)

        current_pos_diff_ = @inbounds current_coords(system, particle) -
                                      current_coords(system, neighbor)
        # On GPUs, convert `Float64` coordinates to `Float32` after computing the difference
        current_pos_diff = convert.(eltype(system), current_pos_diff_)
        current_distance = norm(current_pos_diff)

        dv_stress = m_b * (pk1_rho2_a + pk1_rho2_b) * grad_kernel

        dv_penalty_force_ = @inbounds dv_penalty_force(penalty_force, particle, neighbor,
                                                       initial_pos_diff, initial_distance,
                                                       current_pos_diff, current_distance,
                                                       system, m_a, m_b, rho_a, rho_b)

        dv_viscosity = @inbounds dv_viscosity_tlsph(system, v_system, particle, neighbor,
                                                    current_pos_diff, current_distance,
                                                    m_a, m_b, rho_a, rho_b, grad_kernel)

        dv_particle = dv_stress + dv_penalty_force_ + dv_viscosity

        for i in 1:ndims(system)
            @inbounds dv[i, particle] += dv_particle[i]
        end

        # TODO continuity equation for boundary model with `ContinuityDensity`?
    end

    return dv
end

# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::AbstractFluidSystem, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    # Skip interaction if TLSPH systems are integrated separately
    integrate_tlsph || return dv

    sound_speed = system_sound_speed(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # Only consider particles with a distance > 0.
        # See `src/general/smoothing_kernels.jl` for more details.
        distance^2 < eps(initial_smoothing_length(particle_system)^2) && return

        grad_kernel = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        dv_fs = structure_fluid_interaction(v_particle_system,
                                            v_neighbor_system,
                                            particle_system,
                                            neighbor_system,
                                            particle,
                                            neighbor,
                                            pos_diff,
                                            distance,
                                            sound_speed,
                                            grad_kernel, m_b, rho_a, rho_b)

        material_mass = @inbounds particle_system.mass[particle]
        for dim in eachindex(dv_fs)
            @inbounds dv[dim, particle] += dv_fs[dim] * m_b / material_mass
        end

        structure_fluid_continuity!(dv, v_particle_system, v_neighbor_system,
                                    particle, neighbor, m_b, rho_a, rho_b,
                                    particle_system, neighbor_system,
                                    grad_kernel)
    end

    return dv
end

# Structure-boundary interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::TotalLagrangianSPHSystem,
                   neighbor_system::Union{WallBoundarySystem, OpenBoundarySystem}, semi;
                   integrate_tlsph=semi.integrate_tlsph[])
    # TODO continuity equation?
    return dv
end
