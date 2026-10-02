# Shared structure-fluid interaction helpers used by multiple structure schemes.
@propagate_inbounds function write_fluid_force!(dv,
                                                particle_system::TotalLagrangianSPHSystem,
                                                force_particle, particle)
    material_mass = particle_system.mass[particle]
    for i in 1:ndims(particle_system)
        dv[i, particle] += force_particle[i] / material_mass
    end

    return dv
end

@propagate_inbounds function write_fluid_force!(dv, particle_system::RigidBodySystem,
                                                force_particle, particle)
    force_per_particle = particle_system.force_per_particle
    for i in 1:ndims(particle_system)
        force_per_particle[i, particle] += force_particle[i]
    end

    return dv
end

# Match the pressure and viscosity corrections used by the fluid-side RHS.
@inline function structure_fluid_force_correction(system::AbstractFluidSystem,
                                                  particle, rho_a, rho_b)
    return zero(rho_a), 1, 1
end

@inline function structure_fluid_force_correction(system::WeaklyCompressibleSPHSystem,
                                                  particle, rho_a, rho_b)
    viscosity_correction, pressure_correction,
    _ = free_surface_correction(system_correction(system), system, rho_a, rho_b)

    return zero(rho_a), viscosity_correction, pressure_correction
end

@propagate_inbounds function structure_fluid_force_correction(system::EntropicallyDampedSPHSystem,
                                                              particle, rho_a, rho_b)
    return average_pressure(system, particle), 1, 1
end

function interact_structure_fluid!(dv, v_particle_system, u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   particle_system,
                                   neighbor_system::AbstractFluidSystem, semi;
                                   eachparticle=each_integrated_particle(particle_system))
    sound_speed = system_sound_speed(neighbor_system)
    correction = system_correction(neighbor_system)
    shifting = shifting_technique(neighbor_system)
    surface_tension = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    neighborhood_search = get_neighborhood_search(particle_system, neighbor_system, semi)
    backend = semi.parallelization_backend

    # Match the fluid-side zero-distance check exactly. WCSPH scales this threshold
    # with the compact support, while EDAC and IISPH use the smoothing length.
    compact_support_ = compact_support(neighbor_system, particle_system)
    h = initial_smoothing_length(neighbor_system)
    almostzero = neighbor_system isa WeaklyCompressibleSPHSystem ?
                 sqrt(eps(compact_support_^2)) : sqrt(eps(h^2))

    @threaded semi for particle in eachparticle
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure particles
        # corresponding to the rest density of the fluid and not the material density.
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        v_a = @inbounds current_velocity(v_particle_system, particle_system, particle)

        # Accumulate force components and density rate in one static vector.
        init = zero(SVector{ndims(particle_system) + 1, eltype(particle_system)})

        # Keep the returned name out of the closure to avoid allocations.
        force_drho_particle_ = @inbounds mapreduce_neighbor(+, system_coords,
                                                            neighbor_coords,
                                                            neighborhood_search, backend,
                                                            particle;
                                                            init) do particle,
                                                                     neighbor,
                                                                     pos_diff,
                                                                     distance
            m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)
            rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)
            v_b = @inbounds current_velocity(v_neighbor_system, neighbor_system, neighbor)

            # Boundary density has its own kernel support and zero-distance check.
            drho_particle = @inbounds add_continuity_equation(zero(rho_a),
                                                              particle_system,
                                                              neighbor_system,
                                                              particle, neighbor, pos_diff,
                                                              distance, m_b, rho_a, rho_b,
                                                              v_a, v_b)

            # Only pressure/momentum contributions use the fluid-side pair cutoffs.
            init_pair = vcat(zero(v_a), SVector(drho_particle))
            distance > compact_support_ && return init_pair
            skip_zero_distance(neighbor_system) && distance < almostzero && return init_pair

            # Corrected gradients need not be odd; evaluate the fluid-side gradient.
            grad_kernel_fluid = smoothing_kernel_grad_unsafe(neighbor_system, -pos_diff,
                                                             distance, neighbor)

            # Use the hydrodynamic pressure defined by the structure's boundary model.
            p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)
            p_a = @inbounds neighbor_pressure(v_particle_system, particle_system, particle,
                                              p_b)

            p_avg, viscosity_correction,
            pressure_correction = @inbounds structure_fluid_force_correction(neighbor_system,
                                                                             neighbor,
                                                                             rho_b, rho_a)

            # Fluid-first ordering preserves the actual fluid force, including the
            # approaching-particle condition of artificial viscosity.
            dv_pressure = pressure_acceleration(neighbor_system, particle_system,
                                                neighbor, particle, m_b, m_a,
                                                p_b - p_avg, p_a - p_avg, rho_b, rho_a,
                                                -pos_diff, distance, grad_kernel_fluid,
                                                correction)
            dv_fluid = dv_pressure * pressure_correction

            dv_fluid = @inbounds add_dv_viscosity(dv_fluid, neighbor_system,
                                                  particle_system,
                                                  v_neighbor_system, v_particle_system,
                                                  neighbor, particle, -pos_diff, distance,
                                                  sound_speed, m_b, m_a, rho_b, rho_a,
                                                  v_b, v_a, grad_kernel_fluid,
                                                  viscosity_correction)

            dv_fluid = @inbounds add_dv_shifting(dv_fluid, shifting, neighbor_system,
                                                 particle_system, v_neighbor_system,
                                                 v_particle_system, neighbor, particle,
                                                 m_b, m_a, rho_b, rho_a, v_b, v_a,
                                                 -pos_diff, distance, grad_kernel_fluid,
                                                 correction)

            dv_particle = @inbounds add_dv_adhesion(-dv_fluid, surface_tension,
                                                    neighbor_system, particle_system,
                                                    neighbor, particle, pos_diff, distance)

            return vcat(m_b * dv_particle, SVector(drho_particle))
        end

        @inbounds write_fluid_force!(dv, particle_system, force_drho_particle_, particle)
        @inbounds write_drho_particle!(dv, particle_system, force_drho_particle_[end],
                                       particle)
    end

    return dv
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::AbstractStructureSystem,
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b)
    return drho_particle
end

@inline function add_continuity_equation(drho_particle,
                                         particle_system::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                                TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                         neighbor_system::AbstractFluidSystem,
                                         particle, neighbor, pos_diff, distance,
                                         m_b, rho_a, rho_b, v_a, v_b)
    # Density evolves with the boundary's own kernel and correction, not with the
    # neighboring fluid's corrected gradient evaluated in the opposite direction.
    grad_kernel = smoothing_kernel_grad(particle_system, pos_diff, distance, particle)

    return add_continuity_equation(drho_particle,
                                   density_calculator(neighbor_system),
                                   m_b, rho_a, rho_b, v_a, v_b, grad_kernel, particle)
end

@inline function write_drho_particle!(dv, ::AbstractSystem, drho_particle, particle)
    return dv
end

@propagate_inbounds function write_drho_particle!(dv,
                                                  ::Union{RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                                          TotalLagrangianSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}}},
                                                  drho_particle, particle)
    dv[end, particle] += drho_particle

    return dv
end
