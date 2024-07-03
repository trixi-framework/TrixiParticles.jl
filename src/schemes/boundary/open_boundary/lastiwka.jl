struct BoundaryModelLastiwka end

@inline function update_quantities!(system, ::BoundaryModelLastiwka,
                                    v, u, v_ode, u_ode, semi, t)
    (; density, pressure, cache, flow_direction, sound_speed,
    reference_velocity, reference_pressure, reference_density) = system

    # Update quantities based on the characteristic variables
    @threaded system for particle in each_moving_particle(system)
        particle_position = current_coords(u, system, particle)

        J1 = cache.characteristics[1, particle]
        J2 = cache.characteristics[2, particle]
        J3 = cache.characteristics[3, particle]

        rho_ref = reference_density(particle_position, t)
        density[particle] = rho_ref + ((-J1 + 0.5 * (J2 + J3)) / sound_speed^2)

        p_ref = reference_pressure(particle_position, t)
        pressure[particle] = p_ref + 0.5 * (J2 + J3)

        v_ref = reference_velocity(particle_position, t)
        rho = density[particle]
        v_ = v_ref + ((J2 - J3) / (2 * sound_speed * rho)) * flow_direction

        for dim in 1:ndims(system)
            v[dim, particle] = v_[dim]
        end
    end

    return system
end

function update_final!(system, ::BoundaryModelLastiwka, v, u, v_ode, u_ode, semi, t)
    @trixi_timeit timer() "evaluate characteristics" evaluate_characteristics!(system, v, u,
                                                                               v_ode, u_ode,
                                                                               semi, t)
end

# ==== Characteristics
# J1: Associated with convection and entropy and propagates at flow velocity.
# J2: Propagates downstream to the local flow
# J3: Propagates upstream to the local flow
function evaluate_characteristics!(system, v, u, v_ode, u_ode, semi, t)
    (; volume, cache, boundary_zone) = system
    (; characteristics, previous_characteristics) = cache

    for particle in eachparticle(system)
        previous_characteristics[1, particle] = characteristics[1, particle]
        previous_characteristics[2, particle] = characteristics[2, particle]
        previous_characteristics[3, particle] = characteristics[3, particle]
    end

    set_zero!(characteristics)
    set_zero!(volume)

    # Evaluate the characteristic variables with the fluid system
    evaluate_characteristics!(system, system.fluid_system, v, u, v_ode, u_ode, semi, t)

    # Only some of the in-/outlet particles are in the influence of the fluid particles.
    # Thus, we compute the characteristics for the particles that are outside the influence
    # of fluid particles by using the average of the values of the previous time step.
    # See eq. 27 in Negi (2020) https://doi.org/10.1016/j.cma.2020.113119
    @threaded system for particle in each_moving_particle(system)

        # Particle is outside of the influence of fluid particles
        if isapprox(volume[particle], 0.0)

            # Using the average of the values at the previous time step for particles which
            # are outside of the influence of fluid particles.
            avg_J1 = 0.0
            avg_J2 = 0.0
            avg_J3 = 0.0
            counter = 0

            for neighbor in each_moving_particle(system)
                # Make sure that only neighbors in the influence of
                # the fluid particles are used.
                if volume[neighbor] > sqrt(eps())
                    avg_J1 += previous_characteristics[1, neighbor]
                    avg_J2 += previous_characteristics[2, neighbor]
                    avg_J3 += previous_characteristics[3, neighbor]
                    counter += 1
                end
            end

            characteristics[1, particle] = avg_J1 / counter
            characteristics[2, particle] = avg_J2 / counter
            characteristics[3, particle] = avg_J3 / counter
        else
            characteristics[1, particle] /= volume[particle]
            characteristics[2, particle] /= volume[particle]
            characteristics[3, particle] /= volume[particle]
        end
        prescribe_conditions!(characteristics, particle, boundary_zone)
    end

    return system
end

function evaluate_characteristics!(system, neighbor_system::FluidSystem,
                                   v, u, v_ode, u_ode, semi, t)
    (; volume, sound_speed, cache, flow_direction,
    reference_velocity, reference_pressure, reference_density) = system
    (; characteristics) = cache

    v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)
    u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

    nhs = get_neighborhood_search(system, neighbor_system, semi)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Loop over all fluid neighbors within the kernel cutoff
    for_particle_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                          nhs) do particle, neighbor, pos_diff, distance
        neighbor_position = current_coords(u_neighbor_system, neighbor_system, neighbor)

        # Determine current and prescribed quantities
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        rho_ref = reference_density(neighbor_position, t)

        p_b = particle_pressure(v_neighbor_system, neighbor_system, neighbor)
        p_ref = reference_pressure(neighbor_position, t)

        v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        v_neighbor_ref = reference_velocity(neighbor_position, t)

        # Determine characteristic variables
        density_term = -sound_speed^2 * (rho_b - rho_ref)
        pressure_term = p_b - p_ref
        velocity_term = rho_b * sound_speed * (dot(v_b - v_neighbor_ref, flow_direction))

        kernel_ = smoothing_kernel(neighbor_system, distance)

        characteristics[1, particle] += (density_term + pressure_term) * kernel_
        characteristics[2, particle] += (velocity_term + pressure_term) * kernel_
        characteristics[3, particle] += (-velocity_term + pressure_term) * kernel_

        volume[particle] += kernel_
    end

    return system
end

@inline function prescribe_conditions!(characteristics, particle, ::OutFlow)
    # J3 is prescribed (i.e. determined from the exterior of the domain).
    # J1 and J2 is transimtted from the domain interior.
    characteristics[3, particle] = zero(eltype(characteristics))

    return characteristics
end

@inline function prescribe_conditions!(characteristics, particle, ::InFlow)
    # Allow only J3 to propagate upstream to the boundary
    characteristics[1, particle] = zero(eltype(characteristics))
    characteristics[2, particle] = zero(eltype(characteristics))

    return characteristics
end
