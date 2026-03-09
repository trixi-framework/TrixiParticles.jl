# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidSPHSystem,
                   neighbor_system::Union{AbstractFluidSystem, OpenBoundarySystem}, semi)
    sound_speed = system_sound_speed(neighbor_system)
    surface_tension = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Accumulate pairwise fluid forces per rigid particle first, then reduce them to a
    # single resultant force/torque for the rigid-body update below.
    force_per_particle = particle_system.cache.force_per_particle
    set_zero!(force_per_particle)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(particle_system, neighbor_system,
                           system_coords, neighbor_coords, semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance

        # Only consider particles with a distance > 0.
        # See `src/general/smoothing_kernels.jl` for more details.
        distance^2 < eps(initial_smoothing_length(particle_system)^2) && return

        # Apply the same force to the structure particle that the fluid particle
        # experiences due to the structure particle.
        # In fluid-structure interaction, use the "hydrodynamic mass" of the structure
        # particles corresponding to the rest density of the fluid.
        m_a = hydrodynamic_mass(particle_system, particle)
        m_b = hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = current_density(v_particle_system, particle_system, particle)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)

        # Use the fluid kernel in order to get the same force as in
        # fluid-structure interaction.
        grad_kernel = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

        # In fluid-structure interaction, use the "hydrodynamic pressure" of the
        # structure particles corresponding to the chosen boundary model.
        p_a = current_pressure(v_particle_system, particle_system, particle)
        p_b = current_pressure(v_neighbor_system, neighbor_system, neighbor)

        # Particle and neighbor are switched in the following two calls.
        # This yields the opposite force of the fluid-structure interaction,
        # because `pos_diff` is flipped.
        dv_boundary = pressure_acceleration(neighbor_system, particle_system,
                                            neighbor, particle,
                                            m_b, m_a, p_b, p_a, rho_b, rho_a,
                                            pos_diff, distance, grad_kernel,
                                            system_correction(neighbor_system))

        dv_viscosity_ = dv_viscosity(neighbor_system, particle_system,
                                     v_neighbor_system, v_particle_system,
                                     neighbor, particle, pos_diff, distance,
                                     sound_speed, m_b, m_a, rho_b, rho_a,
                                     grad_kernel)

        dv_adhesion = adhesion_force(surface_tension, neighbor_system, particle_system,
                                     neighbor, particle, pos_diff, distance)

        dv_particle = dv_boundary + dv_viscosity_ + dv_adhesion

        for i in 1:ndims(particle_system)
            # `pressure_acceleration`/`dv_viscosity` return acceleration-like pair contributions.
            # Multiply by the interacting fluid mass to recover the force on this rigid particle.
            force_per_particle[i, particle] += dv_particle[i] * m_b
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    apply_resultant_force_and_torque!(dv, particle_system, semi)

    return dv
end

# Reduce the accumulated fluid forces to rigid-body resultants and apply the corresponding
# translational and rotational acceleration to every rigid particle.
function apply_resultant_force_and_torque!(dv, particle_system::RigidSPHSystem, semi)
    (; cache) = particle_system
    total_mass = cache.total_mass

    # Guard against degenerate systems and clear the cached rigid-body quantities as well.
    if total_mass <= eps(eltype(particle_system))
        cache.resultant_force[] = zero(cache.resultant_force[])
        cache.resultant_torque[] = zero(cache.resultant_torque[])
        cache.angular_acceleration_force[] = zero(cache.angular_acceleration_force[])
        return dv
    end

    # Reduce all pairwise forces to one net force and one net torque around the center of mass.
    total_force,
    total_torque = resultant_force_and_torque(particle_system,
                                              cache.force_per_particle,
                                              cache.relative_coordinates)

    # Convert the rigid-body resultants into translational and angular accelerations.
    translational_acceleration = total_force / total_mass
    angular_acceleration_force = cache.inverse_inertia[] * total_torque
    cache.resultant_force[] = total_force
    cache.resultant_torque[] = total_torque
    cache.angular_acceleration_force[] = angular_acceleration_force

    @threaded semi for particle in each_integrated_particle(particle_system)
        relative_position = extract_svector(cache.relative_coordinates, particle_system,
                                            particle)
        # For rigid bodies, the instantaneous acceleration of a material point is
        # `a_com + alpha x r` in this force-driven part of the RHS.
        rotational_acceleration = cross_product(angular_acceleration_force,
                                                relative_position)

        for i in 1:ndims(particle_system)
            dv[i, particle] += translational_acceleration[i] + rotational_acceleration[i]
        end
    end

    return dv
end

# Sum pairwise particle forces into a single net force and torque about the current
# center of mass of the rigid body.
function resultant_force_and_torque(particle_system::RigidSPHSystem{<:Any, <:Any, NDIMS},
                                    force_per_particle,
                                    relative_coordinates) where {NDIMS}
    total_force = zero(SVector{NDIMS, eltype(particle_system)})
    total_torque = zero(particle_system.cache.resultant_torque[])

    for particle in each_integrated_particle(particle_system)
        particle_force = extract_svector(force_per_particle, particle_system, particle)
        relative_position = extract_svector(relative_coordinates, particle_system, particle)
        total_force += particle_force

        # Torque is taken about the current center of mass, using the particle's current
        # relative position inside the rigid body.
        total_torque += cross_product(relative_position, particle_force)
    end

    return total_force, total_torque
end

# Default rigid boundary models keep density fixed, so structure-fluid coupling does not
# contribute to a density RHS entry.
@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem,
                                      neighbor_system::Union{AbstractFluidSystem,
                                                             OpenBoundarySystem},
                                      grad_kernel)
    # Most rigid boundary models keep their density fixed, so no continuity update is needed.
    return dv
end

# Dummy-particle rigid boundaries with `ContinuityDensity` reuse the fluid-compatible
# density update for the rigid particle.
@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidSPHSystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                      neighbor_system::Union{AbstractFluidSystem,
                                                             OpenBoundarySystem},
                                      grad_kernel)
    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    # Dummy rigid particles reuse the fluid-compatible density update of the neighbor system.
    continuity_equation!(dv, density_calculator(neighbor_system), m_b, rho_a, rho_b, v_diff,
                         grad_kernel, particle)
end

function reset_contact_manifold_cache!(cache)
    set_zero!(cache.contact_manifold_count)
    set_zero!(cache.contact_manifold_weight_sum)
    set_zero!(cache.contact_manifold_penetration_sum)
    set_zero!(cache.contact_manifold_normal_sum)
    set_zero!(cache.contact_manifold_wall_velocity_sum)
    set_zero!(cache.contact_manifold_tangential_displacement_sum)

    return cache
end

@inline function wall_contact_pair_weight(neighbor_system::WallBoundarySystem,
                                          distance, neighbor, ELTYPE)
    density = convert(ELTYPE, neighbor_system.initial_condition.density[neighbor])
    density <= eps(ELTYPE) && return zero(ELTYPE)

    volume = convert(ELTYPE, neighbor_system.initial_condition.mass[neighbor]) / density
    kernel_weight = convert(ELTYPE, smoothing_kernel(neighbor_system, distance, neighbor))

    return max(kernel_weight * volume, zero(ELTYPE))
end

function find_or_add_contact_manifold!(cache, particle, normal, normal_merge_cos, ELTYPE)
    manifold_count = cache.contact_manifold_count[particle]
    normal_sum = cache.contact_manifold_normal_sum

    best_index = 1
    best_dot = -one(ELTYPE)

    for manifold in 1:manifold_count
        normal_norm_squared = zero(ELTYPE)
        dot_value = zero(ELTYPE)
        for dim in eachindex(normal)
            normal_value = normal_sum[dim, manifold, particle]
            normal_norm_squared += normal_value^2
            dot_value += normal_value * normal[dim]
        end

        if normal_norm_squared > eps(ELTYPE)
            dot_value /= sqrt(normal_norm_squared)
        else
            dot_value = one(ELTYPE)
        end

        if dot_value >= normal_merge_cos
            return manifold
        end

        if dot_value > best_dot
            best_dot = dot_value
            best_index = manifold
        end
    end

    max_manifolds = size(cache.contact_manifold_weight_sum, 1)
    if manifold_count < max_manifolds
        manifold_count += 1
        cache.contact_manifold_count[particle] = manifold_count
        return manifold_count
    end

    return best_index
end

function accumulate_contact_manifold!(cache, particle, manifold, contact_weight, normal,
                                      wall_velocity, penetration_effective,
                                      tangential_displacement)
    cache.contact_manifold_weight_sum[manifold, particle] += contact_weight
    cache.contact_manifold_penetration_sum[manifold, particle] +=
        contact_weight * penetration_effective

    for dim in eachindex(normal)
        cache.contact_manifold_normal_sum[dim, manifold, particle] +=
            contact_weight * normal[dim]
        cache.contact_manifold_wall_velocity_sum[dim, manifold, particle] +=
            contact_weight * wall_velocity[dim]
        cache.contact_manifold_tangential_displacement_sum[dim, manifold, particle] +=
            contact_weight * tangential_displacement[dim]
    end

    return cache
end

@inline function weighted_manifold_vector(cache_array, manifold, particle, weight_sum,
                                          ::Val{NDIMS}, ELTYPE) where {NDIMS}
    return SVector{NDIMS, ELTYPE}(ntuple(@inline(dim->cache_array[dim, manifold, particle] /
                                                    weight_sum),
                                         Val(NDIMS)))
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidSPHSystem{<:Any, <:Any, NDIMS},
                   neighbor_system::WallBoundarySystem, semi) where {NDIMS}
    contact_model = particle_system.boundary_contact_model
    isnothing(contact_model) && return dv

    force_per_particle = particle_system.cache.force_per_particle
    set_zero!(force_per_particle)
    reset_contact_manifold_cache!(particle_system.cache)
    particle_system.cache.boundary_contact_count[] = 0
    particle_system.cache.max_boundary_penetration[] = zero(eltype(particle_system))

    neighbor_system_index = system_indices(neighbor_system, semi)
    ELTYPE = eltype(particle_system)
    zero_tangential = zero(SVector{NDIMS, ELTYPE})
    contact_map = particle_system.cache.contact_tangential_displacement
    normal_merge_cos = convert(ELTYPE, 0.995)

    # Gather grid-independent contact manifolds from all wall-neighbor contacts.
    update_contact_manifold_cache!(particle_system, neighbor_system,
                                   u_particle_system,
                                   v_neighbor_system, u_neighbor_system,
                                   semi,
                                   contact_model,
                                   normal_merge_cos, zero_tangential,
                                   ELTYPE)

    # Apply one interaction force per manifold to make the model independent of
    # boundary-neighbor multiplicity.
    for particle in each_integrated_particle(particle_system)
        n_manifolds = particle_system.cache.contact_manifold_count[particle]
        n_manifolds == 0 && continue

        v_particle = current_velocity(v_particle_system, particle_system, particle)

        for manifold in 1:n_manifolds
            weight_sum = particle_system.cache.contact_manifold_weight_sum[manifold, particle]
            weight_sum <= eps(ELTYPE) && continue

            normal = weighted_manifold_vector(particle_system.cache.contact_manifold_normal_sum,
                                              manifold, particle, weight_sum, Val(NDIMS),
                                              ELTYPE)
            normal_norm = norm(normal)
            normal_norm <= eps(ELTYPE) && continue
            normal /= normal_norm

            v_boundary = weighted_manifold_vector(particle_system.cache.contact_manifold_wall_velocity_sum,
                                                  manifold, particle, weight_sum,
                                                  Val(NDIMS), ELTYPE)
            penetration_effective = particle_system.cache.contact_manifold_penetration_sum[manifold,
                                                                                            particle] /
                                    weight_sum

            relative_velocity = v_particle - v_boundary
            normal_velocity = dot(relative_velocity, normal)
            tangential_velocity = relative_velocity - normal_velocity * normal

            normal_force_magnitude,
            normal_force_friction_reference = normal_contact_force_components(contact_model,
                                                                              penetration_effective,
                                                                              normal_velocity,
                                                                              ELTYPE)
            normal_force_magnitude <= 0 && continue

            contact_key = (neighbor_system_index, particle, manifold)
            tangential_displacement = isnothing(contact_map) ?
                                      zero_tangential :
                                      get(contact_map, contact_key, zero_tangential)
            tangential_force = tangential_contact_force(contact_model,
                                                        tangential_displacement,
                                                        tangential_velocity,
                                                        normal_force_friction_reference,
                                                        ELTYPE)
            interaction_force = normal_force_magnitude * normal + tangential_force

            for dim in 1:NDIMS
                force_per_particle[dim, particle] += interaction_force[dim]
            end

            particle_system.cache.boundary_contact_count[] += 1
            particle_system.cache.max_boundary_penetration[] = max(particle_system.cache.max_boundary_penetration[],
                                                                   penetration_effective)
        end
    end

    if contact_model.torque_free
        remove_resultant_contact_torque!(force_per_particle, particle_system)
    end

    apply_resultant_force_and_torque!(dv, particle_system, semi)

    return dv
end

function remove_resultant_contact_torque!(force_per_particle,
                                          particle_system::RigidSPHSystem{<:Any, <:Any, 2})
    relative_coordinates = particle_system.cache.relative_coordinates
    torque = zero(eltype(particle_system))
    inertia_measure = zero(eltype(particle_system))

    for particle in each_integrated_particle(particle_system)
        relative_position = extract_svector(relative_coordinates, particle_system, particle)
        particle_force = extract_svector(force_per_particle, particle_system, particle)
        torque += cross(relative_position, particle_force)
        inertia_measure += dot(relative_position, relative_position)
    end

    if abs(torque) <= eps(eltype(particle_system)) ||
       inertia_measure <= eps(eltype(particle_system))
        return force_per_particle
    end

    alpha = -torque / inertia_measure
    for particle in each_integrated_particle(particle_system)
        relative_position = extract_svector(relative_coordinates, particle_system, particle)
        correction_force = alpha * SVector(-relative_position[2], relative_position[1])

        @inbounds begin
            force_per_particle[1, particle] += correction_force[1]
            force_per_particle[2, particle] += correction_force[2]
        end
    end

    return force_per_particle
end

function remove_resultant_contact_torque!(force_per_particle,
                                          particle_system::RigidSPHSystem{<:Any, <:Any, 3})
    return force_per_particle
end

@inline function normal_contact_force(contact_model::RigidBoundaryContactModel,
                                      penetration, normal_velocity, ELTYPE)
    normal_force, _ = normal_contact_force_components(contact_model, penetration,
                                                      normal_velocity, ELTYPE)

    return normal_force
end

@inline function normal_friction_reference_force(contact_model::RigidBoundaryContactModel,
                                                 penetration, normal_velocity, ELTYPE)
    _, friction_reference_force = normal_contact_force_components(contact_model, penetration,
                                                                  normal_velocity, ELTYPE)

    return friction_reference_force
end

@inline function normal_contact_force_components(contact_model::RigidBoundaryContactModel,
                                                 penetration, normal_velocity, ELTYPE)
    elastic_force = contact_model.normal_stiffness * penetration
    # Kelvin-Voigt normal damping: oppose relative normal motion in both
    # compression and decompression; clamp to avoid artificial attraction.
    damping_force = -contact_model.normal_damping * normal_velocity
    normal_force = max(elastic_force + damping_force, zero(ELTYPE))
    # Use the same (total) normal load for tangential-history limiting and
    # tangential-force evaluation for physically consistent Coulomb limits.
    friction_reference_force = normal_force

    return normal_force, friction_reference_force
end

function tangential_contact_force(contact_model::RigidBoundaryContactModel,
                                  tangential_displacement,
                                  tangential_velocity,
                                  normal_force_friction_reference, ELTYPE)
    force_trial = -contact_model.tangential_stiffness * tangential_displacement -
                  contact_model.tangential_damping * tangential_velocity

    trial_norm = norm(force_trial)
    tangential_speed = norm(tangential_velocity)
    static_limit = contact_model.static_friction_coefficient *
                   normal_force_friction_reference

    # Stay in the static branch as long as the tangential spring-damper force is admissible.
    # This avoids high-frequency stick/slip toggling when the tangential speed is close to zero.
    if trial_norm <= static_limit
        return force_trial
    end

    kinetic_limit = contact_model.kinetic_friction_coefficient *
                    normal_force_friction_reference
    kinetic_limit <= eps(ELTYPE) && return zero(force_trial)

    # Smooth kinetic friction near zero slip velocity to remove force discontinuities
    # that can force adaptive ODE solvers into extremely small time steps.
    regularization_velocity = max(contact_model.stick_velocity_tolerance,
                                  sqrt(eps(ELTYPE)))

    if tangential_speed > eps(ELTYPE)
        speed_factor = tanh(tangential_speed / regularization_velocity)
        return -kinetic_limit * speed_factor * tangential_velocity / tangential_speed
    end

    if trial_norm > eps(ELTYPE)
        return -kinetic_limit * force_trial / trial_norm
    end

    return zero(force_trial)
end

# Structure-structure and structure-boundary interactions are currently not modeled.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidSPHSystem,
                   neighbor_system::Union{AbstractStructureSystem,
                                          AbstractBoundarySystem}, semi)
    return dv
end
