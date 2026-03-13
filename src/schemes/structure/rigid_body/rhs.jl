function finalize_interaction!(particle_system::RigidBodySystem,
                               dv, v, u, dv_ode, v_ode, u_ode, semi)
    apply_kinematic_acceleration!(dv, particle_system, semi)
    apply_resultant_force_and_torque!(dv, particle_system, semi)

    return particle_system
end

# Rigid-body kinematics depend only on the current rigid state, so apply them once
# after all pairwise interactions instead of revisiting them for every rigid-rigid pair.
function apply_kinematic_acceleration!(dv, particle_system::RigidBodySystem, semi)
    @threaded semi for particle in each_integrated_particle(particle_system)
        relative_position = @inbounds extract_svector(particle_system.relative_coordinates,
                                                      particle_system, particle)
        rotational_acceleration = rigid_kinematic_acceleration(particle_system,
                                                               relative_position)

        for i in 1:ndims(particle_system)
            @inbounds dv[i, particle] += rotational_acceleration[i]
        end
    end

    return dv
end

# Structure-fluid interaction
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidBodySystem,
                   neighbor_system::AbstractFluidSystem, semi)
    sound_speed = system_sound_speed(neighbor_system)
    surface_tension = surface_tension_model(neighbor_system)

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Accumulate pairwise fluid forces per rigid particle first, then reduce them to a
    # single resultant force/torque after all rigid-fluid interactions have been processed.
    (; force_per_particle) = particle_system

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
        m_a = @inbounds hydrodynamic_mass(particle_system, particle)
        m_b = @inbounds hydrodynamic_mass(neighbor_system, neighbor)

        rho_a = @inbounds current_density(v_particle_system, particle_system, particle)
        rho_b = @inbounds current_density(v_neighbor_system, neighbor_system, neighbor)

        # Use the fluid kernel in order to get the same force as in
        # fluid-structure interaction.
        grad_kernel = smoothing_kernel_grad(neighbor_system, pos_diff, distance, neighbor)

        # In fluid-structure interaction, use the "hydrodynamic pressure" of the
        # structure particles corresponding to the chosen boundary model.
        p_a = @inbounds current_pressure(v_particle_system, particle_system, particle)
        p_b = @inbounds current_pressure(v_neighbor_system, neighbor_system, neighbor)

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
            @inbounds force_per_particle[i, particle] += dv_particle[i] * m_b
        end

        continuity_equation!(dv, v_particle_system, v_neighbor_system,
                             particle, neighbor, pos_diff, distance,
                             m_b, rho_a, rho_b,
                             particle_system, neighbor_system, grad_kernel)
    end

    return dv
end

# Reduce the accumulated fluid forces to rigid-body resultants and apply the corresponding
# translational and rotational acceleration to every rigid particle.
function apply_resultant_force_and_torque!(dv, particle_system::RigidBodySystem, semi)
    total_mass = particle_system.total_mass

    # Reduce all pairwise forces to one net force and one net torque around the center of mass.
    total_force = zero(SVector{ndims(particle_system), eltype(particle_system)})
    total_torque = zero(particle_system.resultant_torque[])

    for particle in each_integrated_particle(particle_system)
        particle_force = extract_svector(particle_system.force_per_particle,
                                         particle_system,
                                         particle)
        relative_position = extract_svector(particle_system.relative_coordinates,
                                            particle_system,
                                            particle)
        total_force += particle_force

        # Torque is taken about the current center of mass, using the particle's current
        # relative position inside the rigid body.
        total_torque += cross_product(relative_position, particle_force)
    end

    translational_acceleration = total_force / total_mass
    angular_acceleration_force = particle_system.inverse_inertia[] * total_torque
    particle_system.resultant_force[] = total_force
    particle_system.resultant_torque[] = total_torque
    particle_system.angular_acceleration_force[] = angular_acceleration_force

    @threaded semi for particle in each_integrated_particle(particle_system)
        relative_position = @inbounds extract_svector(particle_system.relative_coordinates,
                                                      particle_system, particle)
        # For rigid bodies, the instantaneous acceleration of a material point is
        # `a_com + alpha x r` in this force-driven part of the RHS.
        rotational_acceleration_force = cross_product(angular_acceleration_force,
                                                      relative_position)

        for i in eachindex(translational_acceleration)
            @inbounds dv[i,
                         particle] += translational_acceleration[i] +
                                      rotational_acceleration_force[i]
        end
    end

    return dv
end

# Default rigid boundary models keep density fixed, so structure-fluid coupling does not
# contribute to a density RHS entry.
@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidBodySystem,
                                      neighbor_system, grad_kernel)
    # Most rigid boundary models keep their density fixed, so no continuity update is needed.
    return dv
end

# Dummy-particle rigid boundaries with `ContinuityDensity` reuse the fluid-compatible
# density update for the rigid particle.
@inline function continuity_equation!(dv, v_particle_system, v_neighbor_system,
                                      particle, neighbor, pos_diff, distance,
                                      m_b, rho_a, rho_b,
                                      particle_system::RigidBodySystem{<:BoundaryModelDummyParticles{ContinuityDensity}},
                                      neighbor_system, grad_kernel)
    v_diff = current_velocity(v_particle_system, particle_system, particle) -
             current_velocity(v_neighbor_system, neighbor_system, neighbor)

    # Dummy rigid particles reuse the fluid-compatible density update of the neighbor system.
    continuity_equation!(dv, density_calculator(neighbor_system), m_b, rho_a, rho_b, v_diff,
                         grad_kernel, particle)
end

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidBodySystem{<:Any, CM, NDIMS},
                   neighbor_system::WallBoundarySystem, semi) where {CM, NDIMS}
    contact_model = particle_system.contact_model

    # Here a "contact manifold" means one cluster of wall neighbors for one rigid particle
    # that appears to belong to the same local wall patch. A flat wall contact usually
    # produces one manifold, while a corner/edge contact may produce several. We reduce each
    # manifold to one averaged contact state before computing forces.
    #
    # Rebuild the manifold cache for this rigid-wall system pair before reducing it to forces.
    set_zero!(particle_system.cache.contact_manifold_count)
    set_zero!(particle_system.cache.contact_manifold_weight_sum)
    set_zero!(particle_system.cache.contact_manifold_penetration_sum)
    set_zero!(particle_system.cache.contact_manifold_normal_sum)
    set_zero!(particle_system.cache.contact_manifold_wall_velocity_sum)

    ELTYPE = eltype(particle_system)

    # First gather all penetrating wall neighbors into a small set of contact manifolds per
    # rigid particle. This avoids applying one noisy contact force per wall particle.
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi; points=each_integrated_particle(particle_system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        # The manifold cache is shared across all wall neighbors of this system pair, so build
        # it deterministically with a serial traversal.
        accumulate_wall_contact_pair!(particle_system, v_neighbor_system, neighbor_system,
                                      particle, neighbor, pos_diff, distance, contact_model)
    end

    # Apply one force contribution per manifold using the averaged normal, penetration, and
    # wall velocity stored in the cache.
    for particle in each_integrated_particle(particle_system)
        n_manifolds = particle_system.cache.contact_manifold_count[particle]
        n_manifolds == 0 && continue

        v_particle = current_velocity(v_particle_system, particle_system, particle)

        for manifold_index in 1:n_manifolds
            weight_sum = particle_system.cache.contact_manifold_weight_sum[manifold_index,
                                                                           particle]
            weight_sum <= eps(ELTYPE) && continue

            normal = SVector{NDIMS, ELTYPE}(ntuple(@inline(dim->particle_system.cache.contact_manifold_normal_sum[dim,
                                                                                                                  manifold_index,
                                                                                                                  particle] /
                                                                weight_sum),
                                                   Val(NDIMS)))
            normal_norm = norm(normal)
            normal_norm <= eps(ELTYPE) && continue
            normal /= normal_norm

            v_boundary = SVector{NDIMS, ELTYPE}(ntuple(@inline(dim->particle_system.cache.contact_manifold_wall_velocity_sum[dim,
                                                                                                                             manifold_index,
                                                                                                                             particle] /
                                                                    weight_sum),
                                                       Val(NDIMS)))
            penetration_effective = particle_system.cache.contact_manifold_penetration_sum[manifold_index,
                                                                                           particle] /
                                    weight_sum

            relative_velocity = v_particle - v_boundary
            normal_velocity = dot(relative_velocity, normal)

            # Only the normal spring-dashpot part is kept in the basic collision.
            elastic_force = contact_model.normal_stiffness * penetration_effective
            damping_force = -contact_model.normal_damping * normal_velocity
            normal_force_magnitude = max(elastic_force + damping_force, zero(ELTYPE))
            normal_force_magnitude <= 0 && continue

            interaction_force = normal_force_magnitude * normal

            for dim in 1:NDIMS
                particle_system.force_per_particle[dim, particle] += interaction_force[dim]
            end
        end
    end

    return dv
end
@inline function accumulate_wall_contact_pair!(particle_system::RigidBodySystem,
                                               v_neighbor_system,
                                               neighbor_system::WallBoundarySystem,
                                               particle, neighbor, pos_diff, distance,
                                               contact_model::RigidContactModel)
    ELTYPE = eltype(particle_system)
    distance <= eps(ELTYPE) && return nothing

    penetration = contact_model.contact_distance - distance
    penetration <= 0 && return nothing

    normal = pos_diff / distance
    wall_velocity = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    # Use the wall particle volume times kernel value as a local averaging weight for the
    # contact manifold. This keeps the manifold reduction close to the SPH discretization.
    density = convert(ELTYPE, neighbor_system.initial_condition.density[neighbor])
    density <= eps(ELTYPE) && return nothing

    volume = convert(ELTYPE, neighbor_system.initial_condition.mass[neighbor]) / density
    kernel_weight = convert(ELTYPE, smoothing_kernel(neighbor_system, distance, neighbor))
    contact_weight = max(kernel_weight * volume, zero(ELTYPE))
    contact_weight <= eps(ELTYPE) && return nothing

    # For adjacent wall samples on the same locally flat face, the tangential offset is
    # about one wall particle spacing. Use the corresponding viewing angle from the rigid
    # particle to decide whether this contact should join an existing manifold.
    wall_spacing = convert(ELTYPE, particle_spacing(neighbor_system, neighbor))
    normal_merge_cos = wall_spacing <= eps(ELTYPE) ? one(ELTYPE) :
                       distance / hypot(distance, wall_spacing)

    manifold_index = find_or_create_contact_manifold!(particle_system.cache, particle,
                                                      normal,
                                                      normal_merge_cos)
    accumulate_contact_manifold_sums!(particle_system.cache, particle, manifold_index,
                                      contact_weight, normal, wall_velocity, penetration)

    return nothing
end
function find_or_create_contact_manifold!(cache, particle, normal,
                                          normal_merge_cos::ELTYPE) where {ELTYPE}
    manifold_count = cache.contact_manifold_count[particle]
    normal_sum = cache.contact_manifold_normal_sum

    # Try to assign this wall neighbor to an existing manifold of the same rigid particle.
    # Two contacts belong to the same manifold when their normals are as aligned as expected
    # for adjacent samples of the same local wall patch at the current contact distance. If
    # all manifold slots are used, overwrite the best-matching one instead of creating more
    # state.
    best_index = 1
    best_dot = -one(ELTYPE)

    for manifold_index in 1:manifold_count
        normal_norm_squared = zero(ELTYPE)
        dot_value = zero(ELTYPE)
        for dim in eachindex(normal)
            normal_value = normal_sum[dim, manifold_index, particle]
            normal_norm_squared += normal_value^2
            dot_value += normal_value * normal[dim]
        end

        if normal_norm_squared > eps(ELTYPE)
            dot_value /= sqrt(normal_norm_squared)
        else
            dot_value = one(ELTYPE)
        end

        if dot_value >= normal_merge_cos
            return manifold_index
        end

        if dot_value > best_dot
            best_dot = dot_value
            best_index = manifold_index
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

function accumulate_contact_manifold_sums!(cache, particle, manifold_index, contact_weight,
                                           normal, wall_velocity, penetration_effective)
    # Store weighted sums so the final interaction step can recover one averaged contact
    # state per manifold instead of reacting to every wall particle individually. The summed
    # data describes one effective contact patch: averaged normal, wall velocity, and
    # penetration for that rigid particle/manifold pair.
    cache.contact_manifold_weight_sum[manifold_index, particle] += contact_weight
    cache.contact_manifold_penetration_sum[manifold_index,
                                           particle] += contact_weight *
                                                        penetration_effective

    for dim in eachindex(normal)
        cache.contact_manifold_normal_sum[dim, manifold_index,
                                          particle] += contact_weight * normal[dim]
        cache.contact_manifold_wall_velocity_sum[dim, manifold_index,
                                                 particle] += contact_weight *
                                                              wall_velocity[dim]
    end

    return cache
end

# Collisions between rigid bodies are not yet implemented
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidBodySystem,
                   neighbor_system::RigidBodySystem, semi)
    return dv
end

# Rigid bodies passing through open boundaries are currently not modeled.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                   neighbor_system::RigidBodySystem, semi)
    return dv
end

# Structure-structure and structure-boundary/open-boundary interactions are currently not modeled.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidBodySystem,
                   neighbor_system::Union{AbstractStructureSystem,
                                          AbstractBoundarySystem,
                                          OpenBoundarySystem}, semi)
    return dv
end

# Rigid systems without an explicit contact model ignore wall neighbors.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidBodySystem{<:Any, Nothing, NDIMS},
                   neighbor_system::WallBoundarySystem, semi) where {NDIMS}
    return dv
end
