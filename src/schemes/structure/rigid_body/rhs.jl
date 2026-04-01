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
                   particle_system::RigidBodySystem,
                   neighbor_system::WallBoundarySystem, semi)
    contact_model = particle_system.contact_model

    # Rigid-wall collision is assembled in two phases for each rigid-wall system pair:
    #
    # 1. Traverse all wall neighbors of every rigid particle and cluster penetrating wall
    #    samples into a small number of "contact manifolds". A manifold is a discrete
    #    approximation of one locally smooth contact patch. A flat wall under one rigid
    #    particle usually produces one manifold, while corners or edges can produce several.
    #
    # 2. Reduce each manifold to one averaged contact state and evaluate exactly one
    #    spring-dashpot force from that state. This avoids reacting to every wall sample
    #    individually, which would be both noisy and resolution-dependent.
    #
    # The manifold cache is pair-local scratch storage. It is rebuilt from zero for every
    # rigid-wall interaction pair and only the resulting force contributions survive in
    # `force_per_particle`, which is later reduced to a rigid-body resultant.
    set_zero!(particle_system.cache.contact_manifold_count)
    set_zero!(particle_system.cache.contact_manifold_weight_sum)
    set_zero!(particle_system.cache.contact_manifold_penetration_sum)
    set_zero!(particle_system.cache.contact_manifold_normal_sum)
    set_zero!(particle_system.cache.contact_manifold_wall_velocity_sum)

    NDIMS = ndims(particle_system)
    ELTYPE = eltype(particle_system)
    contact_count = 0
    max_contact_penetration = zero(ELTYPE)

    # First gather all penetrating wall neighbors into a small set of contact manifolds per
    # rigid particle. This avoids applying one noisy contact force per wall particle.
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi; points=each_integrated_particle(particle_system),
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       pos_diff, distance
        # Building manifolds mutates shared cache entries for the current rigid particle and can
        # merge a new wall sample into an existing manifold. Keep this pass serial so manifold
        # assignment stays deterministic and free of synchronization overhead.
        accumulate_wall_contact_pair!(particle_system, v_neighbor_system, neighbor_system,
                                      particle, neighbor, pos_diff, distance, contact_model)
    end

    # Apply one force contribution per manifold using the averaged normal, penetration, and
    # wall velocity stored in the cache.
    @threaded semi for particle in each_integrated_particle(particle_system)
        n_manifolds = particle_system.cache.contact_manifold_count[particle]
        if n_manifolds > 0
            v_particle = current_velocity(v_particle_system, particle_system, particle)
            contact_manifold_normal_sum = particle_system.cache.contact_manifold_normal_sum
            contact_manifold_wall_velocity_sum = particle_system.cache.contact_manifold_wall_velocity_sum

            for manifold_index in 1:n_manifolds
                weight_sum = particle_system.cache.contact_manifold_weight_sum[manifold_index,
                                                                               particle]
                if weight_sum > eps(ELTYPE)
                    # Recover the weighted-average manifold normal and wall velocity from the
                    # accumulated sums. The normal is re-normalized because averaging neighboring
                    # wall samples generally changes its length.
                    normal = extract_svector(contact_manifold_normal_sum, Val(NDIMS),
                                             manifold_index, particle) / weight_sum
                    normal_norm = norm(normal)
                    if normal_norm > eps(ELTYPE)
                        normal /= normal_norm

                        v_boundary = extract_svector(contact_manifold_wall_velocity_sum,
                                                     Val(NDIMS), manifold_index,
                                                     particle) / weight_sum
                        # Penetration is averaged with the same local SPH-like weight that was
                        # used when building the manifold, so the effective contact state
                        # represents one weighted contact patch rather than one arbitrary sample.
                        penetration_effective = particle_system.cache.contact_manifold_penetration_sum[manifold_index,
                                                                                                       particle] /
                                                weight_sum

                        relative_velocity = v_particle - v_boundary
                        normal_velocity = dot(relative_velocity, normal)

                        elastic_force = contact_model.normal_stiffness *
                                        penetration_effective
                        damping_force = -contact_model.normal_damping * normal_velocity
                        normal_force_magnitude = max(elastic_force + damping_force,
                                                     zero(ELTYPE))

                        if normal_force_magnitude > 0
                            interaction_force = normal_force_magnitude * normal

                            for dim in eachindex(interaction_force)
                                particle_system.force_per_particle[dim,
                                                                   particle] += interaction_force[dim]
                            end

                            contact_count += 1
                            max_contact_penetration = max(max_contact_penetration,
                                                          penetration_effective)
                        end
                    end
                end
            end
        end
    end

    particle_system.cache.contact_count[] += contact_count
    particle_system.cache.max_contact_penetration[] = max(particle_system.cache.max_contact_penetration[],
                                                          max_contact_penetration)

    return dv
end

# Process one rigid-particle / wall-particle pair for the manifold-building phase.
#
# This helper filters out non-contacts, computes the local contact state of a single wall
# sample, determines which manifold that sample belongs to, and adds its weighted
# contribution to the manifold sums. The weight is chosen as
# `wall_volume * kernel(distance)`, which keeps the manifold average close to the wall
# discretization used elsewhere in the SPH code.
@inline function accumulate_wall_contact_pair!(particle_system::RigidBodySystem,
                                               v_neighbor_system,
                                               neighbor_system::WallBoundarySystem,
                                               particle, neighbor, pos_diff, distance,
                                               contact_model::RigidContactModel)
    ELTYPE = eltype(particle_system)
    distance <= eps(ELTYPE) && return particle_system

    penetration = contact_model.contact_distance - distance
    penetration <= 0 && return particle_system

    normal = pos_diff / distance
    wall_velocity = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    # Use the wall particle volume times kernel value as a local averaging weight for the
    # contact manifold. This keeps the manifold reduction close to the SPH discretization.
    density = convert(ELTYPE, neighbor_system.initial_condition.density[neighbor])
    density <= eps(ELTYPE) && return particle_system

    volume = convert(ELTYPE, neighbor_system.initial_condition.mass[neighbor]) / density
    kernel_weight = convert(ELTYPE, smoothing_kernel(neighbor_system, distance, neighbor))
    contact_weight = max(kernel_weight * volume, zero(ELTYPE))
    contact_weight <= eps(ELTYPE) && return particle_system

    # For adjacent wall samples on the same locally flat face, the tangential offset is about
    # one wall particle spacing. Viewed from the rigid particle, this implies a predictable
    # deviation between their contact normals. We convert that geometric tolerance to a cosine
    # threshold and merge only wall samples whose normals are at least that well aligned.
    wall_spacing = convert(ELTYPE, particle_spacing(neighbor_system, neighbor))
    normal_merge_cos = wall_spacing <= eps(ELTYPE) ? one(ELTYPE) :
                       distance / hypot(distance, wall_spacing)

    manifold_index = find_or_create_contact_manifold!(particle_system.cache, particle,
                                                      normal,
                                                      normal_merge_cos)
    accumulate_contact_manifold_sums!(particle_system.cache, particle, manifold_index,
                                      contact_weight, normal, wall_velocity, penetration)

    return particle_system
end

# Assign a wall-contact sample to an existing manifold or open a new manifold slot.
#
# The procedure is a greedy clustering step performed separately for every rigid particle:
# compare the sample normal with the current averaged normal of each existing manifold,
# accept the first manifold whose alignment exceeds the merge threshold, otherwise create a
# new manifold if capacity allows. When the preallocated manifold budget is exhausted, fall
# back to the best-aligned existing manifold so contact information is still retained in a
# bounded amount of storage.
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

# Accumulate weighted manifold statistics for one accepted wall-contact sample.
#
# The cache stores sums instead of already-averaged values so multiple wall samples can be
# merged without losing information about their relative importance. The final force pass
# later divides by `weight_sum` once to recover the effective manifold normal, wall velocity,
# and penetration for that rigid particle / manifold pair.
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

function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::RigidBodySystem,
                   neighbor_system::RigidBodySystem, semi)
    contact_model = particle_system.contact_model
    neighbor_contact_model = neighbor_system.contact_model

    # Without a contact model, rigid bodies do not interact with each other at all, so skip
    if isnothing(contact_model) || isnothing(neighbor_contact_model)
        return dv
    end

    # We don't need to model self collision
    particle_system === neighbor_system && return dv

    ELTYPE = eltype(particle_system)
    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    contact_count = 0
    max_contact_penetration = zero(ELTYPE)
    pair_normal_stiffness = (contact_model.normal_stiffness +
                             neighbor_contact_model.normal_stiffness) / 2
    pair_normal_damping = (contact_model.normal_damping +
                           neighbor_contact_model.normal_damping) / 2

    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           semi;
                           points=each_integrated_particle(particle_system)) do particle,
                                                                                neighbor,
                                                                                pos_diff,
                                                                                distance
        # This kernel accumulates contact force only for `particle_system`. The reverse
        # contribution is assembled by the separate `(neighbor_system, particle_system)`
        # interaction pass in `system_interaction!`, which lets this neighbor traversal use
        # the regular parallel backend.
        distance <= eps(ELTYPE) && return dv

        penetration = max(contact_model.contact_distance,
                          neighbor_contact_model.contact_distance) - distance
        penetration <= 0 && return dv

        normal = pos_diff / distance
        particle_velocity = current_velocity(v_particle_system, particle_system, particle)
        neighbor_velocity = current_velocity(v_neighbor_system, neighbor_system, neighbor)
        relative_velocity = particle_velocity - neighbor_velocity
        normal_velocity = dot(relative_velocity, normal)

        elastic_force = pair_normal_stiffness * penetration
        damping_force = -pair_normal_damping * normal_velocity
        normal_force_magnitude = max(elastic_force + damping_force, zero(ELTYPE))
        normal_force_magnitude <= 0 && return dv

        interaction_force = normal_force_magnitude * normal

        for dim in 1:ndims(particle_system)
            particle_system.force_per_particle[dim, particle] += interaction_force[dim]
        end

        contact_count += 1
        max_contact_penetration = max(max_contact_penetration, penetration)
    end

    particle_system.cache.contact_count[] += contact_count
    particle_system.cache.max_contact_penetration[] = max(particle_system.cache.max_contact_penetration[],
                                                          max_contact_penetration)

    return dv
end

# Rigid bodies passing through open boundaries are currently not modeled.
function interact!(dv, v_particle_system, u_particle_system,
                   v_neighbor_system, u_neighbor_system,
                   particle_system::OpenBoundarySystem{<:BoundaryModelDynamicalPressureZhang},
                   neighbor_system::RigidBodySystem, semi)
    return dv
end

# Other rigid interactions, such as rigid-flexible contact or rigid-boundary coupling beyond
# the dedicated methods above, are currently not modeled.
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
                   particle_system::RigidBodySystem{<:Any, Nothing},
                   neighbor_system::WallBoundarySystem, semi)
    return dv
end
