function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search, particle_system::DEMSystem,
                   neighbor_system::DEMSystem)
    damping_coefficient = particle_system.damping_coefficient

    system_coords = current_coordinates(u_particle_system, particle_system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(particle_system, neighbor_system, system_coords, neighbor_coords,
                           neighborhood_search;
                           points=each_moving_particle(particle_system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
        distance < sqrt(eps()) && return

        # Retrieve particle properties.
        m_a = particle_system.mass[particle]
        r_a = particle_system.radius[particle]
        r_b = neighbor_system.radius[neighbor]

        # Compute the overlap (penetration depth).
        overlap = r_a + r_b - distance
        if overlap > 0
            # Compute the unit normal vector (from neighbor to particle).
            normal = pos_diff / distance

            # Compute Normal Force by Dispatching on the Contact Model.
            F_normal = collision_force_normal(particle_system.contact_model,
                                              particle_system, neighbor_system,
                                              overlap, normal,
                                              v_particle_system, v_neighbor_system,
                                              particle, neighbor, damping_coefficient)

            F_tangent = collision_force_tangential(particle_system, neighbor_system,
                                                   overlap, normal, v_particle_system,
                                                   v_neighbor_system,
                                                   particle, neighbor, F_normal)

            interaction_force = F_normal + F_tangent

            # Update the particle acceleration: a = F/m.
            for i in 1:ndims(particle_system)
                dv[i, particle] += interaction_force[i] / m_a
            end

            # Apply a simple position correction to mitigate overlap.
            position_correction!(neighbor_system, u_particle_system, overlap, normal,
                                 particle)
        end
    end

    return dv
end

function interact!(dv, v_particle_system, u_particle_system, v_neighbor_system,
                   u_neighbor_system, neighborhood_search,
                   particle_system::DEMSystem, neighbor_system::BoundaryDEMSystem)
    damping_coefficient = particle_system.damping_coefficient

    # Get the current coordinates for DEM particles and the boundary.
    system_coords = current_coordinates(u_particle_system, particle_system)
    boundary_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # Phase 1: Accumulate boundary neighbor positions for each DEM particle.
    # Here, we build a dictionary mapping a DEM particle index to a vector of
    # nearby boundary positions (each assumed to be an SVector{2,Float64} or SVector{3,Float64}).
    neighbor_dict = Dict{Int, Vector{SVector{2, Float64}}}()
    foreach_point_neighbor(particle_system, neighbor_system, system_coords, boundary_coords,
                           neighborhood_search;
                           points=each_moving_particle(particle_system)) do particle,
                                                                            neighbor,
                                                                            pos_diff,
                                                                            distance
        # For each DEM–boundary pair, pos_diff = DEM particle position - boundary particle position.
        bpos = boundary_coords[:, neighbor]
        if haskey(neighbor_dict, particle)
            push!(neighbor_dict[particle], bpos)
        else
            neighbor_dict[particle] = [bpos]
        end
    end

    # Phase 2: For each DEM particle that has boundary neighbors, compute a refined contact.
    for (particle, boundary_positions) in neighbor_dict
        # Get the DEM particle's position, mass, and radius.
        particle_pos = system_coords[:, particle]
        m_a = particle_system.mass[particle]
        r_a = particle_system.radius[particle]
        # Compute a local effective radius from the neighboring boundary positions.
        effective_radius = 0.5 * r_a

        # Compute the refined contact geometry (effective contact point and normal)
        (cp_effective, refined_normal) = compute_local_contact(particle_pos,
                                                               boundary_positions,
                                                               effective_radius)
        # Correct the raw overlap using the refined contact point.
        d_corrected = norm(particle_pos - cp_effective)
        overlap_corrected = (r_a + effective_radius) - d_corrected

        if overlap_corrected > 0
            # Use the refined normal as the contact normal.
            normal_corrected = refined_normal
            # Compute the relative normal velocity (assuming the boundary is stationary,
            # the relative velocity is simply the DEM particle's velocity projected onto the normal).
            v = current_velocity(v_particle_system, particle_system, particle)
            v_rel_norm = dot(v, normal_corrected)
            m_effective = m_a  # For boundary collisions, effective mass is the particle's mass.

            # Now, call the appropriate contact model's normal collision force routine with the corrected values.
            F_normal = collision_force_normal(particle_system.contact_model,
                                              particle_system, neighbor_system,
                                              overlap_corrected, normal_corrected,
                                              v_particle_system, v_neighbor_system,
                                              particle, 1, damping_coefficient)

            # Compute the tangential force using the corrected overlap and normal.
            F_tangent = collision_force_tangential(particle_system, neighbor_system,
                                                   overlap_corrected, normal_corrected,
                                                   v_particle_system, v_neighbor_system,
                                                   particle, 1, F_normal)

            interaction_force = F_normal + F_tangent

            # Update the DEM particle acceleration: a = F/m.
            for i in 1:ndims(particle_system)
                dv[i, particle] += interaction_force[i] / m_a
            end

            # Apply a position correction to reduce overlap.
            position_correction!(neighbor_system, u_particle_system, overlap_corrected,
                                 normal_corrected, particle)
        end
    end

    return dv
end

# Tangential Force Computation (common for now)
#
# Uses a spring-dashpot model to compute the instantaneous tangential force,
# with a Coulomb friction limit.
@inline function collision_force_tangential(particle_system, neighbor_system,
                                            overlap, normal,
                                            v_particle_system, v_neighbor_system,
                                            particle, neighbor, F_normal)
    # Tangential force parameters
    friction_coefficient = 0.5       # Coulomb friction coefficient [Cundall and Strack, 1979]
    tangential_stiffness = 1e3       # Tangential spring constant
    tangential_damping = 0.001       # Damping coefficient for tangential force

    # Compute relative velocity and extract the tangential component.
    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_rel = v_a - v_b
    v_rel_tangent = v_rel - (dot(v_rel, normal) * normal)

    # Compute tangential force as a spring–dashpot response.
    F_t = -tangential_stiffness * v_rel_tangent - tangential_damping * v_rel_tangent

    # Coulomb friction: limit the tangential force to μ * |F_normal|.
    max_tangent = friction_coefficient * norm(F_normal)
    if norm(F_t) > max_tangent && norm(F_t) > 0.0
        F_t = F_t * (max_tangent / norm(F_t))
    end

    return F_t
end

@inline function position_correction!(neighbor_system::DEMSystem,
                                      u_particle_system, overlap, normal, particle)
end

# For boundaries, a simple half-overlap correction is applied.
@inline function position_correction!(neighbor_system::BoundaryDEMSystem,
                                      u_particle_system, overlap, normal, particle)
    for i in 1:ndims(neighbor_system)
        u_particle_system[i, particle] -= 0.5 * overlap * normal[i]
    end
end

"""
    compute_local_contact(particle_pos, boundary_positions, effective_radius)

Given the DEM particle’s position `particle_pos` (an AbstractVector with 2 elements)
and a vector of neighboring boundary particle positions (`boundary_positions`, each
an SVector{2, T}), along with the effective radius used for rounding the boundary,
this function computes:
  - cp_effective: the effective contact point on the boundary,
  - normal: the unit contact normal (pointing from the boundary toward the particle).

It does so by sorting the boundary positions by their polar angle relative to `particle_pos`,
selecting the two “closest” in angle, fitting a local line (tangent) through them, projecting
the DEM particle onto that line, and offsetting the projection by `effective_radius` along the normal.
"""
function compute_local_contact(particle_pos, boundary_positions, effective_radius)
    # Stack the boundary positions into a d×n matrix (each column is one boundary point).
    X = hcat(boundary_positions...)
    n = size(X, 2)
    # Compute the centroid of the boundary points.
    centroid = sum(X, dims=2) / n
    # Compute the deviations from the centroid.
    Xc = X .- centroid
    # Compute the d×d covariance matrix.
    C = Xc * transpose(Xc) / n
    # Compute the eigen-decomposition of the symmetric covariance matrix.
    eig = eigen(Symmetric(C))
    # The eigenvector corresponding to the smallest eigenvalue defines the direction
    # of least variance—that is, the local normal to the boundary.
    min_idx = argmin(eig.values)
    normal = eig.vectors[:, min_idx]
    # Project the particle position onto the hyperplane defined by the centroid and the normal.
    diff = particle_pos .- vec(centroid)  # Ensure centroid is a vector.
    proj = particle_pos .- dot(diff, normal) * normal
    # Compute the contact normal as the unit vector from the projection to the particle.
    diff2 = particle_pos .- proj
    norm_diff = norm(diff2)
    contact_normal = norm_diff > 0 ? diff2 / norm_diff : normal
    # The effective contact point is the projection offset by effective_radius along the contact normal.
    cp_effective = proj .+ effective_radius * contact_normal
    return cp_effective, contact_normal
end
