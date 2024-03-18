abstract type RefinementCriteria{NDIMS, ELTYPE} end

struct CubicSplitting{ELTYPE}
    epsilon :: ELTYPE
    alpha   :: ELTYPE

    function CubicSplitting(; epsilon=0.5, alpha=0.5)
        new{typeof(epsilon)}(epsilon, alpha)
    end
end

function (refinement_pattern::CubicSplitting)(system::System{2})
    (; initial_condition) = system
    (; particle_spacing) = initial_condition
    (; epsilon) = refinement_pattern

    direction_1 = normalize([1.0, 1.0])
    direction_2 = normalize([1.0, -1.0])
    direction_3 = -direction_1
    direction_4 = -direction_2

    relative_position = hcat(particle_spacing * epsilon * direction_1,
                             particle_spacing * epsilon * direction_2,
                             particle_spacing * epsilon * direction_3,
                             particle_spacing * epsilon * direction_4)

    return reinterpret(reshape, SVector{2, typeof(epsilon)}, relative_position)
end

# TODO: Clarify refinement pattern. Cubic splitting? Triangular or hexagonal?
# See https://www.sciencedirect.com/science/article/pii/S0020740319317023
@inline nchilds(system, rp::CubicSplitting) = 2^ndims(system)

@inline mass_child(system, mass, rp::CubicSplitting) = mass / nchilds(system, rp)

@inline smoothing_length_child(system, refinement_pattern) = refinement_pattern.alpha *
                                                             system.smoothing_length

@inline particle_spacing_child(system, refinement_pattern) = system.initial_condition.particle_spacing *
                                                             refinement_pattern.epsilon

# ==== Refinement criteria
struct RefinementZone{NDIMS, ELTYPE} <: RefinementCriteria{NDIMS, ELTYPE}
    zone_origin  :: SVector{NDIMS, ELTYPE}
    spanning_set :: Vector{SVector}

    function RefinementZone(plane_points, zone_width)
        NDIMS = length(plane_points)
        ELTYPE = typeof(zone_width)

        # Vectors spanning the zone.
        spanning_set = spanning_vectors(plane_points, zone_width)

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        zone_origin = SVector(plane_points[1]...)

        return new{NDIMS, ELTYPE}(zone_origin, spanning_set_)
    end
end

@inline Base.ndims(::RefinementCriteria{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::RefinementCriteria{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

function spanning_vectors(plane_points, zone_width)

    # Convert to tuple
    return spanning_vectors(tuple(plane_points...), zone_width)
end

function spanning_vectors(plane_points::NTuple{2}, zone_width)
    plane_size = plane_points[2] - plane_points[1]

    # Calculate normal vector of plane
    b = Vector(normalize([-plane_size[2]; plane_size[1]]) * zone_width)

    return hcat(b, plane_size)
end

function spanning_vectors(plane_points::NTuple{3}, zone_width)
    # Vectors spanning the plane
    edge1 = plane_points[2] - plane_points[1]
    edge2 = plane_points[3] - plane_points[1]

    if !isapprox(dot(edge1, edge2), 0.0, atol=1e-7)
        throw(ArgumentError("the provided points do not span a rectangular plane"))
    end

    # Calculate normal vector of plane
    c = Vector(normalize(cross(edge2, edge1)) * zone_width)

    return hcat(c, edge1, edge2)
end

@inline function (refinement_criterion::RefinementZone)(system, particle,
                                                        v, u, v_ode, u_ode, semi)
    (; zone_origin, spanning_set) = refinement_criterion
    particle_position = current_coords(u, system, particle) - zone_origin

    for dim in 1:ndims(system)
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone.
        if !(0 <= dot(particle_position, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in refinement zone.
            return false
        end
    end

    # Particle is in refinement zone.
    return true
end
