# Criteria of refinement:
#
# - fixed (/moving?) refinement zone
# - number of neighbors
# - problem specific criteria (e.g. high velocity gradient)

abstract type RefinementCriteria{NDIMS, ELTYPE} end

struct RefinementZone{NDIMS, ELTYPE, ZO} <: RefinementCriteria{NDIMS, ELTYPE}
    zone_origin  :: ZO
    spanning_set :: Vector{SVector}

    function RefinementZone(edge_lengths;
                            zone_origin=ntuple(_ -> 0.0, length(edge_lengths)),
                            rotation=nothing) # TODO
        NDIMS = length(edge_lengths)
        ELTYPE = eltype(edge_lengths)

        if zone_origin isa Function
            zone_origin_function = zone_origin
        else
            zone_origin_function = (v, u, v_ode, u_ode, t, system, semi) -> SVector(zone_origin...)
        end

        # Vectors spanning the zone.
        spanning_set_ = I(NDIMS) .* edge_lengths'

        if !isnothing(rotation)
            # rotate vecs
            spanning_set = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set_)
        else
            spanning_set = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set_)

        end

        return new{NDIMS, ELTYPE,
                   typeof(zone_origin_function)}(zone_origin_function, spanning_set)
    end
end

@inline Base.ndims(::RefinementCriteria{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::RefinementCriteria{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline function (refinement_criterion::RefinementZone)(system, particle,
                                                        v, u, v_ode, u_ode, semi, t;
                                                        padding=0.0)
    (; zone_origin, spanning_set) = refinement_criterion
    particle_position = current_coords(u, system, particle) -
                        zone_origin(v, u, v_ode, u_ode, t, system, semi)

    for dim in 1:ndims(system)
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone.
        if !(padding < dot(particle_position, span_dim) < dot(span_dim, span_dim) - padding)

            # Particle is not in refinement zone.
            return false
        end
    end

    # Particle is in refinement zone.
    return true
end
