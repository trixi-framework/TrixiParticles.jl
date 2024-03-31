# Criteria of refinement:
#
# - fixed (/moving?) refinement zone
# - number of neighbors
# - problem specific criteria (e.g. high velocity gradient)

abstract type RefinementCriteria{NDIMS, ELTYPE} end

struct RefinementZone{NDIMS, ELTYPE, ZO} <: RefinementCriteria{NDIMS, ELTYPE}
    zone_origin  :: ZO
    spanning_set :: Vector{SVector}

    function RefinementZone(; edge_length_x=0.0, edge_length_y=0.0, edge_length_z=nothing,
                            zone_origin, rotation=nothing) # TODO
        if isnothing(edge_length_z)
            NDIMS = 2
        elseif edge_length_z < eps()
            throw(ArgumentError("`edge_length_z` must be either `nothing` for a 2D problem or greater than zero for a 3D problem"))
        else
            NDIMS = 3
        end

        ELTYPE = eltype(edge_length_x)

        if edge_length_x * edge_length_y < eps()
            throw(ArgumentError("edge lengths must be greater than zero"))
        end

        if length(zone_origin) != NDIMS
            throw(ArgumentError("`zone_origin` must be a `Vector` of size $NDIMS for a $NDIMS-D Problem"))
        end

        if zone_origin isa Function
            zone_origin_function = zone_origin
        else
            zone_origin_function = (v, u, v_ode, u_ode, t, system, semi) -> SVector(zone_origin...)
        end

        edge_lengths = if NDIMS == 2
            [edge_length_x, edge_length_y]
        else
            [edge_length_x, edge_length_y, edge_length_z]
        end

        # Vectors spanning the zone.
        spanning_set_ = I(NDIMS) .* edge_lengths'

        spanning_set = if !isnothing(rotation)
            # rotate vecs
            reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set_)
        else
            reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set_)
        end

        return new{NDIMS, ELTYPE,
                   typeof(zone_origin_function)}(zone_origin_function, spanning_set)
    end
end

@inline Base.ndims(::RefinementCriteria{NDIMS}) where {NDIMS} = NDIMS
@inline Base.eltype(::RefinementCriteria{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = ELTYPE

@inline function (refinement_criterion::RefinementZone)(system, particle,
                                                        v, u, v_ode, u_ode, semi, t;
                                                        padding=false)
    (; smoothing_length) = system
    (; zone_origin, spanning_set) = refinement_criterion
    particle_position = current_coords(u, system, particle) -
                        zone_origin(v, u, v_ode, u_ode, t, system, semi)

    padding = padding ? 0.5smoothing_length : 0.0

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
