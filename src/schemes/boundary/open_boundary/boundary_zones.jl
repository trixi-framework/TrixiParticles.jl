"""
    InFlow

Inflow boundary zone for [`OpenBoundarySPHSystem`](@ref)

# Keywords
- `plane`: Points defining the boundary zones front plane.
           The points must either span a rectangular plane in 3D or a line in 2D.
- `flow_direction`: Vector defining the flow direction.
- `open_boundary_layers`: Number of particle layers in upstream direction.
- `initial_condition`: TODO
- `particle_spacing`: TODO
- `density`: TODO
"""
struct InFlow{NDIMS, IC, S, ZO, ZW, FD}
    initial_condition :: IC
    spanning_set      :: S
    zone_origin       :: ZO
    zone_width        :: ZW
    flow_direction    :: FD

    function InFlow(; plane=nothing, flow_direction, density=nothing,
                    particle_spacing=nothing, initial_condition=nothing,
                    open_boundary_layers::Integer=0)
        if open_boundary_layers < sqrt(eps())
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction.
        flow_direction_ = normalize(SVector(flow_direction...))

        if isnothing(initial_condition)
            # Sample particles in boundary zone.
            initial_condition = extrude_geometry(plane; particle_spacing, density,
                                                 direction=-flow_direction_,
                                                 n_extrude=open_boundary_layers)
        end

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        # Vectors spanning the boundary zone/box.
        spanning_set, zone_origin = calculate_spanning_vectors(plane, zone_width)

        # First vector of `spanning_vectors` is normal to the inflow plane.
        # The normal vector must point in upstream direction for an inflow boundary.
        dot_ = dot(normalize(spanning_set[:, 1]), flow_direction_)

        if !isapprox(abs(dot_), 1.0, atol=1e-7)
            throw(ArgumentError("flow direction and normal vector of " *
                                "inflow-plane do not correspond"))
        else
            # Flip the inflow vector correspondingly
            spanning_set[:, 1] .*= -dot_
        end

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        return new{NDIMS, typeof(initial_condition),
                   typeof(spanning_set_), typeof(zone_origin), typeof(zone_width),
                   typeof(flow_direction_)}(initial_condition, spanning_set_, zone_origin,
                                            zone_width, flow_direction_)
    end
end

"""
    OutFlow

Outflow boundary zone for [`OpenBoundarySPHSystem`](@ref)


# Keywords
- `plane`: Points defining the boundary zones front plane.
           The points must either span a rectangular plane in 3D or a line in 2D.
- `flow_direction`: Vector defining the flow direction.
- `open_boundary_layers`: Number of particle layers in upstream direction.
- `initial_condition`: TODO
- `particle_spacing`: TODO
- `density`: TODO
"""
struct OutFlow{NDIMS, IC, S, ZO, ZW, FD}
    initial_condition :: IC
    spanning_set      :: S
    zone_origin       :: ZO
    zone_width        :: ZW
    flow_direction    :: FD

    function OutFlow(; plane=nothing, flow_direction, density=nothing,
                     particle_spacing=nothing, initial_condition=nothing,
                     open_boundary_layers::Integer=0)
        if open_boundary_layers < sqrt(eps())
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction.
        flow_direction_ = normalize(SVector(flow_direction...))

        if isnothing(initial_condition)
            # Sample particles in boundary zone.
            initial_condition = extrude_geometry(plane; particle_spacing,
                                                 direction=flow_direction_, density,
                                                 n_extrude=open_boundary_layers)
        end

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        # Vectors spanning the boundary zone/box.
        spanning_set, zone_origin = calculate_spanning_vectors(plane, zone_width)

        # First vector of `spanning_vectors` is normal to the outflow plane.
        # The normal vector must point in downstream direction for an outflow boundary.
        dot_ = dot(normalize(spanning_set[:, 1]), flow_direction_)

        if !isapprox(abs(dot_), 1.0, atol=1e-7)
            throw(ArgumentError("flow direction and normal vector of " *
                                "outflow-plane do not correspond"))
        else
            # Flip the inflow vector correspondingly
            spanning_set[:, 1] .*= dot_
        end

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        return new{NDIMS, typeof(initial_condition),
                   typeof(spanning_set_), typeof(zone_origin), typeof(zone_width),
                   typeof(flow_direction_)}(initial_condition, spanning_set_, zone_origin,
                                            zone_width, flow_direction_)
    end
end

@inline Base.ndims(::Union{InFlow{NDIMS}, OutFlow{NDIMS}}) where {NDIMS} = NDIMS

# function calculate_spanning_vectors(plane::Shapes, zone_width)
#     # TODO: Handle differently
# end

function calculate_spanning_vectors(plane, zone_width)
    return spanning_vectors(plane, zone_width), SVector(plane[1]...)
end

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
