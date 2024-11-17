@doc raw"""
    InFlow(; plane, flow_direction, density, particle_spacing,
           initial_condition=nothing, extrude_geometry=nothing,
           open_boundary_layers::Integer)

Inflow boundary zone for [`OpenBoundarySPHSystem`](@ref).

The specified plane (line in 2D or rectangle in 3D) will be extruded in upstream
direction (the direction opposite to `flow_direction`) to create a box for the boundary zone.
There are three ways to specify the actual shape of the inflow:
1. Don't pass `initial_condition` or `extrude_geometry`. The boundary zone box will then
   be filled with inflow particles (default).
2. Specify `extrude_geometry` by passing a 1D shape in 2D or a 2D shape in 3D,
   which is then extruded in upstream direction to create the inflow particles.
   - In 2D, the shape must be either an initial condition with 2D coordinates, which lies
     on the line specified by `plane`, or an initial condition with 1D coordinates, which lies
     on the line specified by `plane` when a y-coordinate of `0` is added.
   - In 3D, the shape must be either an initial condition with 3D coordinates, which lies
     in the rectangle specified by `plane`, or an initial condition with 2D coordinates,
     which lies in the rectangle specified by `plane` when a z-coordinate of `0` is added.
3. Specify `initial_condition` by passing a 2D initial condition in 2D or a 3D initial condition in 3D,
   which will be used for the inflow particles.

!!! note "Note"
    Particles outside the boundary zone box will be removed.

# Keywords
- `plane`: Tuple of points defining a part of the surface of the domain.
           The points must either span a line in 2D or a rectangle in 3D.
           This line or rectangle is then extruded in upstream direction to obtain
           the boundary zone.
           In 2D, pass two points ``(A, B)``, so that the interval ``[A, B]`` is
           the inflow surface.
           In 3D, pass three points ``(A, B, C)``, so that the rectangular inflow surface
           is spanned by the vectors ``\widehat{AB}`` and ``\widehat{AC}``.
           These two vectors must be orthogonal.
- `flow_direction`: Vector defining the flow direction.
- `open_boundary_layers`: Number of particle layers in upstream direction.
- `particle_spacing`: The spacing between the particles (see [`InitialCondition`](@ref)).
- `density`: Particle density (see [`InitialCondition`](@ref)).
- `initial_condition=nothing`: `InitialCondition` for the inflow particles.
                               Particles outside the boundary zone will be removed.
                               Do not use together with `extrude_geometry`.
- `extrude_geometry=nothing`: 1D shape in 2D or 2D shape in 3D, which lies on the plane
                              and is extruded upstream to obtain the inflow particles.
                              See point 2 above for more details.

# Examples
```julia
# 2D
plane_points = ([0.0, 0.0], [0.0, 1.0])
flow_direction=[1.0, 0.0]

inflow = InFlow(; plane=plane_points, particle_spacing=0.1, flow_direction, density=1.0,
                open_boundary_layers=4)

# 3D
plane_points = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
flow_direction=[0.0, 0.0, 1.0]

inflow = InFlow(; plane=plane_points, particle_spacing=0.1, flow_direction, density=1.0,
                open_boundary_layers=4)

# 3D particles sampled as cylinder
circle = SphereShape(0.1, 0.5, (0.5, 0.5), 1.0, sphere_type=RoundSphere())

inflow = InFlow(; plane=plane_points, particle_spacing=0.1, flow_direction, density=1.0,
                extrude_geometry=circle, open_boundary_layers=4)
```

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct InFlow{NDIMS, IC, S, ZO, ZW, FD}
    initial_condition :: IC
    spanning_set      :: S
    zone_origin       :: ZO
    zone_width        :: ZW
    flow_direction    :: FD

    function InFlow(; plane, flow_direction, density, particle_spacing,
                    initial_condition=nothing, extrude_geometry=nothing,
                    open_boundary_layers::Integer)
        if open_boundary_layers <= 0
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction
        flow_direction_ = normalize(SVector(flow_direction...))

        # Sample particles in boundary zone
        if isnothing(initial_condition) && isnothing(extrude_geometry)
            initial_condition = TrixiParticles.extrude_geometry(plane; particle_spacing,
                                                                density,
                                                                direction=-flow_direction_,
                                                                n_extrude=open_boundary_layers)
        elseif !isnothing(extrude_geometry)
            initial_condition = TrixiParticles.extrude_geometry(extrude_geometry;
                                                                particle_spacing,
                                                                density,
                                                                direction=-flow_direction_,
                                                                n_extrude=open_boundary_layers)
        end

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        # Vectors spanning the boundary zone/box
        spanning_set, zone_origin = calculate_spanning_vectors(plane, zone_width)

        # First vector of `spanning_vectors` is normal to the inflow plane.
        # The normal vector must point in upstream direction for an inflow boundary.
        dot_ = dot(normalize(spanning_set[:, 1]), flow_direction_)

        if !isapprox(abs(dot_), 1.0, atol=1e-7)
            throw(ArgumentError("`flow_direction` is not normal to inflow plane"))
        end

        # Flip the normal vector to point in the opposite direction of `flow_direction`
        spanning_set[:, 1] .*= -sign(dot_)

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        # Remove particles outside the boundary zone.
        # This check is only necessary when `initial_condition` or `extrude_geometry` are passed.
        ic = remove_outside_particles(initial_condition, spanning_set_, zone_origin)

        return new{NDIMS, typeof(ic), typeof(spanning_set_), typeof(zone_origin),
                   typeof(zone_width),
                   typeof(flow_direction_)}(ic, spanning_set_, zone_origin, zone_width,
                                            flow_direction_)
    end
end

@doc raw"""
    OutFlow(; plane, flow_direction, density, particle_spacing,
            initial_condition=nothing, extrude_geometry=nothing,
            open_boundary_layers::Integer)

Outflow boundary zone for [`OpenBoundarySPHSystem`](@ref).

The specified plane (line in 2D or rectangle in 3D) will be extruded in downstream
direction (the direction in `flow_direction`) to create a box for the boundary zone.
There are three ways to specify the actual shape of the outflow:
1. Don't pass `initial_condition` or `extrude_geometry`. The boundary zone box will then
   be filled with outflow particles (default).
2. Specify `extrude_geometry` by passing a 1D shape in 2D or a 2D shape in 3D,
   which is then extruded in downstream direction to create the outflow particles.
    - In 2D, the shape must be either an initial condition with 2D coordinates, which lies
      on the line specified by `plane`, or an initial condition with 1D coordinates, which lies
      on the line specified by `plane` when a y-coordinate of `0` is added.
    -  In 3D, the shape must be either an initial condition with 3D coordinates, which lies
      in the rectangle specified by `plane`, or an initial condition with 2D coordinates,
      which lies in the rectangle specified by `plane` when a z-coordinate of `0` is added.
3. Specify `initial_condition` by passing a 2D initial condition in 2D or a 3D initial condition in 3D,
   which will be used for the outflow particles.

!!! note "Note"
    Particles outside the boundary zone box will be removed.

# Keywords
- `plane`: Tuple of points defining a part of the surface of the domain.
           The points must either span a line in 2D or a rectangle in 3D.
           This line or rectangle is then extruded in downstream direction to obtain
           the boundary zone.
           In 2D, pass two points ``(A, B)``, so that the interval ``[A, B]`` is
           the outflow surface.
           In 3D, pass three points ``(A, B, C)``, so that the rectangular outflow surface
           is spanned by the vectors ``\widehat{AB}`` and ``\widehat{AC}``.
           These two vectors must be orthogonal.
- `flow_direction`: Vector defining the flow direction.
- `open_boundary_layers`: Number of particle layers in downstream direction.
- `particle_spacing`: The spacing between the particles (see [`InitialCondition`](@ref)).
- `density`: Particle density (see [`InitialCondition`](@ref)).
- `initial_condition=nothing`: `InitialCondition` for the outflow particles.
                               Particles outside the boundary zone will be removed.
                               Do not use together with `extrude_geometry`.
- `extrude_geometry=nothing`: 1D shape in 2D or 2D shape in 3D, which lies on the plane
                              and is extruded downstream to obtain the outflow particles.
                              See point 2 above for more details.

# Examples
```julia
# 2D
plane_points = ([0.0, 0.0], [0.0, 1.0])
flow_direction = [1.0, 0.0]

outflow = OutFlow(; plane=plane_points, particle_spacing=0.1, flow_direction, density=1.0,
                  open_boundary_layers=4)

# 3D
plane_points = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
flow_direction = [0.0, 0.0, 1.0]

outflow = OutFlow(; plane=plane_points, particle_spacing=0.1, flow_direction, density=1.0,
                  open_boundary_layers=4)

# 3D particles sampled as cylinder
circle = SphereShape(0.1, 0.5, (0.5, 0.5), 1.0, sphere_type=RoundSphere())

outflow = OutFlow(; plane=plane_points, particle_spacing=0.1, flow_direction, density=1.0,
                  extrude_geometry=circle, open_boundary_layers=4)
```

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct OutFlow{NDIMS, IC, S, ZO, ZW, FD}
    initial_condition :: IC
    spanning_set      :: S
    zone_origin       :: ZO
    zone_width        :: ZW
    flow_direction    :: FD

    function OutFlow(; plane, flow_direction, density, particle_spacing,
                     initial_condition=nothing, extrude_geometry=nothing,
                     open_boundary_layers::Integer)
        if open_boundary_layers <= 0
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction
        flow_direction_ = normalize(SVector(flow_direction...))

        # Sample particles in boundary zone
        if isnothing(initial_condition) && isnothing(extrude_geometry)
            initial_condition = TrixiParticles.extrude_geometry(plane; particle_spacing,
                                                                density,
                                                                direction=flow_direction_,
                                                                n_extrude=open_boundary_layers)
        elseif !isnothing(extrude_geometry)
            initial_condition = TrixiParticles.extrude_geometry(extrude_geometry;
                                                                particle_spacing, density,
                                                                direction=flow_direction_,
                                                                n_extrude=open_boundary_layers)
        end

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        # Vectors spanning the boundary zone/box
        spanning_set, zone_origin = calculate_spanning_vectors(plane, zone_width)

        # First vector of `spanning_vectors` is normal to the outflow plane.
        # The normal vector must point in downstream direction for an outflow boundary.
        dot_ = dot(normalize(spanning_set[:, 1]), flow_direction_)

        if !isapprox(abs(dot_), 1.0, atol=1e-7)
            throw(ArgumentError("`flow_direction` is not normal to outflow plane"))
        end

        # Flip the normal vector to point in `flow_direction`
        spanning_set[:, 1] .*= sign(dot_)

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        # Remove particles outside the boundary zone.
        # This check is only necessary when `initial_condition` or `extrude_geometry` are passed.
        ic = remove_outside_particles(initial_condition, spanning_set_, zone_origin)

        return new{NDIMS, typeof(ic), typeof(spanning_set_), typeof(zone_origin),
                   typeof(zone_width),
                   typeof(flow_direction_)}(ic, spanning_set_, zone_origin, zone_width,
                                            flow_direction_)
    end
end

@inline Base.ndims(::Union{InFlow{NDIMS}, OutFlow{NDIMS}}) where {NDIMS} = NDIMS

function calculate_spanning_vectors(plane, zone_width)
    return spanning_vectors(Tuple(plane), zone_width), SVector(plane[1]...)
end

function spanning_vectors(plane_points::NTuple{2}, zone_width)
    plane_size = plane_points[2] - plane_points[1]

    # Calculate normal vector of plane
    b = normalize([-plane_size[2], plane_size[1]]) * zone_width

    return hcat(b, plane_size)
end

function spanning_vectors(plane_points::NTuple{3}, zone_width)
    # Vectors spanning the plane
    edge1 = plane_points[2] - plane_points[1]
    edge2 = plane_points[3] - plane_points[1]

    # Check if the edges are linearly dependent (to avoid degenerate planes)
    if isapprox(norm(cross(edge1, edge2)), 0.0; atol=eps())
        throw(ArgumentError("the vectors `AB` and `AC` must not be collinear"))
    end

    # Calculate normal vector of plane
    c = Vector(normalize(cross(edge2, edge1)) * zone_width)

    return hcat(c, edge1, edge2)
end

@inline function is_in_boundary_zone(boundary_zone::Union{InFlow, OutFlow}, particle_coords)
    (; zone_origin, spanning_set) = boundary_zone
    particle_position = particle_coords - zone_origin

    return is_in_boundary_zone(spanning_set, particle_position)
end

@inline function is_in_boundary_zone(spanning_set::AbstractArray,
                                     particle_position::SVector{NDIMS}) where {NDIMS}
    for dim in 1:NDIMS
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone
        if !(0 <= dot(particle_position, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in boundary zone
            return false
        end
    end

    # Particle is in boundary zone
    return true
end

function remove_outside_particles(initial_condition, spanning_set, zone_origin)
    (; coordinates, density, particle_spacing) = initial_condition

    in_zone = fill(true, nparticles(initial_condition))

    for particle in eachparticle(initial_condition)
        current_position = current_coords(coordinates, initial_condition, particle)
        particle_position = current_position - zone_origin

        in_zone[particle] = is_in_boundary_zone(spanning_set, particle_position)
    end

    return InitialCondition(; coordinates=coordinates[:, in_zone], density=first(density),
                            particle_spacing)
end
