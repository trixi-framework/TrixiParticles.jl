struct BidirectionalFlow end

struct InFlow end

struct OutFlow end

@doc raw"""
    BoundaryZone(; plane, plane_normal, density, particle_spacing,
                 initial_condition=nothing, extrude_geometry=nothing,
                 open_boundary_layers::Integer, average_inflow_velocity=true,
                 boundary_type=BidirectionalFlow(),
                 reference_density=nothing, reference_pressure=nothing,
                 reference_velocity=nothing)

Boundary zone for [`OpenBoundarySPHSystem`](@ref).

The specified plane (line in 2D or rectangle in 3D) will be extruded in the direction
opposite to `plane_normal` to create a box for the boundary zone.
There are three ways to specify the actual shape of the boundary zone:
1. Don't pass `initial_condition` or `extrude_geometry`. The boundary zone box will then
   be filled with boundary particles (default).
2. Specify `extrude_geometry` by passing a 1D shape in 2D or a 2D shape in 3D,
   which is then extruded in the direction opposite to `plane_normal` to create the boundary particles.
   - In 2D, the shape must be either an initial condition with 2D coordinates, which lies
     on the line specified by `plane`, or an initial condition with 1D coordinates, which lies
     on the line specified by `plane` when a y-coordinate of `0` is added.
   - In 3D, the shape must be either an initial condition with 3D coordinates, which lies
     in the rectangle specified by `plane`, or an initial condition with 2D coordinates,
     which lies in the rectangle specified by `plane` when a z-coordinate of `0` is added.
3. Specify `initial_condition` by passing a 2D initial condition in 2D or a 3D initial condition in 3D,
   which will be used for the boundary particles.

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
- `plane_normal`: Vector defining the plane normal. It always points inside the fluid domain.
- `boundary_type=BidirectionalFlow()`: Specify the type of the boundary. Available types are
    - `InFlow()` for an inflow boundary
    - `OutFlow()` for an outflow boundary
    - `BidirectionalFlow()` (default) for an bidirectional flow boundary
- `open_boundary_layers`: Number of particle layers in the direction opposite to `plane_normal`.
- `particle_spacing`: The spacing between the particles (see [`InitialCondition`](@ref)).
- `density`: Particle density (see [`InitialCondition`](@ref)).
- `initial_condition=nothing`: `InitialCondition` for the inflow particles.
                               Particles outside the boundary zone will be removed.
                               Do not use together with `extrude_geometry`.
- `extrude_geometry=nothing`: 1D shape in 2D or 2D shape in 3D, which lies on the plane
                              and is extruded upstream to obtain the inflow particles.
                              See point 2 above for more details.
- `average_inflow_velocity=true`: If `true`, the extrapolated inflow velocity is averaged
                                  to impose a uniform inflow profile.
                                  When no velocity is prescribed at the inflow,
                                  the velocity is extrapolated from the fluid domain.
                                  Thus, turbulent flows near the inflow can lead to
                                  anisotropic buffer-particles distribution,
                                  resulting in a potential numerical instability.
                                  Averaging mitigates these effects.
- `reference_velocity`: Reference velocity is either a function mapping each particle's coordinates
                        and time to its velocity, an array where the ``i``-th column holds
                        the velocity of particle ``i`` or, for a constant fluid velocity,
                        a vector holding this velocity.
- `reference_pressure`: Reference pressure is either a function mapping each particle's coordinates
                        and time to its pressure, a vector holding the pressure of each particle,
                        or a scalar for a constant pressure over all particles.
- `reference_density`: Reference density is either a function mapping each particle's coordinates
                       and time to its density, a vector holding the density of each particle,
                       or a scalar for a constant density over all particles.

!!! note "Note"
    The reference values (`reference_velocity`, `reference_pressure`, `reference_density`)
    can also be set to `nothing`.
    In this case, they will either be extrapolated from the fluid domain ([BoundaryModelTafuniMirroring](@ref BoundaryModelTafuniMirroring))
    or evolved using the characteristic flow variables ([BoundaryModelLastiwkaCharacteristics](@ref BoundaryModelLastiwkaCharacteristics)).

# Examples
```julia
# 2D
plane_points = ([0.0, 0.0], [0.0, 1.0])
plane_normal=[1.0, 0.0]

inflow = BoundaryZone(; plane=plane_points, plane_normal, particle_spacing=0.1, density=1.0,
                      open_boundary_layers=4, boundary_type=InFlow())

# 3D
plane_points = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
plane_normal=[0.0, 0.0, 1.0]

outflow = BoundaryZone(; plane=plane_points, plane_normal, particle_spacing=0.1, density=1.0,
                       open_boundary_layers=4, boundary_type=OutFlow())

# 3D particles sampled as cylinder
circle = SphereShape(0.1, 0.5, (0.5, 0.5), 1.0, sphere_type=RoundSphere())

bidirectional_flow = BoundaryZone(; plane=plane_points, plane_normal, particle_spacing=0.1,
                                  density=1.0, extrude_geometry=circle, open_boundary_layers=4)
```

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct BoundaryZone{IC, S, ZO, ZW, FD, PN, R}
    initial_condition       :: IC
    spanning_set            :: S
    zone_origin             :: ZO
    zone_width              :: ZW
    flow_direction          :: FD
    plane_normal            :: PN
    reference_values        :: R
    average_inflow_velocity :: Bool
    prescribed_density      :: Bool
    prescribed_pressure     :: Bool
    prescribed_velocity     :: Bool
end

function BoundaryZone(; plane, plane_normal, density, particle_spacing,
                      initial_condition=nothing, extrude_geometry=nothing,
                      open_boundary_layers::Integer, average_inflow_velocity=true,
                      boundary_type=BidirectionalFlow(),
                      reference_density=nothing, reference_pressure=nothing,
                      reference_velocity=nothing)
    if open_boundary_layers <= 0
        throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
    end

    # `plane_normal` always points in fluid domain
    plane_normal_ = normalize(SVector(plane_normal...))

    if boundary_type isa BidirectionalFlow
        flow_direction = nothing

    elseif boundary_type isa InFlow
        # Unit vector pointing in downstream direction
        flow_direction = plane_normal_

    elseif boundary_type isa OutFlow
        # Unit vector pointing in downstream direction
        flow_direction = -plane_normal_
    end

    ic, spanning_set_, zone_origin,
    zone_width = set_up_boundary_zone(plane, plane_normal_, flow_direction, density,
                                      particle_spacing, initial_condition,
                                      extrude_geometry, open_boundary_layers;
                                      boundary_type=boundary_type)

    NDIMS = ndims(ic)
    ELTYPE = eltype(ic)
    if !(reference_velocity isa Function || isnothing(reference_velocity) ||
         (reference_velocity isa Vector && length(reference_velocity) == NDIMS))
        throw(ArgumentError("`reference_velocity` must be either a function mapping " *
                            "each particle's coordinates and time to its velocity, " *
                            "an array where the ``i``-th column holds the velocity of particle ``i`` " *
                            "or, for a constant fluid velocity, a vector of length $NDIMS for a $(NDIMS)D problem holding this velocity"))
    else
        if reference_velocity isa Function
            test_result = reference_velocity(zeros(NDIMS), 0.0)
            if length(test_result) != NDIMS
                throw(ArgumentError("`velocity` function must be of dimension $NDIMS"))
            end
        end
        # We need this dummy for type stability reasons
        velocity_dummy = SVector(ntuple(dim -> ELTYPE(Inf), NDIMS))
        velocity_ref = wrap_reference_function(reference_velocity, velocity_dummy)
    end

    if !(reference_pressure isa Function || reference_pressure isa Real ||
         isnothing(reference_pressure))
        throw(ArgumentError("`reference_pressure` must be either a function mapping " *
                            "each particle's coordinates and time to its pressure, " *
                            "a vector holding the pressure of each particle, or a scalar"))
    else
        if reference_pressure isa Function
            test_result = reference_pressure(zeros(NDIMS), 0.0)
            if length(test_result) != 1
                throw(ArgumentError("`reference_pressure` function must be a scalar function"))
            end
        end
        # We need this dummy for type stability reasons
        pressure_dummy = ELTYPE(Inf)
        pressure_ref = wrap_reference_function(reference_pressure, pressure_dummy)
    end

    if !(reference_density isa Function || reference_density isa Real ||
         isnothing(reference_density))
        throw(ArgumentError("`reference_density` must be either a function mapping " *
                            "each particle's coordinates and time to its density, " *
                            "a vector holding the density of each particle, or a scalar"))
    else
        if reference_density isa Function
            test_result = reference_density(zeros(NDIMS), 0.0)
            if length(test_result) != 1
                throw(ArgumentError("`reference_density` function must be a scalar function"))
            end
        end
        # We need this dummy for type stability reasons
        density_dummy = ELTYPE(Inf)
        density_ref = wrap_reference_function(reference_density, density_dummy)
    end

    prescribed_pressure = isnothing(reference_pressure) ? false : true
    prescribed_density = isnothing(reference_density) ? false : true
    prescribed_velocity = isnothing(reference_velocity) ? false : true

    reference_values = (reference_velocity=velocity_ref, reference_pressure=pressure_ref,
                        reference_density=density_ref)

    coordinates_svector = reinterpret(reshape, SVector{NDIMS, ELTYPE}, ic.coordinates)

    if prescribed_pressure
        ic.pressure .= pressure_ref.(coordinates_svector, 0)
    end
    if prescribed_density
        ic.density .= density_ref.(coordinates_svector, 0)
        ic.mass .= ic.density * ic.particle_spacing^NDIMS
    end
    if prescribed_velocity
        ic.velocity .= stack(velocity_ref.(coordinates_svector, 0))
    end

    return BoundaryZone(ic, spanning_set_, zone_origin, zone_width,
                        flow_direction, plane_normal_, reference_values,
                        average_inflow_velocity, prescribed_density, prescribed_pressure,
                        prescribed_velocity)
end

function boundary_type_name(boundary_zone::BoundaryZone)
    (; flow_direction, plane_normal) = boundary_zone

    # Bidirectional flow
    isnothing(flow_direction) && return "bidirectional_flow"

    # Outflow
    signbit(dot(flow_direction, plane_normal)) && return "outflow"

    # Inflow
    return "inflow"
end

function Base.show(io::IO, boundary_zone::BoundaryZone)
    @nospecialize boundary_zone # reduce precompilation time

    print(io, "BoundaryZone(")
    print(io, ") with ", nparticles(boundary_zone.initial_condition), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", boundary_zone::BoundaryZone)
    @nospecialize boundary_zone # reduce precompilation time

    if get(io, :compact, false)
        show(io, boundary_zone)
    else
        summary_header(io, "BoundaryZone")
        summary_line(io, "boundary type", boundary_type_name(boundary_zone))
        summary_line(io, "#particles", nparticles(boundary_zone.initial_condition))
        summary_line(io, "width", round(boundary_zone.zone_width, digits=3))
        summary_footer(io)
    end
end

function set_up_boundary_zone(plane, plane_normal, flow_direction, density,
                              particle_spacing, initial_condition, extrude_geometry,
                              open_boundary_layers; boundary_type)
    if boundary_type isa InFlow
        extrude_direction = -flow_direction
    elseif boundary_type isa OutFlow
        extrude_direction = flow_direction
    elseif boundary_type isa BidirectionalFlow
        # `plane_normal` is always pointing in the fluid domain
        extrude_direction = -plane_normal
    end

    # Sample particles in boundary zone
    if isnothing(initial_condition) && isnothing(extrude_geometry)
        initial_condition = TrixiParticles.extrude_geometry(plane; particle_spacing,
                                                            density,
                                                            direction=extrude_direction,
                                                            n_extrude=open_boundary_layers)
    elseif !isnothing(extrude_geometry)
        initial_condition = TrixiParticles.extrude_geometry(extrude_geometry;
                                                            particle_spacing,
                                                            density,
                                                            direction=extrude_direction,
                                                            n_extrude=open_boundary_layers)
    else
        initial_condition = initial_condition
    end

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    zone_width = open_boundary_layers * initial_condition.particle_spacing

    # Vectors spanning the boundary zone/box
    spanning_set, zone_origin = calculate_spanning_vectors(plane, zone_width)

    # First vector of `spanning_vectors` is normal to the boundary plane.
    dot_plane_normal = dot(normalize(spanning_set[:, 1]), plane_normal)

    if !isapprox(abs(dot_plane_normal), 1.0, atol=1e-7)
        throw(ArgumentError("`plane_normal` is not normal to the boundary plane"))
    end

    if boundary_type == InFlow()
        # First vector of `spanning_vectors` is normal to the boundary plane
        dot_flow = dot(normalize(spanning_set[:, 1]), flow_direction)

        # The vector must point in upstream direction for an inflow boundary.
        # Flip the normal vector to point in the opposite direction of `flow_direction`.
        spanning_set[:, 1] .*= -sign(dot_flow)

    elseif boundary_type == OutFlow()
        # First vector of `spanning_vectors` is normal to the boundary plane
        dot_flow = dot(normalize(spanning_set[:, 1]), flow_direction)

        # The vector must point in downstream direction for an outflow boundary.
        # Flip the normal vector to point in `flow_direction`.
        spanning_set[:, 1] .*= sign(dot_flow)

    elseif boundary_type == BidirectionalFlow()
        # Flip the normal vector to point opposite to fluid domain
        spanning_set[:, 1] .*= -sign(dot_plane_normal)
    end

    spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

    # Remove particles outside the boundary zone.
    # This check is only necessary when `initial_condition` or `extrude_geometry` are passed.
    ic = remove_outside_particles(initial_condition, spanning_set_, zone_origin)

    return ic, spanning_set_, zone_origin, zone_width
end

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

@inline function is_in_boundary_zone(boundary_zone, particle_coords)
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

function update_boundary_zone_indices!(system, u, boundary_zones, semi)
    set_zero!(system.boundary_zone_indices)

    @threaded semi for particle in each_moving_particle(system)
        particle_coords = current_coords(u, system, particle)

        for (zone_id, boundary_zone) in enumerate(boundary_zones)
            # Check if boundary particle is in the boundary zone
            if is_in_boundary_zone(boundary_zone, particle_coords)
                system.boundary_zone_indices[particle] = zone_id
            end
        end
    end

    return system
end

function current_boundary_zone(system, particle)
    return system.boundary_zones[system.boundary_zone_indices[particle]]
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

function wrap_reference_function(function_::Nothing, ref_dummy)
    # Return a dummy value for type stability
    return @inline((coords, t)->ref_dummy)
end

function wrap_reference_function(function_::Function, ref_dummy)
    # Already a function
    return function_
end

function wrap_reference_function(constant_scalar_::Number, ref_dummy)
    return @inline((coords, t)->constant_scalar_)
end

function wrap_reference_function(constant_vector_::AbstractVector,
                                 ref_dummy::SVector{NDIMS, ELTYPE}) where {NDIMS, ELTYPE}
    return @inline((coords, t)->SVector{NDIMS, ELTYPE}(constant_vector_))
end

function apply_reference_pressure(system, particle, pos, t)
    (; pressure_references) = system.cache

    zone_id = system.boundary_zone_indices[particle]

    return apply_ith_function(pressure_references, zone_id, pos, t)
end

function apply_reference_density(system, particle, pos, t)
    (; density_references) = system.cache

    zone_id = system.boundary_zone_indices[particle]

    return apply_ith_function(density_references, zone_id, pos, t)
end

function apply_reference_velocity(system, particle, pos, t)
    (; velocity_references) = system.cache

    zone_id = system.boundary_zone_indices[particle]

    return apply_ith_function(velocity_references, zone_id, pos, t)
end
