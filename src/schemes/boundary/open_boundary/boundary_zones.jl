struct BidirectionalFlow end

struct InFlow end

struct OutFlow end

@doc raw"""
    BoundaryZone(; boundary_face, face_normal, density, particle_spacing,
                 initial_condition=nothing, extrude_geometry=nothing,
                 open_boundary_layers::Integer, average_inflow_velocity=true,
                 boundary_type=BidirectionalFlow(),
                 rest_pressure=zero(eltype(density)),
                 reference_density=nothing, reference_pressure=nothing,
                 reference_velocity=nothing)

Boundary zone for [`OpenBoundarySystem`](@ref).

The specified `boundary_face` (line in 2D or rectangle in 3D) will be extruded in the direction
opposite to `face_normal` to create a box for the boundary zone.
To specify the `boundary_face`, pass the required vertices as described below.
For complex 3D simulations, these vertices can also be extracted from an STL file
(see [`planar_geometry_to_face`](@ref)).
There are three ways to specify the actual shape of the boundary zone:
1. Don't pass `initial_condition` or `extrude_geometry`. The boundary zone box will then
   be filled with boundary particles (default).
2. Specify `extrude_geometry` by passing a 1D shape in 2D or a 2D shape in 3D,
   which is then extruded in the direction opposite to `face_normal` to create the boundary particles.
   - In 2D, the shape must be either an initial condition with 2D coordinates, which lies
     on the line specified by `boundary_face`, or an initial condition with 1D coordinates,
     which lies on the line specified by `boundary_face` when a y-coordinate of `0` is added.
   - In 3D, the shape must be either an initial condition with 3D coordinates, which lies
     in the rectangle specified by `boundary_face`, or an initial condition with 2D coordinates,
     which lies in the rectangle specified by `boundary_face` when a z-coordinate of `0` is added.
3. Specify `initial_condition` by passing a 2D initial condition in 2D or a 3D initial condition in 3D,
   which will be used for the boundary particles.

!!! note "Note"
    Particles outside the boundary zone box will be removed.

# Keywords
- `boundary_face`: Tuple of vertices defining a part of the surface of the domain.
                   The vertices must either span a line in 2D or a rectangle in 3D.
                   This line or rectangle is then extruded in upstream direction to obtain
                   the boundary zone.
                   In 2D, pass two vertices ``(A, B)``, so that the interval ``[A, B]`` is
                   the inflow surface.
                   In 3D, pass three vertices ``(A, B, C)``, so that the rectangular inflow surface
                   is spanned by the vectors ``\widehat{AB}`` and ``\widehat{AC}``.
                   These two vectors must be orthogonal.
- `face_normal`: Vector defining the normal of the `boundary_face`. It always points inside the fluid domain.
- `boundary_type=BidirectionalFlow()`: Specify the type of the boundary. Available types are
    - `InFlow()` for an inflow boundary
    - `OutFlow()` for an outflow boundary
    - `BidirectionalFlow()` (default) for an bidirectional flow boundary
- `open_boundary_layers`: Number of particle layers in the direction opposite to `face_normal`.
- `particle_spacing`: The spacing between the particles (see [`InitialCondition`](@ref)).
- `density`: Particle density (see [`InitialCondition`](@ref)).
- `initial_condition=nothing`: `InitialCondition` for the inflow particles.
                               Particles outside the boundary zone will be removed.
                               Do not use together with `extrude_geometry`.
- `extrude_geometry=nothing`: 1D shape in 2D or 2D shape in 3D, which lies on the `boundary_face`
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
                        and time to its velocity, or, for a constant fluid velocity,
                        a vector holding this velocity.
- `reference_pressure`: Reference pressure is either a function mapping each particle's coordinates
                        and time to its pressure, or a scalar for a constant pressure over all particles.
- `reference_density`: Reference density is either a function mapping each particle's coordinates
                       and time to its density, or a scalar for a constant density over all particles.
- `rest_pressure=0.0`: For `BoundaryModelDynamicalPressureZhang`, a rest pressure is required when the pressure is not prescribed.
                       This should match the rest pressure of the fluid system.
                       Per default it is set to zero (assuming a gauge pressure system).
                       - For `EntropicallyDampedSPHSystem`: Use the initial pressure from the `InitialCondition`
                       - For `WeaklyCompressibleSPHSystem`: Use the background pressure from the equation of state

!!! note "Note"
    The reference values (`reference_velocity`, `reference_pressure`, `reference_density`)
    can also be set to `nothing`.
    In this case, they will either be extrapolated from the fluid domain ([BoundaryModelMirroringTafuni](@ref BoundaryModelMirroringTafuni))
    or evolved using the characteristic flow variables ([BoundaryModelCharacteristicsLastiwka](@ref BoundaryModelCharacteristicsLastiwka)).

# Examples
```jldoctest; output=false
# 2D
face_vertices = ([0.0, 0.0], [0.0, 1.0])
face_normal = [1.0, 0.0]

# Constant reference velocity:
velocity_const = [1.0, 0.0]

inflow_1 = BoundaryZone(; boundary_face=face_vertices, face_normal, particle_spacing=0.1,
                        density=1.0, open_boundary_layers=4, boundary_type=InFlow(),
                        reference_velocity=velocity_const)

# Reference velocity as a function (parabolic velocity profile):
velocity_func = (pos, t) -> SVector(4.0 * pos[2] * (1.0 - pos[2]), 0.0)

inflow_2 = BoundaryZone(; boundary_face=face_vertices, face_normal, particle_spacing=0.1,
                        density=1.0, open_boundary_layers=4, boundary_type=InFlow(),
                        reference_velocity=velocity_func)

# 3D
face_vertices = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
face_normal = [0.0, 0.0, 1.0]

# Constant reference pressure:
pressure_const = 0.0

outflow_1 = BoundaryZone(; boundary_face=face_vertices, face_normal, particle_spacing=0.1,
                         density=1.0, open_boundary_layers=4, boundary_type=OutFlow(),
                         reference_pressure=pressure_const)

# Reference pressure as a function (y-dependent profile, sinusoidal in time):
pressure_func = (pos, t) -> pos[2] * sin(2pi * t)

outflow_2 = BoundaryZone(; boundary_face=face_vertices, face_normal, particle_spacing=0.1,
                         density=1.0, open_boundary_layers=4, boundary_type=OutFlow(),
                         reference_pressure=pressure_func)

# 3D particles sampled as cylinder
circle = SphereShape(0.1, 0.5, (0.5, 0.5), 1.0, sphere_type=RoundSphere())

bidirectional_flow = BoundaryZone(; boundary_face=face_vertices, face_normal,
                                  particle_spacing=0.1, density=1.0,
                                  boundary_type=BidirectionalFlow(),
                                  extrude_geometry=circle, open_boundary_layers=4)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ BoundaryZone                                                                                     │
│ ════════════                                                                                     │
│ boundary type: ………………………………………… bidirectional_flow                                               │
│ #particles: ………………………………………………… 234                                                              │
│ width: ……………………………………………………………… 0.4                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

!!! warning "Experimental Implementation"
    This is an experimental feature and may change in any future releases.
"""
struct BoundaryZone{IC, S, ZO, ZW, FD, FN, ELTYPE, R}
    initial_condition :: IC
    spanning_set      :: S
    zone_origin       :: ZO
    zone_width        :: ZW
    flow_direction    :: FD
    face_normal       :: FN
    rest_pressure     :: ELTYPE # Only required for `BoundaryModelDynamicalPressureZhang`
    reference_values  :: R
    # Note that the following can't be static type parameters, as all boundary zones in a system
    # must have the same type, so that we can loop over them in a type-stable way.
    average_inflow_velocity :: Bool
    prescribed_density      :: Bool
    prescribed_pressure     :: Bool
    prescribed_velocity     :: Bool
end

function BoundaryZone(; boundary_face, face_normal, density, particle_spacing,
                      initial_condition=nothing, extrude_geometry=nothing,
                      open_boundary_layers::Integer, average_inflow_velocity=true,
                      boundary_type=BidirectionalFlow(),
                      rest_pressure=zero(eltype(density)),
                      reference_density=nothing, reference_pressure=nothing,
                      reference_velocity=nothing)
    if open_boundary_layers <= 0
        throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
    end

    # `face_normal` always points in fluid domain
    face_normal_ = normalize(SVector(face_normal...))

    ic, flow_direction, spanning_set_, zone_origin,
    zone_width = set_up_boundary_zone(boundary_face, face_normal_, density,
                                      particle_spacing, initial_condition, extrude_geometry,
                                      open_boundary_layers, boundary_type)

    NDIMS = ndims(ic)
    ELTYPE = eltype(ic)
    if !(reference_velocity isa Function || isnothing(reference_velocity) ||
         (reference_velocity isa Vector && length(reference_velocity) == NDIMS))
        throw(ArgumentError("`reference_velocity` must be either a function mapping " *
                            "each particle's coordinates and time to its velocity, " *
                            "or, for a constant fluid velocity, a vector of length $NDIMS for a $(NDIMS)D problem holding this velocity"))
    else
        if reference_velocity isa Function
            test_result = reference_velocity(zeros(NDIMS), 0.0)
            if length(test_result) != NDIMS
                throw(ArgumentError("`velocity` function must be of dimension $NDIMS"))
            end
        end
        # We need this dummy for type stability reasons
        velocity_dummy = SVector(ntuple(dim -> convert(ELTYPE, Inf), NDIMS))
        velocity_ref = wrap_reference_function(reference_velocity, velocity_dummy)
    end

    if !(reference_pressure isa Function || reference_pressure isa Real ||
         reference_pressure isa AbstractPressureModel || isnothing(reference_pressure))
        throw(ArgumentError("`reference_pressure` must be either a function mapping " *
                            "each particle's coordinates and time to its pressure, " *
                            "a scalar, or a pressure model"))
    else
        if reference_pressure isa Function
            test_result = reference_pressure(zeros(NDIMS), 0.0)
            if length(test_result) != 1
                throw(ArgumentError("`reference_pressure` function must be a scalar function"))
            end
            pressure_ref = reference_pressure
        elseif reference_pressure isa AbstractPressureModel
            pressure_ref = reference_pressure
            pressure_ref.pressure[] = rest_pressure
        else
            # We need this dummy for type stability reasons
            pressure_dummy = convert(ELTYPE, Inf)
            pressure_ref = wrap_reference_function(reference_pressure, pressure_dummy)
        end
    end

    if !(reference_density isa Function || reference_density isa Real ||
         isnothing(reference_density))
        throw(ArgumentError("`reference_density` must be either a function mapping " *
                            "each particle's coordinates and time to its density, " *
                            "or a scalar"))
    else
        if reference_density isa Function
            test_result = reference_density(zeros(NDIMS), 0.0)
            if length(test_result) != 1
                throw(ArgumentError("`reference_density` function must be a scalar function"))
            end
        end
        # We need this dummy for type stability reasons
        density_dummy = convert(ELTYPE, Inf)
        density_ref = wrap_reference_function(reference_density, density_dummy)
    end

    prescribed_pressure = isnothing(reference_pressure) ? false : true
    prescribed_density = isnothing(reference_density) ? false : true
    prescribed_velocity = isnothing(reference_velocity) ? false : true

    reference_values = (reference_velocity=velocity_ref, reference_pressure=pressure_ref,
                        reference_density=density_ref)

    coordinates_svector = reinterpret(reshape, SVector{NDIMS, eltype(ic.coordinates)},
                                      ic.coordinates)

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
                        flow_direction, face_normal_, rest_pressure, reference_values,
                        average_inflow_velocity, prescribed_density, prescribed_pressure,
                        prescribed_velocity)
end

function boundary_type_name(boundary_zone::BoundaryZone)
    (; flow_direction, face_normal) = boundary_zone

    if isnothing(flow_direction)
        return "bidirectional_flow"
    elseif signbit(dot(flow_direction, face_normal))
        return "outflow"
    else
        return "inflow"
    end
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
        summary_line(io, "width", round(boundary_zone.zone_width, digits=6))
        summary_footer(io)
    end
end

function set_up_boundary_zone(boundary_face, face_normal, density, particle_spacing,
                              initial_condition, extrude_geometry, open_boundary_layers,
                              boundary_type)
    if boundary_type isa InFlow
        # Unit vector pointing in downstream direction
        flow_direction = face_normal
    elseif boundary_type isa OutFlow
        # Unit vector pointing in downstream direction
        flow_direction = -face_normal
    elseif boundary_type isa BidirectionalFlow
        flow_direction = nothing
    end

    # Sample particles in boundary zone
    if isnothing(initial_condition) && isnothing(extrude_geometry)
        initial_condition = TrixiParticles.extrude_geometry(boundary_face; particle_spacing,
                                                            density,
                                                            direction=(-face_normal),
                                                            n_extrude=open_boundary_layers)
    elseif !isnothing(extrude_geometry)
        initial_condition = TrixiParticles.extrude_geometry(extrude_geometry;
                                                            particle_spacing,
                                                            density,
                                                            direction=(-face_normal),
                                                            n_extrude=open_boundary_layers)
    else
        initial_condition = initial_condition
    end

    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    zone_width = open_boundary_layers * initial_condition.particle_spacing

    # Vectors spanning the boundary zone/box
    spanning_set, zone_origin = calculate_spanning_vectors(boundary_face, zone_width)

    # First vector of `spanning_vectors` is normal to the boundary face.
    dot_face_normal = dot(normalize(spanning_set[:, 1]), face_normal)

    if !isapprox(abs(dot_face_normal), 1)
        throw(ArgumentError("`face_normal` is not normal to the boundary face"))
    end

    if boundary_type isa InFlow
        # First vector of `spanning_vectors` is normal to the boundary face
        dot_flow = dot(normalize(spanning_set[:, 1]), flow_direction)

        # The vector must point in upstream direction for an inflow boundary.
        # Flip the normal vector to point in the opposite direction of `flow_direction`.
        spanning_set[:, 1] .*= -sign(dot_flow)

    elseif boundary_type isa OutFlow
        # First vector of `spanning_vectors` is normal to the boundary face
        dot_flow = dot(normalize(spanning_set[:, 1]), flow_direction)

        # The vector must point in downstream direction for an outflow boundary.
        # Flip the normal vector to point in `flow_direction`.
        spanning_set[:, 1] .*= sign(dot_flow)

    elseif boundary_type isa BidirectionalFlow
        # Flip the normal vector to point opposite to fluid domain
        spanning_set[:, 1] .*= -sign(dot_face_normal)
    end

    spanning_set_ = reinterpret(reshape, SVector{NDIMS, eltype(spanning_set)}, spanning_set)

    # Remove particles outside the boundary zone.
    # This check is only necessary when `initial_condition` or `extrude_geometry` are passed.
    ic = remove_outside_particles(initial_condition, spanning_set_, zone_origin)

    return ic, flow_direction, spanning_set_, zone_origin, zone_width
end

function calculate_spanning_vectors(boundary_face, zone_width)
    return spanning_vectors(Tuple(boundary_face), zone_width), SVector(boundary_face[1]...)
end

function spanning_vectors(face_vertices::NTuple{2}, zone_width)
    face_size = face_vertices[2] - face_vertices[1]

    # Calculate normal vector of `boundary_face`
    b = normalize([-face_size[2], face_size[1]]) * zone_width

    return hcat(b, face_size)
end

function spanning_vectors(face_vertices::NTuple{3}, zone_width)
    # Vectors spanning the `boundary_face`
    edge1 = face_vertices[2] - face_vertices[1]
    edge2 = face_vertices[3] - face_vertices[1]

    # Check if the edges are linearly dependent (to avoid degenerate planes)
    if isapprox(norm(cross(edge1, edge2)), 0.0; atol=eps())
        throw(ArgumentError("the vectors `AB` and `AC` must not be collinear"))
    end

    # Calculate normal vector of `boundary_face`
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

    @threaded semi for particle in each_integrated_particle(system)
        particle_coords = current_coords(u, system, particle)

        for (zone_id, boundary_zone) in enumerate(boundary_zones)
            # Check if boundary particle is in the boundary zone
            if is_in_boundary_zone(boundary_zone, particle_coords)
                system.boundary_zone_indices[particle] = zone_id
            end
        end

        # Assert that every active buffer particle is assigned to a boundary zone.
        # This should always be true if the boundary zone geometry is set up correctly.
        # However, rare edge cases during particle conversion (`convert_particle!`)
        # may leave a particle unassigned. Potential causes for failure:
        # - `face_normal` is not exactly normal to the `boundary_face`
        #   (fixed in https://github.com/trixi-framework/TrixiParticles.jl/pull/926).
        # - Large downstream domain expansion can shift an inflow particle to the zone edge;
        #   even after upstream adjustment it may remain outside
        #   (fixed in https://github.com/trixi-framework/TrixiParticles.jl/pull/997).
        # - Floating-point rounding when a particle lies almost exactly on the `boundary_face`
        #   during transition, causing a reset just outside the zone
        #   (fixed in https://github.com/trixi-framework/TrixiParticles.jl/pull/997).
        @assert system.boundary_zone_indices[particle] != 0 "No boundary zone found for active buffer particle"
    end

    return system
end

function current_boundary_zone(system, particle)
    return system.boundary_zones[system.boundary_zone_indices[particle]]
end

function remove_outside_particles(initial_condition, spanning_set, zone_origin)
    (; coordinates, velocity, density, particle_spacing) = initial_condition

    in_zone = fill(true, nparticles(initial_condition))

    for particle in eachparticle(initial_condition)
        current_position = current_coords(coordinates, initial_condition, particle)
        particle_position = current_position - zone_origin

        in_zone[particle] = is_in_boundary_zone(spanning_set, particle_position)
    end

    return InitialCondition(; coordinates=coordinates[:, in_zone], density=first(density),
                            velocity=velocity[:, in_zone], particle_spacing)
end

function wrap_reference_function(function_::Nothing, ref_dummy)
    # Return a dummy value for type stability
    return @inline((coords, t)->ref_dummy)
end

function wrap_reference_function(function_::Function, ref_dummy)
    # Already a function
    return function_
end

function wrap_reference_function(constant_scalar::Number, ref_dummy)
    return @inline((coords, t)->constant_scalar)
end

function wrap_reference_function(constant_vector::AbstractVector,
                                 ref_dummy::SVector{NDIMS, ELTYPE}) where {NDIMS, ELTYPE}
    return @inline((coords, t)->SVector{NDIMS, ELTYPE}(constant_vector))
end

function reference_pressure(boundary_zone, v, system, particle, pos, t)
    (; prescribed_pressure) = boundary_zone
    (; pressure_reference_values) = system.cache

    if prescribed_pressure
        zone_id = system.boundary_zone_indices[particle]

        # `pressure_reference_values[zone_id](pos, t)`, but in a type-stable way
        return apply_ith_function(pressure_reference_values, zone_id, pos, t)
    else
        return current_pressure(v, system, particle)
    end
end

function reference_density(boundary_zone, v, system, particle, pos, t)
    (; prescribed_density) = boundary_zone
    (; density_reference_values) = system.cache

    if prescribed_density
        zone_id = system.boundary_zone_indices[particle]

        # `density_reference_values[zone_id](pos, t)`, but in a type-stable way
        return apply_ith_function(density_reference_values, zone_id, pos, t)
    else
        return current_density(v, system, particle)
    end
end

function reference_velocity(boundary_zone, v, system, particle, pos, t)
    (; prescribed_velocity) = boundary_zone
    (; velocity_reference_values) = system.cache

    if prescribed_velocity
        zone_id = system.boundary_zone_indices[particle]

        # `velocity_reference_values[zone_id](pos, t)`, but in a type-stable way
        return apply_ith_function(velocity_reference_values, zone_id, pos, t)
    else
        return current_velocity(v, system, particle)
    end
end
