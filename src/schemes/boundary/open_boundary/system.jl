struct InFlow end

struct OutFlow end

struct OpenBoundarySPHSystem{BZ, NDIMS, ELTYPE <: Real, S, VF} <: FluidSystem{NDIMS}
    initial_condition        :: InitialCondition{ELTYPE}
    mass                     :: Array{ELTYPE, 1} # [particle]
    density                  :: Array{ELTYPE, 1} # [particle]
    volume                   :: Array{ELTYPE, 1} # [particle]
    pressure                 :: Array{ELTYPE, 1} # [particle]
    characteristics          :: Array{ELTYPE, 2} # [characteristics, particle]
    previous_characteristics :: Array{ELTYPE, 2} # [characteristics, particle]
    sound_speed              :: ELTYPE
    boundary_zone            :: BZ
    flow_direction           :: SVector{NDIMS, ELTYPE}
    zone_origin              :: SVector{NDIMS, ELTYPE}
    spanning_set             :: S
    velocity_function        :: VF

    function OpenBoundarySPHSystem(plane_points, boundary_zone, sound_speed;
                                   sample_geometry=plane_points, particle_spacing,
                                   flow_direction, open_boundary_layers=0, density,
                                   velocity=zeros(length(plane_points)),
                                   pressure=0.0, velocity_function=nothing, mass=nothing)
        if !((boundary_zone isa InFlow) || (boundary_zone isa OutFlow))
            throw(ArgumentError("`boundary_zone` must either be of type InFlow or OutFlow"))
        end

        if !(open_boundary_layers isa Int)
            throw(ArgumentError("`open_boundary_layers` must be of type Int"))
        elseif open_boundary_layers < sqrt(eps())
            throw(ArgumentError("`open_boundary_layers` must be positive and greater than zero"))
        end

        # Unit vector pointing in downstream direction.
        flow_direction_ = normalize(SVector(flow_direction...))

        # Sample boundary zone with particles in corresponding direction.
        (boundary_zone isa OutFlow) && (direction = flow_direction_)
        (boundary_zone isa InFlow) && (direction = -flow_direction_)

        # Sample particles in boundary zone.
        initial_condition = ExtrudeGeometry(sample_geometry; particle_spacing, direction,
                                            n_extrude=open_boundary_layers, velocity, mass,
                                            density, pressure)

        zone_width = open_boundary_layers * initial_condition.particle_spacing

        NDIMS = ndims(initial_condition)
        ELTYPE = eltype(initial_condition)

        # Vectors spanning the boundary zone/box.
        spanning_set = spanning_vectors(plane_points, zone_width)

        # First vector of `spanning_vectors` is normal to the in-/outflow plane.
        # The normal vector must point in downstream direction for an outflow boundary and
        # for an inflow boundary the normal vector must point in upstream direction.
        # Thus, rotate the normal vector correspondingly.
        if isapprox(dot(normalize(spanning_set[:, 1]), flow_direction_), 1.0, atol=1e-7)
            # Normal vector points in downstream direction.
            # Flip the inflow vector in upstream direction
            (boundary_zone isa InFlow) && (spanning_set[:, 1] .*= -1)
        elseif isapprox(dot(normalize(spanning_set[:, 1]), flow_direction_), -1.0,
                        atol=1e-7)
            # Normal vector points in upstream direction.
            # Flip the outflow vector in downstream direction
            (boundary_zone isa OutFlow) && (spanning_set[:, 1] .*= -1)
        else
            throw(ArgumentError("flow direction and normal vector of " *
                                "$(typeof(boundary_zone))-plane do not correspond"))
        end

        spanning_set_ = reinterpret(reshape, SVector{NDIMS, ELTYPE}, spanning_set)

        zone_origin = SVector(plane_points[1]...)

        mass = copy(initial_condition.mass)
        pressure = copy(initial_condition.pressure)
        density = copy(initial_condition.density)
        volume = similar(initial_condition.density)

        characteristics = zeros(ELTYPE, 3, length(mass))
        previous_characteristics = zeros(ELTYPE, 4, length(mass))

        return new{typeof(boundary_zone), NDIMS, ELTYPE, typeof(spanning_set_),
                   typeof(velocity_function)}(initial_condition, mass, density, volume,
                                              pressure, characteristics,
                                              previous_characteristics, sound_speed,
                                              boundary_zone, flow_direction_, zone_origin,
                                              spanning_set_, velocity_function)
    end
end

timer_name(::OpenBoundarySPHSystem) = "open_boundary"

function Base.show(io::IO, system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    print(io, "OpenBoundarySPHSystem{", ndims(system), "}(")
    print(io, system.boundary_zone)
    print(io, ") with ", nparticles(system), " particles")
end

function Base.show(io::IO, ::MIME"text/plain", system::OpenBoundarySPHSystem)
    @nospecialize system # reduce precompilation time

    if get(io, :compact, false)
        show(io, system)
    else
        summary_header(io, "OpenBoundarySPHSystem{$(ndims(system))}")
        summary_line(io, "#particles", nparticles(system))
        summary_line(io, "boundary", system.boundary_zone)
        summary_line(io, "flow direction", system.flow_direction)
        summary_line(io, "width", round(norm(system.spanning_set[1]), digits=3))
        summary_footer(io)
    end
end

@inline viscosity_model(system, neighbor_system::OpenBoundarySPHSystem) = system.viscosity

@inline source_terms(system::OpenBoundarySPHSystem) = nothing

@inline function particle_density(v, system::OpenBoundarySPHSystem, particle)
    return system.density[particle]
end

@inline function particle_pressure(v, system::OpenBoundarySPHSystem, particle)
    return system.pressure[particle]
end

@inline set_particle_density(particle, v, system::OpenBoundarySPHSystem, density) = system

@inline set_particle_pressure(particle, v, system, pressure) = system

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

@inline function within_boundary_zone(particle_coords, system)
    (; zone_origin, spanning_set) = system
    particle_positon = particle_coords - zone_origin

    for dim in 1:ndims(system)
        span_dim = spanning_set[dim]
        # Checks whether the projection of the particle position
        # falls within the range of the zone.
        if !(0 <= dot(particle_positon, span_dim) <= dot(span_dim, span_dim))

            # Particle is not in boundary zone.
            return false
        end
    end

    # Particle is in boundary zone.
    return true
end
