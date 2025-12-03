@doc raw"""
    InitialCondition(; coordinates, density, velocity=zeros(size(coordinates, 1)),
                     mass=nothing, pressure=0.0, particle_spacing=-1.0)

Struct to hold the initial configuration of the particles.

The following setups return `InitialCondition`s for commonly used setups:
- [`RectangularShape`](@ref)
- [`SphereShape`](@ref)
- [`RectangularTank`](@ref)
- [`ComplexShape`](@ref)
- [`extrude_geometry`](@ref)

`InitialCondition`s support the set operations `union`, `setdiff` and `intersect` in order
to build more complex geometries.
`InitialCondition`s also support the set operations `setdiff` and `intersect` together
with `TrixiParticles.TriangleMesh` and `TrixiParticles.Polygon` returned by [`load_geometry`](@ref).

# Arguments
- `coordinates`: An array where the $i$-th column holds the coordinates of particle $i$.
- `density`:     Either a vector holding the density of each particle,
                 or a function mapping each particle's coordinates to its density,
                 or a scalar for a constant density over all particles.

# Keywords
- `velocity`:   Either an array where the $i$-th column holds the velocity of particle $i$,
                or a function mapping each particle's coordinates to its velocity,
                or, for a constant fluid velocity, a vector holding this velocity.
                Velocity is constant zero by default.
- `mass`:       Either `nothing` (default) to automatically compute particle mass from particle
                density and spacing, or a vector holding the mass of each particle,
                or a function mapping each particle's coordinates to its mass,
                or a scalar for a constant mass over all particles.
- `pressure`:   Either a vector holding the pressure of each particle,
                or a function mapping each particle's coordinates to its pressure,
                or a scalar for a constant pressure over all particles. This is optional and
                only needed when using the [`EntropicallyDampedSPHSystem`](@ref).
- `particle_spacing`: The spacing between the particles. This is a scalar, as the spacing
                      is assumed to be uniform. This is only needed when using
                      set operations on the `InitialCondition` or for automatic mass calculation.

# Examples
```jldoctest; output = false
# Rectangle filled with particles
initial_condition = RectangularShape(0.1, (3, 4), (-1.0, 1.0), density=1.0)

# Two spheres in one initial condition
initial_condition = union(SphereShape(0.15, 0.5, (-1.0, 1.0), 1.0),
                          SphereShape(0.15, 0.2, (0.0, 1.0), 1.0))

# Rectangle with a spherical hole
shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0)
shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())
initial_condition = setdiff(shape1, shape2)

# Intersect of a rectangle with a sphere. Note that this keeps the particles of the
# rectangle that are in the intersect, while `intersect(shape2, shape1)` would consist of
# the particles of the sphere that are in the intersect.
shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), density=1.0)
shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())
initial_condition = intersect(shape1, shape2)

# Set operations with geometries loaded from files
shape = RectangularShape(0.05, (20, 20), (0.0, 0.0), density=1.0)
file =  pkgdir(TrixiParticles, "examples", "preprocessing", "data", "circle.asc")
geometry = load_geometry(file)

initial_condition_1 = intersect(shape, geometry)
initial_condition_2 = setdiff(shape, geometry)

# Build `InitialCondition` manually
coordinates = [0.0 1.0 1.0
               0.0 0.0 1.0]
velocity = zero(coordinates)
mass = ones(3)
density = 1000 * ones(3)
initial_condition = InitialCondition(; coordinates, velocity, mass, density)

# With functions
initial_condition = InitialCondition(; coordinates, velocity=x -> 2x, mass=1.0, density=1000.0)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ InitialCondition                                                                                 │
│ ════════════════                                                                                 │
│ #dimensions: ……………………………………………… 2                                                                │
│ #particles: ………………………………………………… 3                                                                │
│ particle spacing: ………………………………… -1.0                                                             │
│ eltype: …………………………………………………………… Float64                                                          │
│ coordinate eltype: ……………………………… Float64                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct InitialCondition{ELTYPE, C, MATRIX, VECTOR}
    particle_spacing :: ELTYPE
    coordinates      :: C      # Array{coordinates_eltype, 2}
    velocity         :: MATRIX # Array{ELTYPE, 2}
    mass             :: VECTOR # Array{ELTYPE, 1}
    density          :: VECTOR # Array{ELTYPE, 1}
    pressure         :: VECTOR # Array{ELTYPE, 1}
end

# The default constructor needs to be accessible for Adapt.jl to work with this struct.
# See the comments in general/gpu.jl for more details.
function InitialCondition(; coordinates, density, velocity=zeros(size(coordinates, 1)),
                          mass=nothing, pressure=0.0, particle_spacing=-1.0)
    NDIMS = size(coordinates, 1)

    return InitialCondition{NDIMS}(coordinates, velocity, mass, density,
                                   pressure, particle_spacing)
end

# Function barrier to make `NDIMS` static and therefore SVectors type-stable
function InitialCondition{NDIMS}(coordinates, velocity, mass, density,
                                 pressure, particle_spacing) where {NDIMS}
    if !(particle_spacing isa AbstractFloat)
        throw(ArgumentError("particle spacing must be a floating point number. " *
                            "The type of the particle spacing determines the eltype " *
                            "of the `InitialCondition`."))
    end

    # The arguments can all be functions, so use `particle_spacing` for the eltype
    ELTYPE = typeof(particle_spacing)
    n_particles = size(coordinates, 2)

    if n_particles == 0
        return InitialCondition(particle_spacing, coordinates, zeros(ELTYPE, NDIMS, 0),
                                zeros(ELTYPE, 0), zeros(ELTYPE, 0), zeros(ELTYPE, 0))
    end

    # SVector of coordinates to pass to functions.
    # This will return a vector of SVectors in 2D and 3D, but an 1×n matrix in 1D.
    coordinates_svector_ = reinterpret(reshape, SVector{NDIMS, eltype(coordinates)},
                                       coordinates)
    # In 1D, this will reshape the 1×n matrix to a vector, in 2D/3D it will do nothing
    coordinates_svector = reshape(coordinates_svector_, length(coordinates_svector_))

    if velocity isa AbstractMatrix
        velocities = velocity
    else
        # Assuming `velocity` is a scalar or a function
        velocity_fun = wrap_function(velocity, Val(NDIMS))
        if length(velocity_fun(coordinates_svector[1])) != NDIMS
            throw(ArgumentError("`velocity` must be $NDIMS-dimensional " *
                                "for $NDIMS-dimensional `coordinates`"))
        end
        velocities_svector = velocity_fun.(coordinates_svector)
        velocities = stack(velocities_svector)
    end
    if size(coordinates) != size(velocities)
        throw(ArgumentError("`coordinates` and `velocities` must be of the same size"))
    end

    if density isa AbstractVector
        if length(density) != n_particles
            throw(ArgumentError("Expected: length(density) == size(coordinates, 2)\n" *
                                "Got: size(coordinates, 2) = $(size(coordinates, 2)), " *
                                "length(density) = $(length(density))"))
        end
        densities = density
    else
        density_fun = wrap_function(density, Val(NDIMS))
        densities = density_fun.(coordinates_svector)
    end

    if any(densities .< eps())
        throw(ArgumentError("density must be positive and larger than `eps()`"))
    end

    if pressure isa AbstractVector
        if length(pressure) != n_particles
            throw(ArgumentError("Expected: length(pressure) == size(coordinates, 2)\n" *
                                "Got: size(coordinates, 2) = $(size(coordinates, 2)), " *
                                "length(pressure) = $(length(pressure))"))
        end
        pressures = pressure
    else
        pressure_fun = wrap_function(pressure, Val(NDIMS))
        pressures = pressure_fun.(coordinates_svector)
    end

    if mass isa AbstractVector
        if length(mass) != n_particles
            throw(ArgumentError("Expected: length(mass) == size(coordinates, 2)\n" *
                                "Got: size(coordinates, 2) = $(size(coordinates, 2)), " *
                                "length(mass) = $(length(mass))"))
        end
        masses = mass
    elseif mass === nothing
        if particle_spacing < 0
            throw(ArgumentError("`mass` must be specified when not using `particle_spacing`"))
        end
        particle_volume = particle_spacing^NDIMS
        masses = particle_volume * densities
    else
        mass_fun = wrap_function(mass, Val(NDIMS))
        masses = mass_fun.(coordinates_svector)
    end

    return InitialCondition(particle_spacing, coordinates, ELTYPE.(velocities),
                            ELTYPE.(masses), ELTYPE.(densities), ELTYPE.(pressures))
end

function Base.show(io::IO, ic::InitialCondition)
    @nospecialize ic # reduce precompilation time

    print(io, "InitialCondition{$(eltype(ic)), $(coordinates_eltype(ic))}()")
end

function Base.show(io::IO, ::MIME"text/plain", ic::InitialCondition)
    @nospecialize ic # reduce precompilation time

    if get(io, :compact, false)
        show(io, ic)
    else
        summary_header(io, "InitialCondition")
        summary_line(io, "#dimensions", "$(ndims(ic))")
        summary_line(io, "#particles", "$(nparticles(ic))")
        summary_line(io, "particle spacing", "$(first(ic.particle_spacing))")
        summary_line(io, "eltype", "$(eltype(ic))")
        summary_line(io, "coordinate eltype", "$(coordinates_eltype(ic))")
        summary_footer(io)
    end
end

function wrap_function(function_::Function, ::Val)
    # Already a function
    return function_
end

function wrap_function(constant_scalar::Number, ::Val)
    return coords -> constant_scalar
end

# For vectors and tuples
function wrap_function(constant_vector, ::Val{NDIMS}) where {NDIMS}
    return coords -> SVector{NDIMS}(constant_vector)
end

@inline function Base.ndims(initial_condition::InitialCondition)
    return size(initial_condition.coordinates, 1)
end

@inline Base.eltype(::InitialCondition{ELTYPE}) where {ELTYPE} = ELTYPE

@inline function coordinates_eltype(initial_condition::InitialCondition)
    return eltype(initial_condition.coordinates)
end

@inline nparticles(initial_condition::InitialCondition) = length(initial_condition.mass)

function Base.union(initial_condition::InitialCondition, initial_conditions...)
    particle_spacing = initial_condition.particle_spacing
    ic = first(initial_conditions)

    if initial_condition.particle_spacing isa Vector || ic.particle_spacing isa Vector
        throw(ArgumentError("`union` operation does not support non-uniform particle spacings. " *
                            "Please ensure all `InitialCondition`s use a scalar particle spacing."))
    end

    if ndims(ic) != ndims(initial_condition)
        throw(ArgumentError("all passed initial conditions must have the same dimensionality"))
    end

    if particle_spacing < eps()
        throw(ArgumentError("all passed initial conditions must store a particle spacing"))
    end

    if !isapprox(ic.particle_spacing, particle_spacing)
        throw(ArgumentError("all passed initial conditions must have the same particle spacing"))
    end

    too_close = find_too_close_particles(ic.coordinates, initial_condition.coordinates,
                                         0.75particle_spacing)
    valid_particles = setdiff(eachparticle(ic), too_close)

    coordinates = hcat(initial_condition.coordinates, ic.coordinates[:, valid_particles])
    velocity = hcat(initial_condition.velocity, ic.velocity[:, valid_particles])
    mass = vcat(initial_condition.mass, ic.mass[valid_particles])
    density = vcat(initial_condition.density, ic.density[valid_particles])
    pressure = vcat(initial_condition.pressure, ic.pressure[valid_particles])

    result = InitialCondition{ndims(ic)}(coordinates, velocity, mass, density, pressure,
                                         particle_spacing)

    return union(result, Base.tail(initial_conditions)...)
end

Base.union(initial_condition::InitialCondition) = initial_condition

function Base.setdiff(initial_condition::InitialCondition, initial_conditions...)
    ic = first(initial_conditions)

    if initial_condition.particle_spacing isa Vector || ic.particle_spacing isa Vector
        throw(ArgumentError("`setdiff` operation does not support non-uniform particle spacings. " *
                            "Please ensure all `InitialCondition`s use a scalar particle spacing."))
    end

    if ndims(ic) != ndims(initial_condition)
        throw(ArgumentError("all passed initial conditions must have the same dimensionality"))
    end

    particle_spacing = initial_condition.particle_spacing
    if particle_spacing < eps()
        throw(ArgumentError("the initial condition in the first argument must store a particle spacing"))
    end

    too_close = find_too_close_particles(initial_condition.coordinates, ic.coordinates,
                                         0.75particle_spacing)
    valid_particles = setdiff(eachparticle(initial_condition), too_close)

    coordinates = initial_condition.coordinates[:, valid_particles]
    velocity = initial_condition.velocity[:, valid_particles]
    mass = initial_condition.mass[valid_particles]
    density = initial_condition.density[valid_particles]
    pressure = initial_condition.pressure[valid_particles]

    result = InitialCondition{ndims(ic)}(coordinates, velocity, mass, density, pressure,
                                         particle_spacing)

    return setdiff(result, Base.tail(initial_conditions)...)
end

Base.setdiff(initial_condition::InitialCondition) = initial_condition

function Base.intersect(initial_condition::InitialCondition, initial_conditions...)
    ic = first(initial_conditions)

    if initial_condition.particle_spacing isa Vector || ic.particle_spacing isa Vector
        throw(ArgumentError("`intersect` operation does not support non-uniform particle spacings. " *
                            "Please ensure all `InitialCondition`s use a scalar particle spacing."))
    end

    if ndims(ic) != ndims(initial_condition)
        throw(ArgumentError("all passed initial conditions must have the same dimensionality"))
    end

    particle_spacing = initial_condition.particle_spacing
    if particle_spacing < eps()
        throw(ArgumentError("the initial condition in the first argument must store a particle spacing"))
    end

    too_close = find_too_close_particles(initial_condition.coordinates, ic.coordinates,
                                         0.75particle_spacing)

    coordinates = initial_condition.coordinates[:, too_close]
    velocity = initial_condition.velocity[:, too_close]
    mass = initial_condition.mass[too_close]
    density = initial_condition.density[too_close]
    pressure = initial_condition.pressure[too_close]

    result = InitialCondition{ndims(ic)}(coordinates, velocity, mass, density, pressure,
                                         particle_spacing)

    return intersect(result, Base.tail(initial_conditions)...)
end

Base.intersect(initial_condition::InitialCondition) = initial_condition

function InitialCondition(sol::ODESolution, system, semi; use_final_velocity=false,
                          min_particle_distance=(system.initial_condition.particle_spacing /
                                                 4))
    ic = system.initial_condition

    v_ode, u_ode = sol.u[end].x

    u = wrap_u(u_ode, system, semi)
    v = wrap_u(v_ode, system, semi)

    # Check if particles come too close especially when the surface exhibits large curvature
    too_close = find_too_close_particles(u, min_particle_distance)

    velocity_ = use_final_velocity ? view(v, 1:ndims(system), :) : ic.velocity

    not_too_close = setdiff(eachparticle(system), too_close)

    coordinates = u[:, not_too_close]
    velocity = velocity_[:, not_too_close]
    mass = ic.mass[not_too_close]
    density = ic.density[not_too_close]
    pressure = ic.pressure[not_too_close]

    if length(too_close) > 0
        @info "Removed $(length(too_close)) particles that are too close together"
    end

    return InitialCondition{ndims(ic)}(coordinates, velocity, mass, density, pressure,
                                       ic.particle_spacing)
end

# Find particles in `coords1` that are closer than `max_distance` to any particle in `coords2`
function find_too_close_particles(coords1, coords2, max_distance)
    NDIMS = size(coords1, 1)
    result = Int[]

    nhs = GridNeighborhoodSearch{NDIMS}(search_radius=max_distance,
                                        n_points=size(coords2, 2))
    PointNeighbors.initialize!(nhs, coords1, coords2)

    # We are modifying the vector `result`, so this cannot be parallel
    foreach_point_neighbor(coords1, coords2, nhs;
                           parallelization_backend=SerialBackend()) do particle, _, _, _
        if !(particle in result)
            push!(result, particle)
        end
    end

    return result
end

# Find particles in `coords` that are closer than `min_distance` to any other particle in `coords`
function find_too_close_particles(coords, min_distance)
    NDIMS = size(coords, 1)
    result = Int[]

    nhs = GridNeighborhoodSearch{NDIMS}(search_radius=min_distance,
                                        n_points=size(coords, 2))
    TrixiParticles.initialize!(nhs, coords)

    # We are modifying the vector `result`, so this cannot be parallel
    foreach_point_neighbor(coords, coords, nhs;
                           parallelization_backend=SerialBackend()) do particle, neighbor,
                                                                       _, _
        # Only consider particles with neighbors that are not to be removed
        if particle != neighbor && !(particle in result) && !(neighbor in result)
            push!(result, particle)
        end
    end

    return result
end

function move_particles_to_end!(ic::InitialCondition, particle_ids_to_move)
    invalid_id = findfirst(i -> i <= 0 || i > nparticles(ic), particle_ids_to_move)
    isnothing(invalid_id) || throw(BoundsError(ic, invalid_id))

    sort_key = [i in particle_ids_to_move ? 1 : 0 for i in eachparticle(ic)]
    # Determine a permutation that sorts 'sort_key' in ascending order
    permutation = sortperm(sort_key)

    ic.coordinates .= ic.coordinates[:, permutation]
    ic.velocity .= ic.velocity[:, permutation]
    ic.mass .= ic.mass[permutation]
    ic.density .= ic.density[permutation]
    ic.pressure .= ic.pressure[permutation]

    return ic
end

function move_particles_to_end!(a::AbstractVector, particle_ids_to_move)
    invalid_id = findfirst(i -> i <= 0 || i > length(a), particle_ids_to_move)
    isnothing(invalid_id) || throw(BoundsError(a, invalid_id))

    sort_key = [i in particle_ids_to_move ? 1 : 0 for i in eachindex(a)]
    # determine a permutation that sorts 'sort_key' in ascending order
    permutation = sortperm(sort_key)

    a .= a[permutation]

    return a
end

move_particles_to_end!(a::Real, particle_ids_to_move) = a
