@doc raw"""
    InitialCondition(coordinates, velocities, masses, densities; pressure=0.0,
                     particle_spacing=-1.0)

Struct to hold the initial configuration of the particles.

The following setups return `InitialCondition`s for commonly used setups:
- [`RectangularShape`](@ref)
- [`SphereShape`](@ref)
- [`RectangularTank`](@ref)

`InitialCondition`s support the set operations `union`, `setdiff` and `intersect` in order
to build more complex geometries.

# Arguments
- `coordinates`: An array where the $i$-th column holds the coordinates of particle $i$.
- `velocities`: An array where the $i$-th column holds the velocity of particle $i$.
- `masses`: A vector holding the mass of each particle.
- `densities`: A vector holding the density of each particle.

# Keywords
- `pressure`: Either a vector of pressure values of each particle or a number for a constant
              pressure over all particles. This is optional and only needed when using
              the [`EntropicallyDampedSPHSystem`](@ref).
- `particle_spacing`: The spacing between the particles. This is a number, as the spacing
                      is assumed to be uniform. This is only needed when using
                      set operations on the `InitialCondition`.

# Examples
```julia
# Rectangle filled with particles
initial_condition = RectangularShape(0.1, (3, 4), (-1.0, 1.0), 1.0)

# Two spheres in one initial condition
initial_condition = union(SphereShape(0.15, 0.5, (-1.0, 1.0), 1.0),
                          SphereShape(0.15, 0.2, (0.0, 1.0), 1.0))

# Rectangle with a spherical hole
shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())
initial_condition = setdiff(shape1, shape2)

# Intersect of a rectangle with a sphere. Note that this keeps the particles of the
# rectangle that are in the intersect, while `intersect(shape2, shape1)` would consist of
# the particles of the sphere that are in the intersect.
shape1 = RectangularShape(0.1, (16, 13), (-0.8, 0.0), 1.0)
shape2 = SphereShape(0.1, 0.35, (0.0, 0.6), 1.0, sphere_type=RoundSphere())
initial_condition = intersect(shape1, shape2)

# Build `InitialCondition` manually
coordinates = [0.0 1.0 1.0
               0.0 0.0 1.0]
velocities = zero(coordinates)
masses = ones(3)
densities = 1000 * ones(3)
initial_condition = InitialCondition(coordinates, velocities, masses, densities)
```
"""
struct InitialCondition{ELTYPE}
    particle_spacing :: ELTYPE
    coordinates      :: Array{ELTYPE, 2}
    velocity         :: Array{ELTYPE, 2}
    mass             :: Array{ELTYPE, 1}
    density          :: Array{ELTYPE, 1}
    pressure         :: Array{ELTYPE, 1}

    function InitialCondition(coordinates,
                              velocities::AbstractMatrix, masses::AbstractVector,
                              densities::AbstractVector;
                              pressure=0.0, particle_spacing=-1.0)
        ELTYPE = eltype(coordinates)

        if size(coordinates) != size(velocities)
            throw(ArgumentError("`coordinates` and `velocities` must be of the same size"))
        end

        if !(size(coordinates, 2) == length(masses) == length(densities))
            throw(ArgumentError("Expected: size(coordinates, 2) == length(masses) == length(densities)\n" *
                                "Got: size(coordinates, 2) = $(size(coordinates, 2)), " *
                                "length(masses) = $(length(masses)), " *
                                "length(densities) = $(length(densities))"))
        end

        if pressure isa Number
            pressure = pressure * ones(ELTYPE, length(masses))
        elseif length(pressure) != length(masses)
            throw(ArgumentError("`pressure` must either be a scalar or a vector of the " *
                                "same length as `masses`"))
        end

        return new{ELTYPE}(particle_spacing, coordinates, velocities, masses,
                           densities, pressure)
    end

    # Velocity, mass, density and pressure are given per particle either constant
    # or as functions of the coordinates.
    function InitialCondition(coordinates, velocity, mass, density;
                              pressure=0.0, particle_spacing=-1.0)
        NDIMS = size(coordinates, 1)

        return InitialCondition{NDIMS}(coordinates, velocity, mass, density,
                                       pressure, particle_spacing)
    end

    # Function barrier to make NDIMS static and therefore SVectors type-stable
    function InitialCondition{NDIMS}(coordinates, velocity, mass, density,
                                     pressure, particle_spacing) where {NDIMS}
        ELTYPE = eltype(coordinates)

        coordinates_svector = reinterpret(reshape, SVector{NDIMS, ELTYPE}, coordinates)
        velocity_fun = wrap_function(velocity, Val(NDIMS))
        mass_fun = wrap_function(mass, Val(NDIMS))
        density_fun = wrap_function(density, Val(NDIMS))
        pressure_fun = wrap_function(pressure, Val(NDIMS))

        if length(velocity_fun(coordinates_svector[1])) != NDIMS
            throw(ArgumentError("`velocity` must be $NDIMS-dimensional " *
                                "for $NDIMS-dimensional `coordinates`"))
        end

        velocities_svector = velocity_fun.(coordinates_svector)
        velocities = reinterpret(reshape, ELTYPE, velocities_svector)
        masses = mass_fun.(coordinates_svector)
        densities = density_fun.(coordinates_svector)
        pressures = pressure_fun.(coordinates_svector)

        return new{ELTYPE}(particle_spacing, coordinates, velocities, masses,
                           densities, pressures)
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

@inline function Base.eltype(initial_condition::InitialCondition)
    return eltype(initial_condition.coordinates)
end

@inline nparticles(initial_condition::InitialCondition) = length(initial_condition.mass)

function Base.union(initial_condition::InitialCondition, initial_conditions...)
    particle_spacing = initial_condition.particle_spacing
    ic = first(initial_conditions)

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

    result = InitialCondition(coordinates, velocity, mass, density, pressure=pressure,
                              particle_spacing=particle_spacing)

    return union(result, Base.tail(initial_conditions)...)
end

Base.union(initial_condition::InitialCondition) = initial_condition

function Base.setdiff(initial_condition::InitialCondition, initial_conditions...)
    ic = first(initial_conditions)

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

    result = InitialCondition(coordinates, velocity, mass, density, pressure=pressure,
                              particle_spacing=particle_spacing)

    return setdiff(result, Base.tail(initial_conditions)...)
end

Base.setdiff(initial_condition::InitialCondition) = initial_condition

function Base.intersect(initial_condition::InitialCondition, initial_conditions...)
    ic = first(initial_conditions)

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

    result = InitialCondition(coordinates, velocity, mass, density, pressure=pressure,
                              particle_spacing=particle_spacing)

    return intersect(result, Base.tail(initial_conditions)...)
end

Base.intersect(initial_condition::InitialCondition) = initial_condition

# Find particles in `coords1` that are closer than `max_distance` to any particle in `coords2`
function find_too_close_particles(coords1, coords2, max_distance)
    NDIMS = size(coords1, 1)
    result = Int[]

    nhs = GridNeighborhoodSearch{NDIMS}(max_distance, size(coords2, 2))
    TrixiParticles.initialize!(nhs, coords2)

    # We are modifying the vector `result`, so this cannot be parallel
    TrixiParticles.for_particle_neighbor(coords1, coords2, nhs,
                                         parallel=false) do particle, _, _, _
        if !(particle in result)
            append!(result, particle)
        end
    end

    return result
end
