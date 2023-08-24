struct InitialCondition{ELTYPE}
    particle_spacing :: ELTYPE
    coordinates      :: Array{ELTYPE, 2}
    velocity         :: Array{ELTYPE, 2}
    mass             :: Array{ELTYPE, 1}
    density          :: Array{ELTYPE, 1}
    pressure         :: Array{ELTYPE, 1}

    function InitialCondition(coordinates, velocities, masses, densities; pressure=0.0,
                              particle_spacing=-1.0)
        if size(coordinates) != size(velocities)
            throw(ArgumentError("`coordinates` and `velocities` must be of the same size"))
        end

        if !(size(coordinates, 2) == length(masses) == length(densities))
            throw(ArgumentError("the following must hold: " *
                                "`size(coordinates, 2) == length(masses) == length(densities)`"))
        end

        if pressure isa Number
            pressure = pressure * ones(length(masses))
        elseif length(pressure) != length(masses)
            throw(ArgumentError("`pressure` must either be a scalar or a vector of the " *
                                "same length as `masses`"))
        end

        return new{eltype(coordinates)}(particle_spacing, coordinates, velocities, masses,
                                        densities, pressure)
    end
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

    if !isa(initial_condition.pressure, Number)
        pressure = vcat(initial_condition.pressure, ic.pressure[valid_particles])
    end

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
