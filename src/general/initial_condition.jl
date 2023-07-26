struct InitialCondition{ELTYPE, B}
    coordinates :: Array{ELTYPE, 2}
    velocity    :: Array{ELTYPE, 2}
    mass        :: Array{ELTYPE, 1}
    density     :: Array{ELTYPE, 1}
    pressure    :: Array{ELTYPE, 1}
    buffer      :: B

    function InitialCondition(coordinates, velocities, masses, densities; pressure=0.0,
                              buffer=nothing)
        if size(coordinates) != size(velocities)
            throw(ArgumentError("`coordinates` and `velocities` must be of the same size"))
        end

        if !(size(coordinates, 2) == length(masses) == length(densities))
            throw(ArgumentError("the following must hold: " *
                                "`size(coordinates, 2) == length(masses) == length(densities)`"))
        end

        (buffer ≠ nothing) && (buffer = SystemBuffer(size(coordinates, 2), buffer))

        if pressure isa Number
            pressure = pressure * ones(length(masses))
        elseif length(pressure) != length(masses)
            throw(ArgumentError("`pressure` must either be a scalar or a vector of the " *
                                "same length as `masses`"))
        end

        coordinates, velocities, masses, densities, pressure = allocate_buffer(coordinates,
                                                                               velocities,
                                                                               masses,
                                                                               densities,
                                                                               pressure,
                                                                               buffer)

        return new{eltype(masses),
                   typeof(buffer)}(coordinates, velocities, masses, densities, pressure,
                                   buffer)
    end

    function InitialCondition(initial_conditions...; buffer=nothing)
        NDIMS = size(first(initial_conditions).coordinates, 1)
        if any(ic -> size(ic.coordinates, 1) != NDIMS, initial_conditions)
            throw(ArgumentError("all passed initial conditions must have the same dimensionality"))
        end

        if any(ic -> ic.buffer ≠ nothing, initial_conditions)
            throw(ArgumentError("You have passed `buffer` before. Please pass `buffer` only here."))
        end

        coordinates = hcat((ic.coordinates for ic in initial_conditions)...)
        velocity = hcat((ic.velocity for ic in initial_conditions)...)
        mass = vcat((ic.mass for ic in initial_conditions)...)
        density = vcat((ic.density for ic in initial_conditions)...)
        pressure = vcat((ic.pressure for ic in initial_conditions)...)

        (buffer ≠ nothing) && (buffer = SystemBuffer(size(coordinates, 2), buffer))

        coordinates, velocity, mass, density, pressure = allocate_buffer(coordinates,
                                                                         velocity,
                                                                         mass, density,
                                                                         pressure, buffer)

        # TODO: Throw warning when particles are overlapping
        return new{eltype(coordinates),
                   typeof(buffer)}(coordinates, velocity, mass, density, pressure, buffer)
    end
end

@inline function Base.ndims(initial_condition::InitialCondition)
    return size(initial_condition.coordinates, 1)
end

@inline function Base.eltype(initial_condition::InitialCondition)
    return eltype(initial_condition.coordinates)
end

@inline nparticles(initial_condition::InitialCondition) = length(initial_condition.mass)
