struct InitialCondition{ELTYPE, B}
    coordinates :: Array{ELTYPE, 2}
    velocity    :: Array{ELTYPE, 2}
    mass        :: Array{ELTYPE, 1}
    density     :: Array{ELTYPE, 1}
    buffer      :: B

    function InitialCondition(coordinates, velocities, masses, densities; buffer=nothing)
        if size(coordinates) != size(velocities)
            throw(ArgumentError("`coordinates` and `velocities` must be of the same size"))
        end

        if !(size(coordinates, 2) == length(masses) == length(densities))
            throw(ArgumentError("the following must hold: " *
                                "`size(coordinates, 2) == length(masses) == length(densities)`"))
        end

        (buffer ≠ nothing) && (buffer isa Int) &&
            (buffer = SystemBuffer(size(coordinates, 2), buffer))

        coordinates, velocities, masses, densities = allocate_buffer(coordinates,
                                                                     velocities,
                                                                     masses, densities,
                                                                     buffer)

        return new{eltype(coordinates),
                   typeof(buffer)}(coordinates, velocities, masses, densities, buffer)
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

        (buffer ≠ nothing) && (buffer isa Int) &&
            (buffer = SystemBuffer(size(coordinates, 2), buffer))

        coordinates, velocity, mass, density = allocate_buffer(coordinates, velocity,
                                                               mass, density, buffer)

        # TODO: Throw warning when particles are overlapping
        return new{eltype(coordinates),
                   typeof(buffer)}(coordinates, velocity, mass, density, buffer)
    end
end

@inline function Base.ndims(initial_condition::InitialCondition)
    return size(initial_condition.coordinates, 1)
end

@inline function Base.eltype(initial_condition::InitialCondition)
    return eltype(initial_condition.coordinates)
end

@inline nparticles(initial_condition::InitialCondition) = length(initial_condition.mass)
