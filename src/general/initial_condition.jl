struct InitialCondition{ELTYPE}
    coordinates :: Array{ELTYPE, 2}
    velocity    :: Array{ELTYPE, 2}
    mass        :: Array{ELTYPE, 1}
    density     :: Array{ELTYPE, 1}

    function InitialCondition(coordinates, velocities, masses, densities)
        if size(coordinates) != size(velocities)
            throw(ArgumentError("`coordinates` and `velocities` must be of the same size"))
        end

        if !(size(coordinates, 2) == length(masses) == length(densities))
            throw(ArgumentError("the following must hold: " *
                                "`size(coordinates, 2) == length(masses) == length(densities)`"))
        end

        return new{eltype(coordinates)}(coordinates, velocities, masses, densities)
    end

    function InitialCondition(initial_conditions...)
        NDIMS = size(first(initial_conditions).coordinates, 1)
        if any(ic -> size(ic.coordinates, 1) != NDIMS, initial_conditions)
            throw(ArgumentError("all passed initial conditions must have the same dimensionality"))
        end

        coordinates = hcat((ic.coordinates for ic in initial_conditions)...)
        velocity = hcat((ic.velocity for ic in initial_conditions)...)
        mass = vcat((ic.mass for ic in initial_conditions)...)
        density = vcat((ic.density for ic in initial_conditions)...)

        # TODO: Throw warning when particles are overlapping
        return new{eltype(coordinates)}(coordinates, velocity, mass, density)
    end
end

@inline nparticles(initial_condition::InitialCondition) = length(initial_condition.mass)
