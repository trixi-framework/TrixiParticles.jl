"""
    RectangularShape(particle_spacing, n_particles_per_dimension, min_coordinates;
                     velocity=zeros(length(n_particles_per_dimension)),
                     mass=nothing, density=nothing, pressure=0.0,
                     acceleration=nothing, state_equation=nothing,
                     place_on_shell=false, coordinates_eltype=Float64,
                     coordinates_perturbation=nothing)

Rectangular shape filled with particles. Returns an [`InitialCondition`](@ref).

# Arguments
- `particle_spacing`:           Spacing between the particles. The type of this argument
                                determines the eltype of the initial condition.
- `n_particles_per_dimension`:  Tuple containing the number of particles in x, y and z
                                (only 3D) direction, respectively.
- `min_coordinates`:            Coordinates of the corner in negative coordinate directions.

# Keywords
- `velocity`:       Either a function mapping each particle's coordinates to its velocity,
                    or, for a constant fluid velocity, a vector holding this velocity.
                    Velocity is constant zero by default.
- `mass`:           By default, automatically compute particle mass from particle
                    density and spacing. Can also be a function mapping each particle's
                    coordinates to its mass, or a scalar for a constant mass over all particles.
- `density`:        Either a function mapping each particle's coordinates to its density,
                    or a scalar for a constant density over all particles.
                    Obligatory when not using a state equation. Cannot be used together with
                    `state_equation`.
- `pressure`:       Scalar to set the pressure of all particles to this value.
                    This is only used by the [`EntropicallyDampedSPHSystem`](@ref) and
                    will be overwritten when using an initial pressure function in the system.
                    Cannot be used together with hydrostatic pressure gradient.
- `acceleration`:   In order to initialize particles with a hydrostatic pressure gradient,
                    an acceleration vector can be passed. Note that only accelerations
                    in one coordinate direction and no diagonal accelerations are supported.
                    This will only change the pressure of the particles. When using the
                    [`WeaklyCompressibleSPHSystem`](@ref), pass a `state_equation` as well
                    to initialize the particles with the corresponding density and mass.
                    When using the [`EntropicallyDampedSPHSystem`](@ref), the pressure
                    will be overwritten when using an initial pressure function in the system.
                    This cannot be used together with the `pressure` keyword argument.
- `state_equation`: When calculating a hydrostatic pressure gradient by setting `acceleration`,
                    the `state_equation` will be used to set the corresponding density.
                    Cannot be used together with `density`.
- `place_on_shell = false`: If `place_on_shell=true`, particles will be placed on the shell
                    of the shape. For example, the [`TotalLagrangianSPHSystem`](@ref)
                    requires particles to be placed on the shell of the shape and
                    not half a particle spacing away, as for fluids.
- `coordinates_eltype = Float64`: Eltype of the particle coordinates.
                    See [the docs on GPU support](@ref gpu_support) for more information.
- `coordinates_perturbation`: Add a small random displacement to the particle positions,
                    where the amplitude is `coordinates_perturbation * particle_spacing`.

# Examples
```jldoctest; output = false, setup = :(particle_spacing = 0.1)
# 2D
rectangular = RectangularShape(particle_spacing, (5, 4), (1.0, 2.0), density=1000.0)

# 2D with hydrostatic pressure gradient.
# `state_equation` has to be the same as for the WCSPH system.
state_equation = StateEquationCole(sound_speed=20.0, exponent=7, reference_density=1000.0)
rectangular = RectangularShape(particle_spacing, (5, 4), (1.0, 2.0),
                               acceleration=(0.0, -9.81), state_equation=state_equation)

# 3D
rectangular = RectangularShape(particle_spacing, (5, 4, 7), (1.0, 2.0, 3.0), density=1000.0)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ InitialCondition                                                                                 │
│ ════════════════                                                                                 │
│ #dimensions: ……………………………………………… 3                                                                │
│ #particles: ………………………………………………… 140                                                              │
│ particle spacing: ………………………………… 0.1                                                              │
│ eltype: …………………………………………………………… Float64                                                          │
│ coordinate eltype: ……………………………… Float64                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
function RectangularShape(particle_spacing, n_particles_per_dimension, min_coordinates;
                          velocity=zeros(length(n_particles_per_dimension)),
                          mass=nothing, density=nothing, pressure=0.0,
                          acceleration=nothing, state_equation=nothing,
                          place_on_shell=false, coordinates_eltype=Float64,
                          loop_order=nothing, coordinates_perturbation=nothing)
    if particle_spacing < eps()
        throw(ArgumentError("`particle_spacing` needs to be positive and larger than $(eps())"))
    end

    NDIMS = length(n_particles_per_dimension)

    if length(min_coordinates) != NDIMS
        throw(ArgumentError("`min_coordinates` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    if density !== nothing && any(density .< eps())
        throw(ArgumentError("`density` needs to be positive and larger than $(eps())"))
    end

    ELTYPE = eltype(particle_spacing)
    n_particles = prod(n_particles_per_dimension)

    # The type of the particle spacing determines the eltype of the coordinates
    coordinates = rectangular_shape_coords(convert(coordinates_eltype, particle_spacing),
                                           n_particles_per_dimension,
                                           min_coordinates, place_on_shell=place_on_shell,
                                           loop_order=loop_order)

    if !isnothing(coordinates_perturbation)
        seed!(1)
        amplitude = coordinates_perturbation * particle_spacing
        coordinates .+= rand((-amplitude):(particle_spacing * 1e-3):(amplitude),
                             NDIMS, n_particles)
    end

    # Allow zero acceleration with state equation, but interpret `nothing` acceleration
    # with state equation as a likely mistake.
    if acceleration isa AbstractVector || acceleration isa Tuple
        if pressure != 0.0
            throw(ArgumentError("`pressure` cannot be used together with `acceleration` " *
                                "and `state_equation` (hydrostatic pressure gradient)"))
        end

        if state_equation === nothing
            density_fun = pressure -> density
        else
            if density !== nothing
                throw(ArgumentError("`density` cannot be used together with `acceleration` " *
                                    "and `state_equation` (hydrostatic pressure gradient)"))
            end
            density_fun = pressure -> inverse_state_equation(state_equation, pressure)
        end

        # Initialize hydrostatic pressure
        pressure = Vector{ELTYPE}(undef, n_particles)
        initialize_pressure!(pressure, particle_spacing, acceleration,
                             density_fun, n_particles_per_dimension, loop_order)

        if state_equation !== nothing
            # Weakly compressible case: get density from inverse state equation
            density = inverse_state_equation.(Ref(state_equation), pressure)
        end
    elseif acceleration !== nothing
        throw(ArgumentError("`acceleration` must either be `nothing` or a vector/tuple"))
    elseif state_equation !== nothing
        throw(ArgumentError("`state_equation` must be used together with `acceleration`"))
    end

    if density === nothing
        throw(ArgumentError("`density` must be specified when not using `acceleration` " *
                            "and `state_equation` (hydrostatic pressure gradient)"))
    end

    return InitialCondition(; coordinates, velocity, density, mass, pressure,
                            particle_spacing)
end

# 1D
function loop_permutation(loop_order, NDIMS::Val{1})
    if loop_order === :x_first || loop_order === nothing
        permutation = (1,)
    else
        throw(ArgumentError("$loop_order is not a valid loop order. " *
                            "Possible values are :x_first."))
    end

    return permutation
end

# 2D
function loop_permutation(loop_order, NDIMS::Val{2})
    if loop_order === :y_first || loop_order === nothing
        permutation = (1, 2)
    elseif loop_order === :x_first
        permutation = (2, 1)
    else
        throw(ArgumentError("$loop_order is not a valid loop order. " *
                            "Possible values are :x_first and :y_first."))
    end

    return permutation
end

# 3D
function loop_permutation(loop_order, NDIMS::Val{3})
    if loop_order === :z_first || loop_order === nothing
        permutation = (1, 2, 3)
    elseif loop_order === :y_first
        permutation = (2, 1, 3)
    elseif loop_order === :x_first
        permutation = (3, 2, 1)
    else
        throw(ArgumentError("$loop_order is not a valid loop order. " *
                            "Possible values are :x_first, :y_first and :z_first"))
    end

    return permutation
end

function rectangular_shape_coords(particle_spacing, n_particles_per_dimension,
                                  min_coordinates; place_on_shell=false, loop_order=nothing)
    ELTYPE = eltype(particle_spacing)
    NDIMS = length(n_particles_per_dimension)

    coordinates = Array{ELTYPE, 2}(undef, NDIMS, prod(n_particles_per_dimension))

    # With place_on_shell, particles need to be AT the min coordinates and not half a particle
    # spacing away from it.
    if place_on_shell
        min_coordinates = min_coordinates .- 0.5particle_spacing
    end

    permutation = loop_permutation(loop_order, Val(NDIMS))
    cartesian_indices = CartesianIndices(n_particles_per_dimension)
    permuted_indices = permutedims(cartesian_indices, permutation)

    for particle in eachindex(permuted_indices)
        index = Tuple(permuted_indices[particle])

        # The first particle starts at a distance `0.5particle_spacing` from
        # `min_coordinates` in each dimension.
        coordinates[:, particle] .= min_coordinates .+ particle_spacing .* (index .- 0.5)
    end

    return coordinates
end

function initialize_pressure!(pressure, particle_spacing, acceleration, density_fun,
                              n_particles_per_dimension, loop_order)
    if count(a -> abs(a) > eps(), acceleration) > 1
        throw(ArgumentError("hydrostatic pressure calculation is not supported with " *
                            "diagonal acceleration"))
    end

    # Dimension in which the acceleration is acting
    accel_dim = findfirst(a -> abs(a) > eps(), acceleration)

    # Compute 1D pressure gradient with explicit Euler method
    factor = particle_spacing * abs(acceleration[accel_dim])

    pressure_1d = zeros(n_particles_per_dimension[accel_dim])

    # The first particle is half a particle spacing from the surface, so start with a
    # half step.
    pressure_1d[1] = 0.5factor * density_fun(0.0)

    for i in 1:(length(pressure_1d) - 1)
        # Explicit Euler step
        pressure_1d[i + 1] = pressure_1d[i] + factor * density_fun(pressure_1d[i])
    end

    # If acceleration is pointing in negative coordinate direction, reverse the pressure
    # gradient, because the surface is at the top and the gradient should start from there.
    if sign(acceleration[accel_dim]) < 0
        reverse!(pressure_1d)
    end

    # Loop over all particles and access 1D pressure gradient.
    # Apply permutation depending on loop order to match indexing of the coordinates.
    cartesian_indices = CartesianIndices(n_particles_per_dimension)
    permutation = loop_permutation(loop_order, Val(length(n_particles_per_dimension)))
    permuted_indices = permutedims(cartesian_indices, permutation)
    for particle in eachindex(pressure)
        # The index in the dimension where the acceleration is acting to index 1D pressure
        # vector.
        index_in_accel_dim = permuted_indices[particle][accel_dim]
        pressure[particle] = pressure_1d[index_in_accel_dim]
    end
end
