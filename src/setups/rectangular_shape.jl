"""
    RectangularShape(particle_spacing, n_particles_per_dimension, min_coordinates, density;
                     tlsph=false,
                     init_velocity=ntuple(_ -> 0.0, length(n_particles_per_dimension)),
                     loop_order=:x_first,)

Rectangular shape filled with particles. Returns an [`InitialCondition`](@ref).

# Arguments
- `particle_spacing`:                   Spacing between the particles.
- `n_particles_per_dimension::Tuple`:   Tuple containing the number of particles in x, y and z (only 3D) direction, respectively.
- `min_coordinates::Tuple`:             Coordinates of the corner in negative coordinate directions.
- `density`:                            Initial density of particles.

# Keywords
- `tlsph`:          With the [`TotalLagrangianSPHSystem`](@ref), particles need to be placed
                    on the boundary of the shape and not one particle radius away, as for fluids.
                    When `tlsph=true`, particles will be placed on the boundary of the shape.
- `init_velocity`:  The initial velocity of the fluid particles as `(vel_x, vel_y)` (or `(vel_x, vel_y, vel_z)` in 3D).
- `loop_order`:     To enforce a specific particle indexing by reordering the indexing loop
                    (possible values: `:x_first`, `:y_first`, `:z_first`).

# Examples
2D:
```julia
rectangular = RectangularShape(particle_spacing, (5, 4), (1.0, 2.0), 1000.0)
```
3D:
```julia
rectangular = RectangularShape(particle_spacing, (5, 4, 7), (1.0, 2.0, 3.0), 1000.0)
```
"""
function RectangularShape(particle_spacing, n_particles_per_dimension,
                          min_coordinates, density; pressure=0.0, tlsph=false,
                          init_velocity=ntuple(_ -> 0.0, length(n_particles_per_dimension)),
                          acceleration=nothing, state_equation=nothing,
                          loop_order=nothing)
    if particle_spacing < eps()
        throw(ArgumentError("`particle_spacing` needs to be positive and larger than $(eps())"))
    end

    NDIMS = length(n_particles_per_dimension)

    if length(min_coordinates) != NDIMS
        throw(ArgumentError("`min_coordinates` must be of length $NDIMS for a $(NDIMS)D problem"))
    end

    if density < eps()
        throw(ArgumentError("`density` needs to be positive and larger than $(eps())"))
    end

    ELTYPE = eltype(particle_spacing)

    n_particles = prod(n_particles_per_dimension)

    coordinates = rectangular_shape_coords(particle_spacing, n_particles_per_dimension,
                                           min_coordinates, tlsph=tlsph,
                                           loop_order=loop_order)
    velocities = init_velocity .* ones(ELTYPE, size(coordinates))

    # Allow zero acceleration with state equation, but interpret `nothing` acceleration
    # with state equation as a likely mistake.
    if (acceleration === nothing && state_equation === nothing) ||
       all(abs.(acceleration) .< eps())
        densities = density * ones(ELTYPE, n_particles)
    elseif acceleration isa AbstractVector || acceleration isa Tuple
        if state_equation === nothing
            density_fun = pressure -> density
        else
            density_fun = pressure -> inverse_state_equation(state_equation, pressure)
        end

        # Initialize hydrostatic pressure
        pressure = Vector{ELTYPE}(undef, n_particles)
        initialize_pressure!(pressure, particle_spacing, acceleration,
                             density_fun, n_particles_per_dimension, loop_order)

        if state_equation === nothing
            # Incompressible case
            densities = density * ones(ELTYPE, n_particles)
        else
            # Weakly compressible case: get density from inverse state equation
            densities = inverse_state_equation.(Ref(state_equation), pressure)
        end
    else
        throw(ArgumentError("`acceleration` must either be `nothing` or a vector/tuple"))
    end

    particle_volume = particle_spacing^NDIMS
    masses = particle_volume * densities

    return InitialCondition(coordinates, velocities, masses, densities, pressure=pressure,
                            particle_spacing=particle_spacing)
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
                                  min_coordinates; tlsph=false, loop_order=nothing)
    ELTYPE = eltype(particle_spacing)
    NDIMS = length(n_particles_per_dimension)

    coordinates = Array{ELTYPE, 2}(undef, NDIMS, prod(n_particles_per_dimension))

    # With TLSPH, particles need to be AT the min coordinates and not half a particle
    # spacing away from it.
    if tlsph
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
    factor = particle_spacing * acceleration[accel_dim]

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
