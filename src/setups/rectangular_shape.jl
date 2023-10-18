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
- `tlsph`:          With the [TotalLagrangianSPHSystem](@ref), particles need to be placed
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
                          loop_order=:x_first)
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

    if acceleration === nothing && state_equation === nothing
        densities = density * ones(ELTYPE, n_particles)
    elseif (acceleration isa AbstractVector || acceleration isa Tuple) &&
           state_equation !== nothing
        # Initialize hydrostatic pressure
        pressure = Vector{ELTYPE}(undef, n_particles)
        initialize_pressure!(pressure, particle_spacing, SVector(acceleration),
                             state_equation, n_particles_per_dimension, loop_order)

        # Get density from inverse state equation
        densities = inverse_state_equation.(Ref(state_equation), pressure)
    else
        throw(ArgumentError("`acceleration` and `state_equation` must either be " *
                            "used both or both must be `nothing`"))
    end

    particle_volume = particle_spacing^NDIMS
    masses = particle_volume * densities

    return InitialCondition(coordinates, velocities, masses, densities, pressure=pressure,
                            particle_spacing=particle_spacing)
end

function rectangular_shape_coords(particle_spacing, n_particles_per_dimension,
                                  min_coordinates; tlsph=false, loop_order=:x_first)
    ELTYPE = eltype(particle_spacing)
    NDIMS = length(n_particles_per_dimension)

    coordinates = Array{ELTYPE, 2}(undef, NDIMS, prod(n_particles_per_dimension))

    if tlsph
        min_coordinates = min_coordinates .- 0.5particle_spacing
    end

    initialize_rectangular!(coordinates, particle_spacing, min_coordinates,
                            n_particles_per_dimension, loop_order)

    return coordinates
end

# 2D
function initialize_rectangular!(coordinates, particle_spacing,
                                 min_coordinates,
                                 n_particles_per_dimension::NTuple{2}, loop_order)
    n_particles_x, n_particles_y = n_particles_per_dimension
    particle = 0

    if loop_order === :x_first
        for x in 1:n_particles_x, y in 1:n_particles_y
            particle += 1
            fill_coordinates!(coordinates, particle, min_coordinates, x, y,
                              particle_spacing)
        end

    elseif loop_order === :y_first
        for y in 1:n_particles_y, x in 1:n_particles_x
            particle += 1
            fill_coordinates!(coordinates, particle, min_coordinates, x, y,
                              particle_spacing)
        end

    else
        throw(ArgumentError("$loop_order is not a valid loop order. Possible values are :x_first and :y_first."))
    end
end

# 3D
function initialize_rectangular!(coordinates, particle_spacing,
                                 min_coordinates,
                                 n_particles_per_dimension::NTuple{3}, loop_order)
    n_particles_x, n_particles_y, n_particles_z = n_particles_per_dimension
    particle = 0

    if loop_order === :x_first
        for x in 1:n_particles_x, y in 1:n_particles_y, z in 1:n_particles_z
            particle += 1
            fill_coordinates!(coordinates, particle, min_coordinates, x, y, z,
                              particle_spacing)
        end

    elseif loop_order === :y_first
        for y in 1:n_particles_y, x in 1:n_particles_x, z in 1:n_particles_z
            particle += 1
            fill_coordinates!(coordinates, particle, min_coordinates, x, y, z,
                              particle_spacing)
        end

    elseif loop_order === :z_first
        for z in 1:n_particles_z, y in 1:n_particles_y, x in 1:n_particles_x
            particle += 1
            fill_coordinates!(coordinates, particle, min_coordinates, x, y, z,
                              particle_spacing)
        end

    else
        throw(ArgumentError("$loop_order is not a valid loop order. Possible values are :x_first, :y_first and :z_first"))
    end
end

@inline function fill_coordinates!(coordinates, particle,
                                   min_coordinates, x, y, particle_spacing)
    # The first particle starts at a distance `0.5particle_spacing` from `min_coordinates`
    # in each dimension.
    coordinates[1, particle] = min_coordinates[1] + (x - 0.5) * particle_spacing
    coordinates[2, particle] = min_coordinates[2] + (y - 0.5) * particle_spacing
end

@inline function fill_coordinates!(coordinates, particle,
                                   min_coordinates, x, y, z, particle_spacing)
    # The first particle starts at a distance `0.5particle_spacing` from `min_coordinates`
    # in each dimension.
    coordinates[1, particle] = min_coordinates[1] + (x - 0.5) * particle_spacing
    coordinates[2, particle] = min_coordinates[2] + (y - 0.5) * particle_spacing
    coordinates[3, particle] = min_coordinates[3] + (z - 0.5) * particle_spacing
end

# 2D
function initialize_pressure!(pressure, particle_spacing, acceleration, state_equation,
                              n_particles_per_dimension::NTuple{2}, loop_order)
    n_particles_x, n_particles_y = n_particles_per_dimension

    @inline density(pressure_) = inverse_state_equation(state_equation, pressure_)

    if loop_order !== :x_first
        throw(ArgumentError("hydrostatic pressure calculation is only supported with loop" *
                            "order `:x_first`"))
    end

    # Start with the highest index and loop backwards in order to start at the fluid surface
    particle = prod(n_particles_per_dimension)

    # The hydrostatic pressure is given by the ODE `dp/dr = rho * g`, where `r` is the
    # distance to the surface (water depth), `rho` is the density and `g` is the
    # acceleration.
    # For high water columns (and especially for higher compressibility), this yields
    # better results than just assuming `rho` to be constant.
    #
    # To solve this ODE, we use the explicit Euler method in each dimension
    # independently, matching the particle indexing.
    # This allows diagonal accelerations as well (albeit only on negative coordinate
    # directions).
    pressure_x = 0.0
    pressure_x -= 0.5particle_spacing * acceleration[1] * density(pressure_x)
    for x in n_particles_x:-1:1
        # For the integration in y-direction, start at `pressure_x`
        pressure_y = pressure_x
        pressure_y -= 0.5particle_spacing * acceleration[2] * density(pressure_y)
        for y in n_particles_y:-1:1
            pressure[particle] = pressure_y
            particle -= 1

            # Explicit Euler step
            pressure_y -= particle_spacing * acceleration[2] * density(pressure_y)
        end

        # Explicit Euler step
        pressure_x -= particle_spacing * acceleration[1] * density(pressure_x)
    end
end

# 3D
function initialize_pressure!(pressure, particle_spacing, acceleration, state_equation,
                              n_particles_per_dimension::NTuple{3}, loop_order)
    n_particles_x, n_particles_y, n_particles_z = n_particles_per_dimension

    @inline density(pressure_) = inverse_state_equation(state_equation, pressure_)

    if loop_order !== :x_first
        throw(ArgumentError("hydrostatic pressure calculation is only supported with loop" *
                            "order `:x_first`"))
    end

    # Start with the highest index and loop backwards in order to start at the fluid surface
    particle = prod(n_particles_per_dimension)

    pressure_x = 0.0
    for x in n_particles_x:-1:1
        pressure_y = pressure_x
        for y in n_particles_y:-1:1
            pressure_z = pressure_y
            for z in n_particles_z:-1:1
                pressure[particle] = pressure_z
                particle -= 1

                pressure_z -= particle_spacing * acceleration[3] * density(pressure_z)
            end

            pressure_y -= particle_spacing * acceleration[2] * density(pressure_y)
        end

        pressure_x -= particle_spacing * acceleration[1] * density(pressure_x)
    end
end
