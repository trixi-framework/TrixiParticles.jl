"""
    RectangularShape(particle_spacing, n_particles_per_dimension::NTuple{2},
                     particle_position; density=0.0, loop_order=:x_first)

    RectangularShape(particle_spacing, n_particles_per_dimension::NTuple{3},
                     particle_position; density=0.0, loop_order=:x_first)

Rectangular shape filled with particles.

# Arguments
- `particle_spacing`:             Spacing between the particles
- `n_particles_per_dimension::Tuple`: Tuple containing the number of particles in x, y and z (only 3D) direction, respectively
- `particle_position::Tuple`:    Coordinates of the corner in negative coordinate directions

# Keywords
- `density=0.0`: Specify the density if the `densities` or `masses` fields will be used
- `loop_order`: To enforce a specific particle indexing by reordering the indexing loop (possible values: `:x_first`, `:y_first`, `:z_first`)

# Fields
- `coordinates::Matrix`: Coordinates of the particles
- `masses::Vector`: Masses of the particles
- `densities::Vector`: Densities of the particles

# Examples
2D:
```julia
rectangular = RectangularShape(particle_spacing, (5, 4), (1.0, 2.0))
```
3D:
```julia
rectangular = RectangularShape(particle_spacing, (5, 4, 7), (1.0, 2.0, 3.0))
```
"""
struct RectangularShape{NDIM, ELTYPE <: Real}
    coordinates               :: Array{ELTYPE, 2}
    velocities                :: Array{ELTYPE, 2}
    masses                    :: Vector{ELTYPE}
    densities                 :: Vector{ELTYPE}
    particle_spacing          :: ELTYPE
    n_particles_per_dimension :: NTuple{NDIM, Int}

    function RectangularShape(particle_spacing,
                              n_particles_per_dimension::NTuple{2}, particle_position;
                              density=zero(eltype(particle_spacing)), loop_order=:x_first)
        NDIM = 2
        if length(particle_position) != NDIM
            throw(ArgumentError("`particle_position` must be of length $NDIM for a $(NDIM)D problem"))
        end

        ELTYPE = eltype(particle_spacing)

        n_particles = prod(n_particles_per_dimension)

        coordinates = Array{Float64, 2}(undef, 2, n_particles)
        velocities = zeros(ELTYPE, 2, n_particles)

        # Leave `densities` and `masses` empty if no `density` has been provided
        densities = density * ones(ELTYPE, n_particles * (density > 0))
        masses = density * particle_spacing^2 * ones(ELTYPE, n_particles * (density > 0))

        initialize_rectangular!(coordinates, particle_spacing, particle_position,
                                n_particles_per_dimension, loop_order)

        return new{NDIM, ELTYPE}(coordinates, velocities, masses, densities,
                                 particle_spacing, n_particles_per_dimension)
    end

    function RectangularShape(particle_spacing,
                              n_particles_per_dimension::NTuple{3}, particle_position;
                              density=zero(eltype(particle_spacing)), loop_order=:x_first)
        NDIM = 3
        if length(particle_position) != NDIM
            throw(ArgumentError("`particle_position` must be of length $NDIM for a $(NDIM)D problem"))
        end

        ELTYPE = eltype(particle_spacing)

        n_particles = prod(n_particles_per_dimension)

        coordinates = Array{Float64, 2}(undef, 3, n_particles)
        velocities = zeros(ELTYPE, 3, n_particles)

        # Leave `densities` and `masses` empty if no `density` has been provided
        densities = density * ones(ELTYPE, n_particles * (density > 0))
        masses = density * particle_spacing^3 * ones(ELTYPE, n_particles * (density > 0))

        initialize_rectangular!(coordinates, particle_spacing, particle_position,
                                n_particles_per_dimension, loop_order)

        return new{NDIM, ELTYPE}(coordinates, velocities, masses, densities,
                                 particle_spacing, n_particles_per_dimension)
    end
end

# 2D
function initialize_rectangular!(coordinates, particle_spacing,
                                 particle_position::NTuple{2},
                                 n_particles_per_dimension::NTuple{2}, loop_order)
    n_particles_x, n_particles_y = n_particles_per_dimension
    particle = 0

    if loop_order === :x_first
        for x in 1:n_particles_x, y in 1:n_particles_y
            particle += 1
            fill_coordinates!(coordinates, particle, particle_position, x, y,
                              particle_spacing)
        end

    elseif loop_order === :y_first
        for y in 1:n_particles_y, x in 1:n_particles_x
            particle += 1
            fill_coordinates!(coordinates, particle, particle_position, x, y,
                              particle_spacing)
        end

    else
        throw(ArgumentError("$loop_order is not a valid loop order. Possible values are :x_first and :y_first."))
    end
end

# 3D
function initialize_rectangular!(coordinates, particle_spacing,
                                 particle_position::NTuple{3},
                                 n_particles_per_dimension::NTuple{3}, loop_order)
    n_particles_x, n_particles_y, n_particles_z = n_particles_per_dimension
    particle = 0

    if loop_order === :x_first
        for x in 1:n_particles_x, y in 1:n_particles_y, z in 1:n_particles_z
            particle += 1
            fill_coordinates!(coordinates, particle, particle_position, x, y, z,
                              particle_spacing)
        end

    elseif loop_order === :y_first
        for y in 1:n_particles_y, x in 1:n_particles_x, z in 1:n_particles_z
            particle += 1
            fill_coordinates!(coordinates, particle, particle_position, x, y, z,
                              particle_spacing)
        end

    elseif loop_order === :z_first
        for z in 1:n_particles_z, y in 1:n_particles_y, x in 1:n_particles_x
            particle += 1
            fill_coordinates!(coordinates, particle, particle_position, x, y, z,
                              particle_spacing)
        end

    else
        throw(ArgumentError("$loop_order is not a valid loop order. Possible values are :x_first, :y_first and :z_first"))
    end
end

@inline function fill_coordinates!(coordinates, particle,
                                   particle_position::NTuple{2}, x, y, particle_spacing)
    coordinates[1, particle] = particle_position[1] + (x - 1) * particle_spacing
    coordinates[2, particle] = particle_position[2] + (y - 1) * particle_spacing
end

@inline function fill_coordinates!(coordinates, particle,
                                   particle_position::NTuple{3}, x, y, z, particle_spacing)
    coordinates[1, particle] = particle_position[1] + (x - 1) * particle_spacing
    coordinates[2, particle] = particle_position[2] + (y - 1) * particle_spacing
    coordinates[3, particle] = particle_position[3] + (z - 1) * particle_spacing
end
