"""
    RectangularShape(particle_spacing, n_particles_per_dimension::NTuple{2},
                     particle_postion; density=0.0, loop_order=:x_first)

    RectangularShape(particle_spacing, n_particles_per_dimension::NTuple{3},
                     particle_postion; density=0.0, loop_order=:x_first)

Rectangular shape filled with particles.

# Arguments
- `particle_spacing`:             Spacing between the particles
- `n_particles_per_dimension`:    Tupel for number of particles in x, y and z (for 3D) direction, respectively
- `particle_postion`:    Tupel for starting point of the reactangular in x, y and z (for 3D) direction, respectively

# Keywords
- `density=0.0`: Specify the density if the `densities` or `masses` fields will be used
- `loop_order`: For a desired indexing of the particles (possible symbolic variables: `:x_first`, `:y_first`, `:z_first`)

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
struct RectangularShape{NDIMS, ELTYPE <: Real}
    coordinates               :: Array{ELTYPE, 2}
    masses                    :: Vector{ELTYPE}
    densities                 :: Vector{ELTYPE}
    particle_spacing          :: ELTYPE
    n_particles_per_dimension :: NTuple{NDIMS, Int}

    function RectangularShape(particle_spacing,
                              n_particles_per_dimension::NTuple{2}, particle_postion;
                              density=zero(eltype(particle_spacing)), loop_order=:x_first)
        NDIMS = 2
        if length(particle_postion) != NDIMS
            error("`particle_postion` must be of length $NDIMS for a $(NDIMS)D problem")
        end

        ELTYPE = eltype(particle_spacing)

        n_particles_x = n_particles_per_dimension[1]
        n_particles_y = n_particles_per_dimension[2]

        n_particles = prod(n_particles_per_dimension)

        x_position = particle_postion[1]
        y_position = particle_postion[2]

        coordinates = Array{Float64, 2}(undef, 2, n_particles)

        # Leave `densities` and `masses` empty if no `density` has been provided
        densities = density * ones(ELTYPE, n_particles * (density > 0))
        masses = density * particle_spacing^2 * ones(ELTYPE, n_particles * (density > 0))

        initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                n_particles_x, n_particles_y, loop_order)

        return new{NDIMS, ELTYPE}(coordinates, masses, densities,
                                  particle_spacing, n_particles_per_dimension)
    end

    function RectangularShape(particle_spacing,
                              n_particles_per_dimension::NTuple{3}, particle_postion;
                              density=zero(eltype(particle_spacing)), loop_order=:x_first)
        NDIMS = 3
        if length(particle_postion) != NDIMS
            error("`particle_postion` must be of length $NDIMS for a $(NDIMS)D problem")
        end

        ELTYPE = eltype(particle_spacing)

        n_particles_x = n_particles_per_dimension[1]
        n_particles_y = n_particles_per_dimension[2]
        n_particles_z = n_particles_per_dimension[3]

        n_particles = prod(n_particles_per_dimension)

        x_position = particle_postion[1]
        y_position = particle_postion[2]
        z_position = particle_postion[3]

        coordinates = Array{Float64, 2}(undef, 3, n_particles)

        # Leave `densities` and `masses` empty if no `density` has been provided
        densities = density * ones(ELTYPE, n_particles * (density > 0))
        masses = density * particle_spacing^3 * ones(ELTYPE, n_particles * (density > 0))

        initialize_rectangular!(coordinates, x_position, y_position, z_position,
                                particle_spacing, n_particles_x, n_particles_y,
                                n_particles_z, loop_order)

        return new{NDIMS, ELTYPE}(coordinates, masses, densities,
                                  particle_spacing, n_particles_per_dimension)
    end
end

# 2D
function initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                 n_particles_x, n_particles_y, loop_order)
    boundary_particle = 0
    if loop_order == :x_first
        for x in 1:n_particles_x, y in 1:n_particles_y
            boundary_particle += 1
            fill_coordinates!(coordinates, boundary_particle, x_position, y_position, x, y,
                              particle_spacing)
        end
    elseif loop_order == :y_first
        for y in 1:n_particles_y, x in 1:n_particles_x
            boundary_particle += 1
            fill_coordinates!(coordinates, boundary_particle, x_position, y_position, x, y,
                              particle_spacing)
        end
    else
        throw(ArgumentError("$loop_order is not a valid loop order. Possible values are :x_first and :y_first."))
    end
end

# 3D
function initialize_rectangular!(coordinates, x_position, y_position, z_position,
                                 particle_spacing, n_particles_x, n_particles_y,
                                 n_particles_z, loop_order)
    boundary_particle = 0

    if loop_order == :x_first
        for x in 1:n_particles_x, y in 1:n_particles_y, z in 1:n_particles_z
            boundary_particle += 1
            fill_coordinates!(coordinates, boundary_particle, x_position, y_position,
                              z_position, x, y, z, particle_spacing)
        end
    elseif loop_order == :y_first
        for y in 1:n_particles_y, x in 1:n_particles_x, z in 1:n_particles_z
            boundary_particle += 1
            fill_coordinates!(coordinates, boundary_particle, x_position, y_position,
                              z_position, x, y, z, particle_spacing)
        end
    elseif loop_order == :z_first
        for z in 1:n_particles_z, y in 1:n_particles_y, x in 1:n_particles_x
            boundary_particle += 1
            fill_coordinates!(coordinates, boundary_particle, x_position, y_position,
                              z_position, x, y, z, particle_spacing)
        end
    else
        throw(ArgumentError("$loop_order is not a valid loop order. Possible values are :x_first, :y_first and :z_first"))
    end
end

@inline function fill_coordinates!(coordinates, boundary_particle, x_position, y_position,
                                   x, y, particle_spacing)
    coordinates[1, boundary_particle] = x_position + (x - 1) * particle_spacing
    coordinates[2, boundary_particle] = y_position + (y - 1) * particle_spacing
end

@inline function fill_coordinates!(coordinates, boundary_particle, x_position, y_position,
                                   z_position, x, y, z, particle_spacing)
    coordinates[1, boundary_particle] = x_position + (x - 1) * particle_spacing
    coordinates[2, boundary_particle] = y_position + (y - 1) * particle_spacing
    coordinates[3, boundary_particle] = z_position + (z - 1) * particle_spacing
end
