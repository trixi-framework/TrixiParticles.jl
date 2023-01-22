"""
    RectangularShape(particle_spacing, n_particles_x, n_particles_y,
                     x_position, y_position; density=0.0)

Rectangular shape filled with particles.

# Arguments
- `particle_spacing`:                   Spacing betweeen the particles
- `n_particles_x`, `n_particles_y`:     Number of particles in x and y direction, respectively

# Keywords
- `density=0.0`: Specify the density if the `densities` or `masses` fields will be used
- `x_position`, `y_position`:           Starting point of the reactangular in x and y direction, respectively

# Fields
- `coordinates::Matrix`: Coordinates of the particles
- `masses::Vector`: Masses of the particles
- `densities::Vector`: Densities of the particles

# Example
```julia
rectangular = RectangularShape(particle_spacing,
                               round(Int, rectangular_width/particle_spacing),
                               round(Int, rectangular_height/particle_spacing),
                               0.0, 0.0)
```
"""
struct RectangularShape{NDIMS, ELTYPE <: Real}
    coordinates               :: Array{ELTYPE, 2}
    masses                    :: Vector{ELTYPE}
    densities                 :: Vector{ELTYPE}
    particle_spacing          :: ELTYPE
    n_particles_per_dimension :: NTuple{NDIMS, Int}

    function RectangularShape(particle_spacing, n_particles_x, n_particles_y;
                              x_position=zero(eltype(particle_spacing)),
                              y_position=zero(eltype(particle_spacing)),
                              density=0.0, loop_order=:x_first)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        n_particles = n_particles_y * n_particles_x

        coordinates = Array{Float64, 2}(undef, 2, n_particles)

        densities = density * ones(ELTYPE, n_particles)
        masses = density * particle_spacing^2 * ones(ELTYPE, n_particles)

        initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                n_particles_x, n_particles_y, loop_order)

        n_particles_per_dimension = (n_particles_x, n_particles_y)

        return new{NDIMS, ELTYPE}(coordinates, masses, densities,
                                  particle_spacing, n_particles_per_dimension)
    end

    function RectangularShape(particle_spacing, n_particles_x, n_particles_y, n_particles_z;
                              x_position=zero(eltype(particle_spacing)),
                              y_position=zero(eltype(particle_spacing)),
                              z_position=zero(eltype(particle_spacing)),
                              density=0.0, loop_order=:x_first)
        NDIMS = 3
        ELTYPE = eltype(particle_spacing)

        n_particles = n_particles_y * n_particles_x * n_particles_z

        coordinates = Array{Float64, 2}(undef, 3, n_particles)

        densities = density * ones(ELTYPE, n_particles)
        masses = density * particle_spacing^3 * ones(ELTYPE, n_particles)

        initialize_rectangular!(coordinates, x_position, y_position, z_position,
                                particle_spacing, n_particles_x, n_particles_y,
                                n_particles_z, loop_order)

        n_particles_per_dimension = (n_particles_x, n_particles_y, n_particles_z)

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
