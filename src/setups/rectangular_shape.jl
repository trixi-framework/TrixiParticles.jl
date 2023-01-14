"""
    RectangularShape(particle_spacing, n_particles_x, n_particles_y,
                     x_position, y_position; density=0.0)

Rectangular shape filled with particles.

# Arguments
- `particle_spacing`:                   Spacing betweeen the particles
- `n_particles_x`, `n_particles_y`:     Number of particles in x and y direction, respectively
- `x_position`, `y_position`:           Starting point of the reactangular in x and y direction, respectively

# Keywords
- `density=0.0`: Specify the density if the `densities` or `masses` fields will be used

# Fields
- `coordinates::Array{ELTYPE, 2}`: Array containing the coordinates of the particles
- `masses::Vector{ELTYPE}`: Vector containing the masses of the particles
- `densities::Vector{ELTYPE}`: Vector containing the densities of the particles

# Example
```julia
rectangular = RectangularShape(particle_spacing,
                               round(Int, rectangular_width/particle_spacing),
                               round(Int, rectangular_height/particle_spacing),
                               0.0, 0.0)
```
"""
struct RectangularShape{NDIMS, ELTYPE<:Real}
    coordinates                 ::Array{ELTYPE, 2}
    masses                      ::Vector{ELTYPE}
    densities                   ::Vector{ELTYPE}
    particle_spacing            ::ELTYPE
    n_particles_per_dimension   ::NTuple{NDIMS, Int}

    function RectangularShape(particle_spacing, n_particles_x, n_particles_y,
                              x_position, y_position; density=0.0)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        n_particles = n_particles_y * n_particles_x

        coordinates = Array{Float64, 2}(undef, 2, n_particles)

        densities = density * ones(ELTYPE, n_particles)
        masses = density * particle_spacing^2 * ones(ELTYPE, n_particles)

        initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                n_particles_x, n_particles_y)

        n_particles_per_dimension = (n_particles_x, n_particles_y)

        return new{NDIMS, ELTYPE}(coordinates, masses, densities,
                                  particle_spacing, n_particles_per_dimension)
    end
end


function initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                 n_particles_x, n_particles_y)

    boundary_particle = 0

    for x in 0:n_particles_x-1
        for y in 0:n_particles_y-1
            boundary_particle += 1

            coordinates[1, boundary_particle] = x_position + x * particle_spacing
            coordinates[2, boundary_particle] = y_position + y * particle_spacing
        end
    end
end
