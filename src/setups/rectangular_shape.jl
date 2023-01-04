"""
    RectangularShape(particle_spacing, n_particles_x, n_particles_y,
                     x_position, y_position; density=0.0)

Rectangular shape filled with particles.

The arguments are as follows:
- `particle_spacing`:                   Spacing betweeen the particles
- `n_particles_x`, `n_particles_y`:     Number of particles in x and y direction, respectively.
- `x_position`, `y_position`:           Starting point of the reactangular in x and y direction, respectively.

Specifying the density is optional since only the coordinates of the particles may be needed (see example below).

# Example
```julia
rectangular = RectangularShape(particle_spacing,
                               round(Int, rectangular_width/particle_spacing),
                               round(Int, rectangular_height/particle_spacing),
                               0.0, 0.0)
```
"""
struct RectangularShape{NDIMS, ELTYPE<:Real}
    coordinates             ::Array{ELTYPE, 2}
    masses                  ::Vector{ELTYPE}
    densities               ::Vector{ELTYPE}
    particle_spacing        ::ELTYPE
    spacing_ratio           ::ELTYPE
    n_particles_x           ::Int
    n_particles_y           ::Int

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


        return new{NDIMS, ELTYPE}(coordinates, masses, densities,
                                  particle_spacing,
                                  n_particles_x, n_particles_y)
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
