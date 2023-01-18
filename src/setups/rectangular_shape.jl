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
    radius                    :: Vector{ELTYPE}
    velocity                  :: Array{ELTYPE, 2}
    particle_spacing          :: ELTYPE
    n_particles_per_dimension :: NTuple{NDIMS, Int}

    function RectangularShape(particle_spacing, n_particles_x, n_particles_y,
                              x_position, y_position; density=0.0)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        n_particles = n_particles_y * n_particles_x

        coordinates = Array{Float64, 2}(undef, 2, n_particles)

        densities = density * ones(ELTYPE, n_particles)
        masses = density * particle_spacing^2 * ones(ELTYPE, n_particles)
        velocity = zeros(ELTYPE, n_particles)
        radius = particle_spacing * ones(ELTYPE, n_particles)

        initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                n_particles_x, n_particles_y)

        n_particles_per_dimension = (n_particles_x, n_particles_y)

        return new{NDIMS, ELTYPE}(coordinates, masses, densities, radius, velocity,
                                  particle_spacing, n_particles_per_dimension)
    end

    function RectangularShape(particle_spacing, cornerA, cornerB; density=0.0)
        NDIMS = length(cornerA)
        ELTYPE = eltype(particle_spacing)

        cornerA = collect(cornerA)
        cornerB = collect(cornerB)

        floor_int(x) = floor(Int64, x)
        diff = broadcast(abs, cornerA - cornerB)
        n_particles_i = broadcast(floor_int, diff/particle_spacing) .+ 1
        n_particles = prod(n_particles_i)

        coordinates = Array{Float64, 2}(undef, 2, n_particles)
        densities = density * ones(ELTYPE, n_particles)
        masses = density * particle_spacing^NDIMS * ones(ELTYPE, n_particles)
        velocity = zeros(ELTYPE, NDIMS, n_particles)
        radius = particle_spacing * ones(ELTYPE, n_particles)


        initialize_rectangular!(coordinates, cornerA[1], cornerA[2], particle_spacing,
        n_particles_i[1], n_particles_i[2])

        return new{NDIMS, ELTYPE}(coordinates, masses, densities, radius, velocity,
                                  particle_spacing, Tuple(n_particles_i))
    end
end

function initialize_rectangular!(coordinates, x_position, y_position, particle_spacing,
                                 n_particles_x, n_particles_y)
    boundary_particle = 0

    for x in 0:(n_particles_x - 1)
        for y in 0:(n_particles_y - 1)
            boundary_particle += 1

            coordinates[1, boundary_particle] = x_position + x * particle_spacing
            coordinates[2, boundary_particle] = y_position + y * particle_spacing
        end
    end
end
