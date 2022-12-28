struct RectangularShape{NDIMS, ELTYPE<:Real}
    coordinates             ::Array{ELTYPE, 2}
    masses                  ::Vector{ELTYPE}
    particle_spacing        ::ELTYPE
    spacing_ratio           ::ELTYPE
    n_particles_x           ::Int
    n_particles_y           ::Int

    function RectangularShape(particle_spacing,
                              n_particles_x, n_particles_y,
                              x_position, y_position;
                              density=0.0, spacing_ratio=1.0)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        n_particles = n_particles_y * n_particles_x

        coordinates = Array{Float64, 2}(undef, 2, n_particles)

        masses = density * (particle_spacing / spacing_ratio)^2 * ones(ELTYPE, n_particles)

        initialize_rectangular!(coordinates, x_position, y_position, particle_spacing, spacing_ratio,
                               n_particles_x, n_particles_y)


        return new{NDIMS, ELTYPE}(coordinates, masses,
                                  particle_spacing, spacing_ratio,
                                  n_particles_x, n_particles_y)
    end
end


function initialize_rectangular!(coordinates, x_position, y_position, particle_spacing, spacing_ratio,
                          n_particles_x, n_particles_y)
    new_spacing = particle_spacing / spacing_ratio

    boundary_particle = 0
    for x in 0:n_particles_x-1
        for y in 0:n_particles_y-1
            boundary_particle += 1

            coordinates[1, boundary_particle] = x_position + x * new_spacing
            coordinates[2, boundary_particle] = y_position + y * new_spacing
        end
    end
end
