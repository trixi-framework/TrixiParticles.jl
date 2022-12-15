struct RectangularWall{NDIMS, ELTYPE<:Real}
    coordinates             ::Array{ELTYPE, 2}
    masses                  ::Vector{ELTYPE}
    particle_spacing        ::ELTYPE
    spacing_ratio           ::ELTYPE
    n_layers                ::Int
    n_particles_x           ::Int
    n_particles_y           ::Int

    function RectangularWall(particle_spacing, spacing_ratio, wall_height,
                             wall_position, boundary_density;
                             n_layers=1)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        # Boundary particle data
        n_particles_x,
            n_particles_y = get_boundary_particles_per_dimension(wall_height, particle_spacing,
                                                                  spacing_ratio, n_layers)
        n_particles = n_particles_y * n_particles_x

        coordinates = Array{Float64, 2}(undef, 2, n_particles)

        initialize_wall!(coordinates, wall_position, particle_spacing, spacing_ratio,
                               n_particles_x, n_particles_y)
        masses = boundary_density * (particle_spacing / spacing_ratio)^2 * ones(ELTYPE, n_particles)


        return new{NDIMS, ELTYPE}(coordinates, masses,
                                  particle_spacing, spacing_ratio, n_layers,
                                  n_particles_x, n_particles_y)
    end
end


function initialize_wall!(coordinates, wall_position, particle_spacing, spacing_ratio,
                          n_particles_x, n_particles_y)
    boundary_particle_spacing = particle_spacing / spacing_ratio

    boundary_particle = 0
    for i in 0:n_particles_x-1
        for y in 1:n_particles_y
            boundary_particle += 1

            coordinates[1, boundary_particle] = wall_position + i*boundary_particle_spacing
            coordinates[2, boundary_particle] = y * boundary_particle_spacing
        end
    end
end


function get_boundary_particles_per_dimension(wall_height, particle_spacing, spacing_ratio, n_layers)
    n_particles_x = n_layers
    n_particles_y = round(Int, (wall_height / particle_spacing * spacing_ratio))

    return n_particles_x, n_particles_y
end
