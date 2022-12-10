struct VerticalWall{NDIMS, ELTYPE<:Real}
    boundary_coordinates    ::Array{ELTYPE, 2}
    boundary_masses         ::Vector{ELTYPE}
    particle_spacing        ::ELTYPE
    spacing_ratio           ::ELTYPE
    n_layers                ::Int
    n_boundaries_x          ::Int
    n_boundaries_y          ::Int

    function VerticalWall(particle_spacing, spacing_ratio, wall_height,
                             wall_position, boundary_density;
                             n_layers=1, init_velocity=0.0)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        # Boundary particle data
        n_boundaries_x,
            n_boundaries_y = get_boundary_particles_per_dimension(wall_height, particle_spacing,
                                                                  spacing_ratio, n_layers)
        n_boundaries = n_boundaries_y * n_boundaries_x

        boundary_coordinates = Array{Float64, 2}(undef, 2, n_boundaries)

        initialize_wall!(boundary_coordinates, wall_position, particle_spacing, spacing_ratio,
                               n_boundaries_x, n_boundaries_y)
        boundary_masses = boundary_density * (particle_spacing / spacing_ratio)^2 * ones(ELTYPE, n_boundaries)


        return new{NDIMS, ELTYPE}(boundary_coordinates, boundary_masses,
                                  particle_spacing, spacing_ratio, n_layers,
                                  n_boundaries_x, n_boundaries_y)
    end
end


function initialize_wall!(boundary_coordinates, wall_position, particle_spacing, spacing_ratio,
                                n_boundaries_x, n_boundaries_y)
    boundary_particle_spacing = particle_spacing / spacing_ratio

    boundary_particle = 0
    for i in 0:n_boundaries_x-1
        # Left boundary
        for y in 1:n_boundaries_y
            boundary_particle += 1

            boundary_coordinates[1, boundary_particle] = wall_position + i*boundary_particle_spacing
            boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
        end
    end
end


function get_boundary_particles_per_dimension(wall_height, particle_spacing, spacing_ratio, n_layers)
    n_boundaries_x = n_layers
    n_boundaries_y = round(Int, (wall_height / particle_spacing * spacing_ratio))

    return n_boundaries_x, n_boundaries_y
end
