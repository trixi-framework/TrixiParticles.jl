struct RectangularTank{NDIMS, ELTYPE<:Real}
    particle_coordinates    ::Array{ELTYPE, 2}
    particle_velocities     ::Array{ELTYPE, 2}
    particle_densities      ::Vector{ELTYPE}
    particle_masses         ::Vector{ELTYPE}
    boundary_coordinates    ::Array{ELTYPE, 2}
    boundary_masses         ::Vector{ELTYPE}

    function RectangularTank(particle_spacing, spacing_ratio, fluid_width, fluid_heigth,
                             container_width, container_height, rest_density;
                             init_velocity=0.0, boundary_density=rest_density)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)
        mass = rest_density * particle_spacing^2

        # Particle data
        n_particles_per_dimension = (Int(fluid_width / particle_spacing),
                                     Int(fluid_heigth / particle_spacing))

        particle_coordinates, particle_velocities = initialize_particles(particle_spacing, init_velocity, n_particles_per_dimension)
        particle_densities = rest_density * ones(Float64, prod(n_particles_per_dimension))
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        # Boundary particle data
        n_boundaries_x = Int(container_width / particle_spacing * spacing_ratio) + 1
        n_boundaries_y = Int(container_height / particle_spacing * spacing_ratio) - 1
        n_boundaries   = 2 * n_boundaries_y + n_boundaries_x

        boundary_coordinates = initialize_boundaries(particle_spacing, spacing_ratio,
                                                     n_boundaries, n_boundaries_x, n_boundaries_y)


        boundary_masses = boundary_density * particle_spacing^2 * ones(ELTYPE, n_boundaries)

        return new{NDIMS, ELTYPE}(particle_coordinates, particle_velocities, particle_densities, particle_masses,
                                  boundary_coordinates, boundary_masses)
    end

    function RectangularTank(particle_spacing, spacing_ratio,
                             fluid_width, fluid_heigth, fluid_depth,
                             container_width, container_height, container_depth,
                             rest_density;
                             init_velocity=0.0, boundary_density=rest_density)
        NDIMS = 3
        ELTYPE = eltype(particle_spacing)
        mass = rest_density * particle_spacing^2

        # Particle data
        n_particles_per_dimension = (Int(fluid_width / particle_spacing),
                                     Int(fluid_heigth / particle_spacing),
                                     Int(fluid_depth / particle_spacing))

        particle_coordinates, particle_velocities = initialize_particles(particle_spacing, init_velocity, n_particles_per_dimension)
        particle_densities = rest_density * ones(Float64, prod(n_particles_per_dimension))
        particle_masses = mass * ones(ELTYPE, prod(n_particles_per_dimension))

        # Boundary particle data

        n_boundaries_x = ceil(Int, (container_width + particle_spacing) / particle_spacing * spacing_ratio) - 1
        n_boundaries_y = Int(container_height / particle_spacing * spacing_ratio) + 1
        n_boundaries_z = Int((container_depth + particle_spacing) / particle_spacing * spacing_ratio) - 1
        n_boundaries   = n_boundaries_x * n_boundaries_z + 2 * n_boundaries_x * n_boundaries_y + 2 * (n_boundaries_z + 1) * n_boundaries_y

        boundary_coordinates = initialize_boundaries(particle_spacing, spacing_ratio, n_boundaries,
                                                     n_boundaries_x, n_boundaries_y, n_boundaries_z)


        boundary_masses = boundary_density * particle_spacing^2 * ones(ELTYPE, n_boundaries)

        return new{NDIMS, ELTYPE}(particle_coordinates, particle_velocities, particle_densities, particle_masses,
                                  boundary_coordinates, boundary_masses)
    end
end


function  initialize_particles(particle_spacing, init_velocity, n_particles_per_dimension::Tuple{Int64, Int64})
    particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
    particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))

    for y in 1:n_particles_per_dimension[2],
            x in 1:n_particles_per_dimension[1]
        particle = (x - 1) * n_particles_per_dimension[2] + y

        # Coordinates
        particle_coordinates[1, particle] = x * particle_spacing
        particle_coordinates[2, particle] = y * particle_spacing

        # Velocities
        particle_velocities[1, particle] = init_velocity
        particle_velocities[2, particle] = init_velocity
    end

    return particle_coordinates, particle_velocities
end

function  initialize_particles(particle_spacing, init_velocity, n_particles_per_dimension::Tuple{Int64, Int64, Int64})
    particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
    particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))

    for z in 1:n_particles_per_dimension[3],
            y in 1:n_particles_per_dimension[2],
                x in 1:n_particles_per_dimension[1]
        particle = (x - 1) * n_particles_per_dimension[2] * n_particles_per_dimension[3] +
            (y - 1) * n_particles_per_dimension[3] + z

        # Coordinates
        particle_coordinates[1, particle] = x * particle_spacing
        particle_coordinates[2, particle] = y * particle_spacing
        particle_coordinates[3, particle] = z * particle_spacing

        # Velocities
        particle_velocities[1, particle] = init_velocity
        particle_velocities[2, particle] = init_velocity
        particle_velocities[3, particle] = init_velocity
    end

    return particle_coordinates, particle_velocities
end


function initialize_boundaries(particle_spacing, spacing_ratio, n_boundaries, n_boundaries_x, n_boundaries_y)
    boundary_coordinates = Array{Float64, 2}(undef, 2, n_boundaries)
    boundary_particle_spacing = particle_spacing/spacing_ratio

    # Left boundary
    for y in 1:n_boundaries_y
        boundary_particle = y

        boundary_coordinates[1, boundary_particle] = 0
        boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
    end

    # Right boundary
    for y in 1:n_boundaries_y
        boundary_particle = n_boundaries_y + y

        boundary_coordinates[1, boundary_particle] = n_boundaries_x * boundary_particle_spacing
        boundary_coordinates[2, boundary_particle] = y * boundary_particle_spacing
    end

    # Bottom boundary
    for x in 1:n_boundaries_x
        boundary_particle = 2*n_boundaries_y + x

        boundary_coordinates[1, boundary_particle] = (x - 1) * boundary_particle_spacing
        boundary_coordinates[2, boundary_particle] = 0
    end

    return boundary_coordinates
end

function initialize_boundaries(particle_spacing, spacing_ratio, n_boundaries,
                               n_boundaries_x, n_boundaries_y, n_boundaries_z)
    boundary_coordinates = Array{Float64, 2}(undef, 3, n_boundaries)
    boundary_particle_spacing = particle_spacing/spacing_ratio

    boundary_particle = 0
    # -x boundary (y-z-plane)
    for z in 1:(n_boundaries_z + 1), y in 1:n_boundaries_y
        boundary_particle += 1

        boundary_coordinates[1, boundary_particle] = 0
        boundary_coordinates[2, boundary_particle] = (y - 1) * boundary_particle_spacing
        boundary_coordinates[3, boundary_particle] = (z - 1) * boundary_particle_spacing
    end

    # +x boundary (y-z-plane)
    for z in 1:(n_boundaries_z + 1), y in 1:n_boundaries_y
        boundary_particle += 1

        boundary_coordinates[1, boundary_particle] = n_boundaries_x*boundary_particle_spacing
        boundary_coordinates[2, boundary_particle] = (y - 1) * boundary_particle_spacing
        boundary_coordinates[3, boundary_particle] = (z - 1) * boundary_particle_spacing
    end

    # -z boundary (x-y-plane)
    for y in 1:n_boundaries_y, x in 1:n_boundaries_x
        boundary_particle += 1

        boundary_coordinates[1, boundary_particle] = x * boundary_particle_spacing
        boundary_coordinates[2, boundary_particle] = (y - 1) * boundary_particle_spacing
        boundary_coordinates[3, boundary_particle] = 0
    end

    # +z boundary (x-y-plane)
    for y in 1:n_boundaries_y, x in 1:n_boundaries_x
        boundary_particle += 1

        boundary_coordinates[1, boundary_particle] = x * boundary_particle_spacing
        boundary_coordinates[2, boundary_particle] = (y - 1) * boundary_particle_spacing
        boundary_coordinates[3, boundary_particle] = n_boundaries_z*boundary_particle_spacing
    end

    # Bottom boundary (x-z-plane)
    for z in 1:n_boundaries_z, x in 1:n_boundaries_x
        boundary_particle += 1

        boundary_coordinates[1, boundary_particle] = x * boundary_particle_spacing
        boundary_coordinates[2, boundary_particle] = 0
        boundary_coordinates[3, boundary_particle] = z * boundary_particle_spacing
    end

    return boundary_coordinates
end
