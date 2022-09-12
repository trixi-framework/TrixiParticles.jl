struct RectangularTank{NDIMS, ELTYPE<:Real}
    particle_spacing    ::ELTYPE
    fluid_width         ::ELTYPE
    fluid_heigth        ::ELTYPE
    fluid_depth         ::ELTYPE
    container_width     ::ELTYPE
    container_height    ::ELTYPE
    container_depth     ::ELTYPE
    fluid_density       ::ELTYPE

    function RectangularTank(particle_spacing, fluid_width, fluid_heigth)
        NDIMS = 2
        ELTYPE = eltype(particle_spacing)

        # Particle data
        n_particles_per_dimension = (Int(water_width / particle_spacing),
                                     Int(water_height / particle_spacing))

        particle_coordinates = initialize_particle_coords(particle_spacing, n_particles_per_dimension)

        # Make boundary_conditions a tuple
        boundary_conditions_ = digest_boundary_conditions(boundary_conditions)

        # Make gravity an SVector
        gravity_ = SVector(gravity...)

        cache = (; create_cache(particle_masses, density_calculator, ELTYPE, nparticles)...)

        return new{NDIMS, ELTYPE}(
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, boundary_conditions_, gravity_, neighborhood_search, cache)
    end
end


function  initialize_particle_coords(particle_spacing, n_particles_per_dimension::Tuple{Int64, Int64})
    particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))

    for y in 1:n_particles_per_dimension[2],
            x in 1:n_particles_per_dimension[1]
        particle = (x - 1) * n_particles_per_dimension[2] + y

        particle_coordinates[1, particle] = x * particle_spacing
        particle_coordinates[2, particle] = y * particle_spacing
    end

    return particle_coordinates
end


function  initialize_particle_coords(particle_spacing, n_particles_per_dimension::Tuple{Int64, Int64, Int64})
    particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))

    for z in 1:n_particles_per_dimension[3],
            y in 1:n_particles_per_dimension[2],
                x in 1:n_particles_per_dimension[1]
        particle = (x - 1) * n_particles_per_dimension[2] * n_particles_per_dimension[3] +
            (y - 1) * n_particles_per_dimension[3] + z

        particle_coordinates[1, particle] = x * particle_spacing
        particle_coordinates[2, particle] = y * particle_spacing
        particle_coordinates[3, particle] = z * particle_spacing
    end

    return particle_coordinates
end
