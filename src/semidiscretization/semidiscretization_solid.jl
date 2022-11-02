struct SPHSolidSemidiscretization{NDIMS, ELTYPE<:Real, DC, K, NS, C} <: SPHSemidiscretization{NDIMS}
    density_calculator  ::DC
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    gravity             ::SVector{NDIMS, ELTYPE}
    neighborhood_search ::NS
    cache               ::C

    function SPHSolidSemidiscretization{NDIMS}(particle_masses, particle_densities,
                                               density_calculator,
                                               smoothing_kernel, smoothing_length,
                                               young_modulus, poisson_ratio;
                                               gravity=ntuple(_ -> 0.0, Val(NDIMS)),
                                               neighborhood_search=nothing) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make gravity an SVector
        gravity_ = SVector(gravity...)

        lame_lambda = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2*poisson_ratio))
        lame_mu = 0.5 * young_modulus / (1 + poisson_ratio)

        initial_coordinates = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
        current_coordinates = similar(initial_coordinates)
        correction_matrix   = zeros(ELTYPE, NDIMS, NDIMS, nparticles)
        solid_density = particle_densities
        mass = particle_masses

        cache = (; lame_lambda, lame_mu, mass, solid_density,
                   initial_coordinates, current_coordinates, correction_matrix,
                   create_cache(density_calculator, ELTYPE, nparticles)...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(smoothing_kernel),
                   typeof(neighborhood_search), typeof(cache)}(
            density_calculator, smoothing_kernel, smoothing_length,
            gravity_, neighborhood_search, cache)
    end
end


function semidiscretize(semi::SPHSolidSemidiscretization{NDIMS, ELTYPE, SummationDensity},
                        particle_coordinates, particle_velocities, tspan;
                        n_fixed_particles = 0) where {NDIMS, ELTYPE}
    @unpack smoothing_kernel, smoothing_length, neighborhood_search, cache = semi
    @unpack mass, solid_density, initial_coordinates, current_coordinates, correction_matrix = cache

    # Only save the moving particle positions and velocities in u0
    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(semi), nparticles(semi) - n_fixed_particles)

    for particle in each_moving_particle(u0, semi)
        # Set particle coordinates and initial coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim + ndims(semi), particle] = particle_velocities[dim, particle]
        end
    end

    # Everything in the cache like initial_coordinates should be saved for fixed particles as well
    for particle in eachparticle(semi)
        # Set particle coordinates and initial coordinates
        for dim in 1:ndims(semi)
            initial_coordinates[dim, particle] = particle_coordinates[dim, particle]
            current_coordinates[dim, particle] = particle_coordinates[dim, particle]
        end
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, initial_coordinates, semi)

    # Calculate kernel correction matrix
    calc_correction_matrix!(correction_matrix, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function calc_correction_matrix!(correction_matrix, semi)
    @unpack cache, smoothing_kernel, smoothing_length, neighborhood_search = semi
    @unpack initial_coordinates, mass, solid_density = cache

    # Calculate kernel correction matrix
    for particle in eachparticle(semi)
        L = zeros(eltype(mass), ndims(semi), ndims(semi))

        for neighbor in eachneighbor(particle, initial_coordinates, neighborhood_search, semi)
            volume = mass[neighbor] / solid_density[neighbor]

            initial_pos_diff = get_particle_coords(initial_coordinates, semi, particle) -
                get_particle_coords(initial_coordinates, semi, neighbor)
            initial_distance = norm(initial_pos_diff)

            if initial_distance > eps()
                grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
                    initial_pos_diff / initial_distance

                L -= volume * grad_kernel * transpose(initial_pos_diff)
            end
        end

        correction_matrix[:, :, particle] = inv(L)
    end

    return correction_matrix
end


@inline each_moving_particle(u, semi) = Base.OneTo(size(u, 2))
