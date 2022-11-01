struct SPHSolidSemidiscretization{NDIMS, ELTYPE<:Real, DC, K, NS, C} <: SPHSemidiscretization{NDIMS}
    density_calculator  ::DC
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    gravity             ::SVector{NDIMS, ELTYPE}
    neighborhood_search ::NS
    cache               ::C

    function SPHSolidSemidiscretization{NDIMS}(particle_masses, particle_densities,
                                               density_calculator,
                                               smoothing_kernel, smoothing_length;
                                               gravity=ntuple(_ -> 0.0, Val(NDIMS)),
                                               neighborhood_search=nothing) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make gravity an SVector
        gravity_ = SVector(gravity...)

        cache = (; create_cache(particle_masses, particle_densities,
                                density_calculator, ELTYPE, nparticles, NDIMS)...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(smoothing_kernel),
                   typeof(neighborhood_search), typeof(cache)}(
            density_calculator, smoothing_kernel, smoothing_length,
            gravity_, neighborhood_search, cache)
    end
end


function create_cache(mass, solid_density, density_calculator, eltype, nparticles, ndims)
    initial_coordinates = Array{eltype, 2}(undef, ndims, nparticles)
    correction_matrix   = zeros(eltype, ndims, ndims, nparticles)

    return (; mass, solid_density, initial_coordinates, correction_matrix,
              create_cache(density_calculator, eltype, nparticles)...)
end


function semidiscretize(semi::SPHSolidSemidiscretization{NDIMS, ELTYPE, SummationDensity},
                        particle_coordinates, particle_velocities, tspan) where {NDIMS, ELTYPE}
    @unpack smoothing_kernel, smoothing_length, neighborhood_search, cache = semi
    @unpack mass, solid_density, initial_coordinates, correction_matrix = cache

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(semi), nparticles(semi))

    for particle in eachparticle(semi)
        # Set particle coordinates and initial coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
            initial_coordinates[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim + ndims(semi), particle] = particle_velocities[dim, particle]
        end
    end

    for particle in eachparticle(semi)
        L = sum(eachneighbor(particle, initial_coordinates, neighborhood_search, semi)) do neighbor
            volume = mass[neighbor] / solid_density[neighbor]

            initial_pos_diff = get_particle_coords(initial_coordinates, semi, particle) -
                get_particle_coords(initial_coordinates, semi, neighbor)
            initial_distance = norm(initial_pos_diff)

            if initial_distance < eps()
                return zeros(eltype(mass), ndims(semi), ndims(semi))
            end

            grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
                initial_pos_diff / initial_distance

            return -volume * grad_kernel * transpose(initial_pos_diff)
        end

        correction_matrix[:, :, particle] = inv(L)
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end
