struct SPHFluidSemidiscretization{NDIMS, ELTYPE<:Real, DC, SE, K, V, BC, NS, C} <: SPHSemidiscretization{NDIMS}
    density_calculator  ::DC
    state_equation      ::SE
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    viscosity           ::V
    boundary_conditions ::BC
    gravity             ::SVector{NDIMS, ELTYPE}
    neighborhood_search ::NS
    cache               ::C

    function SPHFluidSemidiscretization{NDIMS}(particle_masses,
                                               density_calculator, state_equation,
                                               smoothing_kernel, smoothing_length;
                                               viscosity=NoViscosity(),
                                               boundary_conditions=nothing,
                                               gravity=ntuple(_ -> 0.0, Val(NDIMS)),
                                               neighborhood_search=nothing) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        # Make boundary_conditions a tuple
        boundary_conditions_ = digest_boundary_conditions(boundary_conditions)

        # Make gravity an SVector
        gravity_ = SVector(gravity...)

        cache = (; create_cache(particle_masses, density_calculator, ELTYPE, nparticles)...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation),
                   typeof(smoothing_kernel), typeof(viscosity), typeof(boundary_conditions_),
                   typeof(neighborhood_search), typeof(cache)}(
            density_calculator, state_equation, smoothing_kernel, smoothing_length,
            viscosity, boundary_conditions_, gravity_, neighborhood_search, cache)
    end
end


function create_cache(mass, density_calculator, eltype, nparticles)
    pressure = Vector{eltype}(undef, nparticles)

    return (; mass, pressure, create_cache(density_calculator, eltype, nparticles)...)
end


function semidiscretize(semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, SummationDensity},
                        particle_coordinates, particle_velocities, tspan) where {NDIMS, ELTYPE}
    @unpack neighborhood_search, boundary_conditions = semi

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(semi), nparticles(semi))

    for particle in eachparticle(semi)
        # Set particle coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim + ndims(semi), particle] = particle_velocities[dim, particle]
        end
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u0, semi)

    # Initialize boundary conditions
    @pixie_timeit timer() "initialize boundary conditions" for bc in boundary_conditions
        initialize!(bc, semi)
    end

    # Compute quantities like density and pressure
    compute_quantities(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function semidiscretize(semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, ContinuityDensity},
                        particle_coordinates, particle_velocities, particle_densities, tspan) where {NDIMS, ELTYPE}
    @unpack neighborhood_search, boundary_conditions = semi

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(semi) + 1, nparticles(semi))

    for particle in eachparticle(semi)
        # Set particle coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim + ndims(semi), particle] = particle_velocities[dim, particle]
        end

        # Set particle densities
        u0[2 * ndims(semi) + 1, particle] = particle_densities[particle]
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u0, semi)

    # Initialize boundary conditions
    @pixie_timeit timer() "initialize boundary conditions" for bc in boundary_conditions
        initialize!(bc, semi)
    end

    # Compute quantities like pressure
    compute_quantities(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end
