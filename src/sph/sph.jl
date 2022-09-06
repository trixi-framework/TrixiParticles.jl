
function semidiscretize(semi::WCSPHSemidiscretization{NDIMS,ELTYPE,SummationDensity},
    particle_coordinates, particle_velocities, tspan) where {NDIMS,ELTYPE}
    @unpack neighborhood_search, boundary_conditions = semi

    u0 = Array{eltype(particle_coordinates),2}(undef, 2 * ndims(semi), nparticles(semi))

    for particle in eachparticle(semi)
        # Set particle coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim+ndims(semi), particle] = particle_velocities[dim, particle]
        end
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u0, semi)

    # Initialize boundary conditions
    @pixie_timeit timer() "initialize boundary conditions" for bc in boundary_conditions
        initialize!(bc, semi)
    end

    return ODEProblem(rhs!, u0, tspan, semi)
end


function semidiscretize(semi::WCSPHSemidiscretization{NDIMS,ELTYPE,ContinuityDensity},
    particle_coordinates, particle_velocities, particle_densities, tspan) where {NDIMS,ELTYPE}
    @unpack neighborhood_search, boundary_conditions = semi

    u0 = Array{eltype(particle_coordinates),2}(undef, 2 * ndims(semi) + 1, nparticles(semi))

    for particle in eachparticle(semi)
        # Set particle coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim+ndims(semi), particle] = particle_velocities[dim, particle]
        end

        # Set particle densities
        u0[2*ndims(semi)+1, particle] = particle_densities[particle]
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u0, semi)

    # Initialize boundary conditions
    @pixie_timeit timer() "initialize boundary conditions" for bc in boundary_conditions
        initialize!(bc, semi)
    end

    return ODEProblem(rhs!, u0, tspan, semi)
end



# Maybe here is no multiple dispatch necessary to distinguish EISPH from ISPH. 
function semidiscretize(semi::EISPHSemidiscretization{NDIMS,ELTYPE,SummationDensity},
    particle_coordinates, particle_velocities, particle_densities, tspan) where {NDIMS,ELTYPE}
    @unpack neighborhood_search, boundary_conditions, cache = semi

    u0 = Array{eltype(particle_coordinates),2}(undef, 2 * ndims(semi) + 1, nparticles(semi))

    for particle in eachparticle(semi)
        # Set particle coordinates
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        # Set particle velocities
        for dim in 1:ndims(semi)
            u0[dim+ndims(semi), particle] = particle_velocities[dim, particle]
        end

        # Set particle densities
        u0[end, particle] = particle_densities[particle]

        cache.density[particle] = particle_densities[particle]
    end

    # Initialize neighborhood search
    @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u0, semi)

    # Initialize boundary conditions
    @pixie_timeit timer() "initialize boundary conditions" for bc in boundary_conditions
        initialize!(bc, semi)
    end

    return ODEProblem(rhs!, u0, tspan, semi)
end


function compute_quantities(u, semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    @threaded for particle in eachparticle(semi)
        compute_quantities_per_particle(u, particle, semi)
    end
    update_pressure(semi)
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_quantities_per_particle(u, particle, semi::WCSPHSemidiscretization{NDIMS,ELTYPE,SummationDensity}) where {NDIMS,ELTYPE}
    @unpack smoothing_kernel, smoothing_length, state_equation,
    neighborhood_search, cache = semi
    @unpack mass, density, pressure = cache

    density[particle] = zero(eltype(density))

    for neighbor in eachneighbor(particle, u, neighborhood_search, semi)
        distance = norm(get_particle_coords(u, semi, particle) -
                        get_particle_coords(u, semi, neighbor))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass[neighbor] * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end

    pressure[particle] = state_equation(density[particle])
end

@inline function compute_quantities_per_particle(u, particle, semi::WCSPHSemidiscretization{NDIMS,ELTYPE,ContinuityDensity}) where {NDIMS,ELTYPE}
    @unpack density_calculator, state_equation, cache = semi
    @unpack pressure = cache

    # Note that @threaded makes this slower with ContinuityDensity
    for particle in eachparticle(semi)
        pressure[particle] = state_equation(get_particle_density(u, cache, density_calculator, particle))
    end
end


@inline function compute_quantities_per_particle(u, particle, semi::EISPHSemidiscretization{NDIMS,ELTYPE,SummationDensity}) where {NDIMS,ELTYPE}
    @unpack  smoothing_kernel, smoothing_length, neighborhood_search, cache, pressure_poisson_eq = semi
    @unpack mass, density, pressure, dt, dv_viscosities = cache

    u[end, particle] = zero(eltype(density))

    for neighbor in eachneighbor(particle, u, neighborhood_search, semi)
        distance = norm(get_intermediate_coords(u, semi, particle, dt[1]) -
                        get_intermediate_coords(u, semi, neighbor, dt[1]))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            u[end, particle] += mass[neighbor] * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end

    pressure[particle] = pressure_poisson_eq(u, semi, particle)
end


@inline function update_pressure(semi::EISPHSemidiscretization)
    @unpack cache = semi
    @unpack pressure, prior_pressure, dv_viscosities = cache

    for particle in eachparticle(semi)
        prior_pressure[particle] = pressure[particle]
        # reset pressureless momentum eq
        dv_viscosities[particle] = zero(eltype(dv_viscosities))
    end    
end
@inline function update_pressure(semi::WCSPHSemidiscretization)
end

function rhs!(du, u, semi, t)
    @unpack smoothing_kernel, smoothing_length,
    boundary_conditions, gravity,
    neighborhood_search = semi
    @pixie_timeit timer() "rhs!" begin
        @pixie_timeit timer() "update neighborhood search" update!(neighborhood_search, u, semi)

        # Reset du
        @pixie_timeit timer() "reset ∂u/∂t" reset_du!(du)

        # Compute quantities
        @pixie_timeit timer() "compute quantities" compute_quantities(u, semi)

        # u[1:3] = coordinates
        # u[4:6] = velocity
        @pixie_timeit timer() "main loop" @threaded for particle in eachparticle(semi)

            # dr = v
            for i in 1:ndims(semi)
                du[i, particle] = u[i+ndims(semi), particle]
            end

            particle_coords = get_particle_coords(u, semi, particle)
            for neighbor in eachneighbor(particle, u, neighborhood_search, semi)
                neighbor_coords = get_particle_coords(u, semi, neighbor)

                pos_diff = particle_coords - neighbor_coords
                distance = norm(pos_diff)

                if eps() < distance <= compact_support(smoothing_kernel, smoothing_length)
                    calc_dv!(du, u, particle, neighbor, pos_diff, distance, semi)

                    continuity_equation!(du, u, particle, neighbor, pos_diff, distance, semi)
                end
            end

            add_gravity!(du, particle, semi)

            # boundary impact
            for bc in boundary_conditions
                calc_boundary_condition_per_particle!(du, u, particle, bc, semi)
            end
        end
    end

    return du
end


@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end


@inline function add_gravity!(du, particle, semi)
    @unpack gravity = semi
    for i in 1:ndims(semi)
        # Gravity
        du[i+ndims(semi), particle] += gravity[i]
    end
    return du
end

@inline function calc_dv!(du, u, particle, neighbor, pos_diff, distance, semi::WCSPHSemidiscretization)
    @unpack smoothing_kernel, smoothing_length,
    density_calculator, viscosity, cache = semi
    @unpack mass, pressure = cache

    density_particle = get_particle_density(u, cache, density_calculator, particle)
    density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)

    # Viscosity
    v_diff = get_particle_vel(u, semi, particle) - get_particle_vel(u, semi, neighbor)
    pi_ab = viscosity(v_diff, pos_diff,
        distance, density_particle, density_neighbor, smoothing_length)

    dv = -mass[neighbor] * (pressure[particle] / density_particle^2 +
                            pressure[neighbor] / density_neighbor^2 + pi_ab) *
         kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    for i in 1:ndims(semi)
        du[ndims(semi)+i, particle] += dv[i]
    end

    return du
end

@inline function calc_dv!(du, u, particle, neighbor, pos_diff, distance, semi::EISPHSemidiscretization)
    @unpack smoothing_kernel, smoothing_length,
    density_calculator, viscosity, cache = semi
    @unpack mass, pressure, dv_viscosities = cache

    density_particle = get_particle_density(u, cache, density_calculator, particle)
    density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)

    # Viscosity
    v_diff = get_particle_vel(u, semi, particle) - get_particle_vel(u, semi, neighbor)
    # TBD: viscosity() has a negativ return value.
    pi_ab = -viscosity(v_diff, pos_diff, distance, density_particle, density_neighbor, smoothing_length)

    dv_pressure = -mass[neighbor] * (pressure[particle] / density_particle^2 +
                                    pressure[neighbor] / density_neighbor^2) *
                    kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    dv_viscosity = mass[neighbor] * pi_ab * kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    dv = dv_pressure + dv_viscosity

    for i in 1:ndims(semi)
        du[i+ndims(semi), particle] += dv[i]
        dv_viscosities[i, particle] += dv_viscosity[i]
    end

    
    return du
end

@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
    semi::WCSPHSemidiscretization{NDIMS,ELTYPE,ContinuityDensity}) where {NDIMS,ELTYPE}
    @unpack smoothing_kernel, smoothing_length, cache = semi
    @unpack mass = cache

    vdiff = get_particle_vel(u, semi, particle) -
            get_particle_vel(u, semi, neighbor)

    du[end, particle] += sum(mass[particle] * vdiff *
                                         kernel_deriv(smoothing_kernel, distance, smoothing_length) .*
                                         pos_diff) / distance

    return du
end

@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
    semi::EISPHSemidiscretization{NDIMS,ELTYPE,SummationDensity}) where {NDIMS,ELTYPE}
    return du
end

@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
    semi::WCSPHSemidiscretization{NDIMS,ELTYPE,SummationDensity}) where {NDIMS,ELTYPE}
    return du
end


@inline function get_particle_coords(u, semi, particle)
    return SVector(ntuple(@inline(dim -> u[dim, particle]), Val(ndims(semi))))
end

@inline function get_particle_vel(u, semi, particle)
    return SVector(ntuple(@inline(dim -> u[dim+ndims(semi), particle]), Val(ndims(semi))))
end

@inline function get_particle_density(u, cache, ::SummationDensity, particle)
    return cache.density[particle]
end

@inline function get_particle_density(u, cache, ::ContinuityDensity, particle)
    return u[end, particle]
end


@inline function get_particle_coords(boundary_container::BoundaryConditionMonaghanKajtar, semi, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(semi))))
end

@inline function get_particle_coords(boundary_container::BoundaryConditionCrespo, semi, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(semi))))
end

@inline function get_particle_coords(boundary_container::BoundaryConditionFixedParticleLiu, semi, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(semi))))
end


# This can be used both for Semidiscretization or boundary container types
@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline nparticles(semi) = length(semi.cache.mass)
@inline Base.ndims(::WCSPHSemidiscretization{NDIMS}) where {NDIMS} = NDIMS
@inline Base.ndims(::EISPHSemidiscretization{NDIMS}) where {NDIMS} = NDIMS
