function compute_quantities(u, semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    @threaded for particle in eachparticle(semi)
        compute_quantities_per_particle(u, particle, semi)
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function compute_quantities_per_particle(u, particle, semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    @unpack smoothing_kernel, smoothing_length, boundary_conditions, state_equation,
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

    # Include boundary particles in the summation
    for bc in boundary_conditions
        compute_boundary_density!(density, u, particle, bc, semi)
    end

    pressure[particle] = state_equation(density[particle])
end

# Include the boundary particles in the density summation
@inline function compute_boundary_density!(density, u, particle, bc, semi)
    return density
end

@inline function compute_boundary_density!(density, u, particle, bc::BoundaryParticlesFrozen, semi)
    @unpack smoothing_kernel, smoothing_length = semi
    @unpack mass, neighborhood_search = bc

    for neighbor in eachneighbor(particle, u, neighborhood_search, semi, particles=eachparticle(bc))
        distance = norm(get_particle_coords(u, semi, particle) -
                        get_particle_coords(bc, semi, neighbor))

        if distance <= compact_support(smoothing_kernel, smoothing_length)
            density[particle] += mass[neighbor] * kernel(smoothing_kernel, distance, smoothing_length)
        end
    end

    return density
end

function compute_quantities(u, semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
    @unpack density_calculator, state_equation, cache = semi
    @unpack pressure = cache

    # Note that @threaded makes this slower with ContinuityDensity
    for particle in eachparticle(semi)
        pressure[particle] = state_equation(get_particle_density(u, cache, density_calculator, particle))
    end
end


function rhs!(du, u, semi::SPHFluidSemidiscretization, t)
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
                du[i, particle] = u[i + ndims(semi), particle]
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

            calc_gravity!(du, particle, semi)

            # boundary impact
            for bc in boundary_conditions
                calc_boundary_condition_per_particle!(du, u, particle, bc, semi)
            end
        end
    end

    return du
end


@inline function calc_dv!(du, u, particle, neighbor, pos_diff, distance, semi)
    @unpack smoothing_kernel, smoothing_length,
            density_calculator, state_equation, viscosity, cache = semi
    @unpack mass, pressure = cache

    density_particle = get_particle_density(u, cache, density_calculator, particle)
    density_neighbor = get_particle_density(u, cache, density_calculator, neighbor)

    # Viscosity
    v_diff = get_particle_vel(u, semi, particle) - get_particle_vel(u, semi, neighbor)
    density_mean = (density_particle + density_neighbor) / 2
    pi_ab = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                      distance, density_mean, smoothing_length)

    grad_kernel = kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance
    m_b = mass[neighbor]
    dv_pressure = -m_b * (pressure[particle] / density_particle^2 +
                          pressure[neighbor] / density_neighbor^2) * grad_kernel
    dv_viscosity = m_b * pi_ab * grad_kernel

    dv = dv_pressure + dv_viscosity

    for i in 1:ndims(semi)
        du[ndims(semi) + i, particle] += dv[i]
    end

    return du
end


@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
                                      semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
    @unpack smoothing_kernel, smoothing_length, cache = semi
    @unpack mass = cache

    vdiff = get_particle_vel(u, semi, particle) -
            get_particle_vel(u, semi, neighbor)

    du[2 * ndims(semi) + 1, particle] += sum(mass[particle] * vdiff *
                                             kernel_deriv(smoothing_kernel, distance, smoothing_length) .*
                                             pos_diff) / distance

    return du
end

@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
                                      semi::SPHFluidSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    return du
end


# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function calc_boundary_condition_per_particle!(du, u, particle,
                                                       boundary_condition,
                                                       semi)
    @unpack smoothing_kernel, smoothing_length,
            density_calculator, state_equation, viscosity, cache = semi
    @unpack mass, neighborhood_search = boundary_condition

    m_a = cache.mass[particle]
    for boundary_particle in eachneighbor(particle, u, neighborhood_search, semi, particles=eachparticle(boundary_condition))
        pos_diff = get_particle_coords(u, semi, particle) -
                   get_particle_coords(boundary_condition, semi, boundary_particle)
        distance = norm(pos_diff)

        if eps() < distance <= compact_support(smoothing_kernel, smoothing_length)
            m_b = mass[boundary_particle]

            dv = boundary_particle_impact(boundary_condition, semi, u, particle, distance, pos_diff, m_a, m_b)

            for i in 1:ndims(semi)
                du[ndims(semi) + i, particle] += dv[i]
            end
        end
    end
end
