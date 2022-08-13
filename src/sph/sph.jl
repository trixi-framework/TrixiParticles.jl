struct SummationDensity end

struct ContinuityDensity end

struct SPHSemidiscretization{NDIMS, ELTYPE<:Real, DC, SE, K, V, BC, NS, C}
    density_calculator  ::DC
    state_equation      ::SE
    smoothing_kernel    ::K
    smoothing_length    ::ELTYPE
    viscosity           ::V
    boundary_conditions ::BC
    gravity             ::SVector{NDIMS, ELTYPE}
    neighborhood_search ::NS
    cache               ::C

    function SPHSemidiscretization{NDIMS}(particle_masses,
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
            viscosity, boundary_conditions_,  gravity_, neighborhood_search, cache)
    end
end


function create_cache(mass, density_calculator, eltype, nparticles)
    pressure = Vector{eltype}(undef, nparticles)

    return (; mass, pressure, create_cache(density_calculator, eltype, nparticles)...)
end

function create_cache(::SummationDensity, eltype, nparticles)
    density = Vector{eltype}(undef, nparticles)

    return (; density)
end

function create_cache(::ContinuityDensity, eltype, nparticles)
    return (; )
end


function semidiscretize(semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity},
                        particle_coordinates, particle_velocities, tspan) where {NDIMS, ELTYPE}

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

    # Compute quantities like density and pressure
    compute_quantities(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function semidiscretize(semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity},
                        particle_coordinates, particle_velocities, particle_densities, tspan) where {NDIMS, ELTYPE}

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

    # Compute quantities like pressure
    compute_quantities(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function compute_quantities(u, semi)
    @threaded for particle in eachparticle(semi)
        compute_quantities_per_particle(u, particle, semi)
    end
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
function compute_quantities_per_particle(u, particle, semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
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

function compute_quantities_per_particle(u, particle, semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
    @unpack density_calculator, state_equation, cache = semi
    @unpack pressure = cache

    pressure[particle] = state_equation(get_particle_density(u, cache, density_calculator, particle))
end


function rhs!(du, u, semi, t)
    @unpack smoothing_kernel, smoothing_length,
            boundary_conditions, gravity,
            neighborhood_search = semi

    @pixie_timeit timer() "rhs!" begin
        @pixie_timeit timer() "initialize neighborhood search" initialize!(neighborhood_search, u, semi)

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

            for i in 1:ndims(semi)
                # Gravity
                du[i + ndims(semi), particle] += gravity[i]
            end
        end

        # Boundary conditions
        @pixie_timeit timer() "Boundary conditions" for bc in boundary_conditions
            calc_boundary_condition!(du, u, bc, semi)
        end
    end

    return du
end


@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end


@inline function calc_dv!(du, u, particle, neighbor, pos_diff, distance, semi)
    @unpack smoothing_kernel, smoothing_length,
            density_calculator, state_equation, viscosity, cache = semi
    @unpack mass, pressure = cache

    density_particle = get_particle_density(u, cache, density_calculator, particle)
    density_neighbor = get_particle_density(u, cache, density_calculator, particle)

    # Viscosity
    v_diff = get_particle_vel(u, semi, particle) - get_particle_vel(u, semi, neighbor)
    density_diff = (density_particle + density_neighbor) / 2
    pi_ij = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                        distance, density_diff, smoothing_length)

    dv = -mass[particle] * (pressure[particle] / density_particle^2 +
                            pressure[neighbor] / density_neighbor^2 + pi_ij) *
        kernel_deriv(smoothing_kernel, distance, smoothing_length) * pos_diff / distance

    for i in 1:ndims(semi)
        du[ndims(semi) + i, particle] += dv[i]
    end

    return du
end


@inline function continuity_equation!(du, u, particle, neighbor, pos_diff, distance,
                              semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
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
                                      semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    return du
end


function calc_boundary_condition!(du, u, boundary_condition::BoundaryConditionMonaghanKajtar, semi)
    @threaded for particle in eachparticle(semi)
        calc_boundary_condition_per_particle!(du, u, particle, boundary_condition, semi)
    end

    return du
end

# Use this function barrier and unpack inside to avoid passing closures to Polyester.jl with @batch (@threaded).
# Otherwise, @threaded does not work here with Julia ARM on macOS.
# See https://github.com/JuliaSIMD/Polyester.jl/issues/88.
@inline function calc_boundary_condition_per_particle!(du, u, particle,
                                                       boundary_condition::BoundaryConditionMonaghanKajtar,
                                                       semi)
    @unpack smoothing_kernel, smoothing_length,
        density_calculator, state_equation, viscosity, cache = semi
    @unpack K, coordinates, mass, spacing = boundary_condition

    for boundary_particle in eachparticle(boundary_condition)
        pos_diff = get_particle_coords(u, semi, particle) -
                   get_boundary_coords(boundary_condition, semi, boundary_particle)
        distance = norm(pos_diff)

        if eps() < distance <= compact_support(smoothing_kernel, smoothing_length)
            # Viscosity
            v_diff = get_particle_vel(u, semi, particle)
            pi_ij = viscosity(state_equation.sound_speed, v_diff, pos_diff, distance,
                              get_particle_density(u, cache, density_calculator, particle), smoothing_length)

            m_b = mass[boundary_particle]

            dv = K * spacing[boundary_particle] * pos_diff / distance^2 *
                kernel(smoothing_kernel, distance, smoothing_length) * 2 * m_b / (cache.mass[particle] + m_b) -
                kernel_deriv(smoothing_kernel, distance, smoothing_length) * m_b * pi_ij * pos_diff / distance

            for i in 1:ndims(semi)
                du[ndims(semi) + i, particle] += dv[i]
            end
        end
    end
end


@inline function get_particle_coords(u, semi, particle)
    return SVector(ntuple(@inline(dim -> u[dim, particle]), Val(ndims(semi))))
end

@inline function get_particle_vel(u, semi, particle)
    return SVector(ntuple(@inline(dim -> u[dim + ndims(semi), particle]), Val(ndims(semi))))
end


@inline function get_particle_density(u, cache, ::SummationDensity, particle)
    return cache.density[particle]
end

@inline function get_particle_density(u, cache, ::ContinuityDensity, particle)
    return u[end, particle]
end


@inline function get_boundary_coords(boundary_container, semi, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(semi))))
end


# This can be used both for Semidiscretization or boundary container types
@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline nparticles(semi) = length(semi.cache.mass)
@inline Base.ndims(::SPHSemidiscretization{NDIMS}) where NDIMS = NDIMS
