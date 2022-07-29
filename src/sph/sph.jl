struct BoundaryParticleContainer{NDIMS, ELTYPE<:Real}
    coordinates     ::Array{ELTYPE, 2}
    mass            ::Vector{ELTYPE}
    spacing         ::Vector{ELTYPE} # 1/β in Monaghan, Kajtar (2009). TODO ELTYPE or hardcoded float?
end

struct SummationDensity end

struct ContinuityDensity end

struct SPHSemidiscretization{NDIMS, ELTYPE<:Real, DC, SE, K, V, C}
    boundaries          ::BoundaryParticleContainer{NDIMS, ELTYPE}
    density_calculator  ::DC
    state_equation      ::SE
    smoothing_kernel    ::K
    viscosity           ::V
    cache               ::C

    function SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                          boundary_masses, boundary_spacings,
                                          density_calculator, state_equation,
                                          smoothing_kernel;
                                          viscosity=NoViscosity()) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        boundaries = BoundaryParticleContainer{NDIMS, ELTYPE}(
            boundary_coordinates, boundary_masses, boundary_spacings
        )

        cache = (; create_cache(particle_masses, density_calculator, ELTYPE, nparticles)...)

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation), typeof(smoothing_kernel), typeof(viscosity), typeof(cache)}(
            boundaries, density_calculator, state_equation, smoothing_kernel, viscosity, cache)
    end

    function SPHSemidiscretization(particle_masses, boundary_coordinates,
                                   boundary_masses, boundary_spacings,
                                   density_calculator, state_equation,
                                   smoothing_kernel;
                                   viscosity=NoViscosity())
        NDIMS = size(boundary_coordinates, 1)

        if NDIMS == 0
            error("boundary_coordinates cannot be empty")
        end

        return SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                            boundary_masses, boundary_spacings,
                                            density_calculator, state_equation,
                                            smoothing_kernel,
                                            viscosity=viscosity)
    end

    function SPHSemidiscretization{NDIMS}(particle_masses, density_calculator, state_equation, smoothing_kernel;
                                          viscosity=NoViscosity()) where NDIMS
        boundary_coordinates = Array{Float64, 2}(undef, 0, 0)
        boundary_masses      = Vector{Float64}(undef, 0)
        boundary_spacings    = Vector{Float64}(undef, 0)

        return SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                            boundary_masses, boundary_spacings,
                                            density_calculator, state_equation,
                                            smoothing_kernel,
                                            viscosity=viscosity)
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
    @unpack boundaries = semi

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(semi), nparticles(semi))

    for particle in eachparticle(semi)
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        for dim in 1:ndims(semi)
            u0[dim + ndims(semi), particle] = particle_velocities[dim, particle]
        end
    end

    compute_quantities!(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function semidiscretize(semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity},
                        particle_coordinates, particle_velocities, particle_densities, tspan) where {NDIMS, ELTYPE}
    @unpack boundaries = semi

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(semi) + 1, nparticles(semi))

    for particle in eachparticle(semi)
        for dim in 1:ndims(semi)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        for dim in 1:ndims(semi)
            u0[dim + ndims(semi), particle] = particle_velocities[dim, particle]
        end

        u0[2 * ndims(semi) + 1, particle] = particle_densities[particle]
    end

    compute_quantities!(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function compute_quantities!(u, semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    @unpack smoothing_kernel, state_equation, cache = semi
    @unpack mass, density, pressure = cache

    h = 0.12 # smoothing length TODO

    for particle in eachparticle(semi)
        @pixie_timeit timer() "Compute density" begin
            density[particle] = sum(eachparticle(semi)) do neighbor
                distance = norm(get_particle_coords(u, semi, particle) -
                                get_particle_coords(u, semi, neighbor))

                if distance > 2 * h
                    return 0
                end

                return mass[neighbor] * kernel(smoothing_kernel, distance, h)
            end
        end

        pressure[particle] = state_equation(density[particle])
    end
end

function compute_quantities!(u, semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
    @unpack density_calculator, state_equation, cache = semi
    @unpack pressure = cache

    for particle in eachparticle(semi)
        pressure[particle] = state_equation(get_particle_density(u, cache, density_calculator, particle))
    end
end


function rhs!(du, u, semi, t)
    @unpack boundaries, smoothing_kernel, density_calculator, state_equation, viscosity, cache = semi

    @pixie_timeit timer() "rhs!" begin
        @unpack mass, pressure = cache

        compute_quantities!(u, semi)

        # u[1:3] = coordinates
        # u[4:6] = velocity
        for particle in eachparticle(semi)
            h = 0.12 # smoothing length TODO

            # dr = v
            for i in 1:ndims(semi)
                du[i, particle] = u[i + ndims(semi), particle]
            end

            # dv (constant smoothing length, Price (31))
            r1 = get_particle_coords(u, semi, particle)
            @pixie_timeit timer() "Compute dv" begin
                dv = sum(eachparticle(semi)) do neighbor
                    m = mass[neighbor]
                    r2 = get_particle_coords(u, semi, neighbor)

                    pos_diff = r1 - r2
                    distance = norm(pos_diff)

                    if eps() < distance <= compact_support(smoothing_kernel, h)
                        density_particle = get_particle_density(u, cache, density_calculator, particle)
                        density_neighbor = get_particle_density(u, cache, density_calculator, particle)

                        # Viscosity
                        v_diff = get_particle_vel(u, semi, particle) - get_particle_vel(u, semi, neighbor)
                        density_diff = (density_particle + density_neighbor) / 2
                        pi_ij = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                                          distance, density_diff, h)

                        result = -m * (pressure[particle] / density_particle^2 +
                                      pressure[neighbor] / density_neighbor^2 + pi_ij) *
                            kernel_deriv(smoothing_kernel, distance, h) * pos_diff / distance
                    else
                        # Don't compute pressure and density terms, just return zero
                        result = zeros(SVector{ndims(semi), eltype(pressure)})
                    end

                    return result
                end

                # Boundary conditions
                @pixie_timeit timer() "Boundary particles" begin
                    if length(boundaries.mass) > 0
                        dv += sum(eachparticle(boundaries)) do boundary_particle
                            pos_diff = get_particle_coords(u, semi, particle) -
                                       get_boundary_coords(boundaries, boundary_particle)
                            distance = norm(pos_diff)

                            if eps() < distance <= compact_support(smoothing_kernel, h)
                                m_b = boundaries.mass[boundary_particle]
                                K = 500 # TODO experiment with this constant

                                # Viscosity
                                v_diff = get_particle_vel(u, semi, particle)
                                pi_ij = viscosity(state_equation.sound_speed, v_diff, pos_diff, distance,
                                                  get_particle_density(u, cache, density_calculator, particle), h)

                                return K * boundaries.spacing[boundary_particle] * pos_diff / distance^2 *
                                    kernel(smoothing_kernel, distance, h) * 2 * m_b / (mass[particle] + m_b) -
                                    kernel_deriv(smoothing_kernel, distance, h) * m_b * pi_ij * pos_diff / distance
                            else
                                return zeros(SVector{ndims(semi), eltype(pressure)})
                            end
                        end
                    end
                end
            end

            for i in 1:ndims(semi)
                # Gravity
                du[i + ndims(semi), particle] = i == 2 ? dv[i] - 9.81 : dv[i]

                # du[i + ndims(semi), particle] = dv[i]
            end

        end

        continuity_equation!(du, u, semi)
    end

    return du
end


function continuity_equation!(du, u, semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
    @unpack smoothing_kernel, cache = semi
    @unpack mass = cache

    for particle in eachparticle(semi)
        h = 0.12 # smoothing length TODO

        r1 = get_particle_coords(u, semi, particle)
        @pixie_timeit timer() "Compute drho" begin
            du[2 * ndims(semi) + 1, particle] = sum(eachparticle(semi)) do neighbor
                m = mass[neighbor]
                r2 = get_particle_coords(u, semi, neighbor)

                diff = r1 - r2
                distance = norm(diff)

                if eps() < distance <= compact_support(smoothing_kernel, h)
                    vdiff = get_particle_vel(u, semi, particle) -
                            get_particle_vel(u, semi, neighbor)

                    result = sum(m * vdiff * kernel_deriv(smoothing_kernel, distance, h) .* diff) / distance
                else
                    # Don't compute pressure and density terms, just return zero
                    result = 0.0
                end

                return result
            end
        end
    end

    return du
end

function continuity_equation!(du, u, semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    return du
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


@inline function get_boundary_coords(boundary_container, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(boundary_container))))
end


@inline eachparticle(semi) = Base.OneTo(nparticles(semi))
@inline nparticles(semi) = length(semi.cache.mass)
@inline nparticles(boundaries::BoundaryParticleContainer) = length(boundaries.mass)
@inline Base.ndims(::SPHSemidiscretization{NDIMS}) where NDIMS = NDIMS
@inline Base.ndims(::BoundaryParticleContainer{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = NDIMS
