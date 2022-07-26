struct ParticleContainer{NDIMS, ELTYPE<:Real}
    mass            ::Vector{ELTYPE}
    density         ::Vector{ELTYPE}
    pressure        ::Vector{ELTYPE}
end

struct BoundaryParticleContainer{NDIMS, ELTYPE<:Real}
    coordinates     ::Array{ELTYPE, 2}
    mass            ::Vector{ELTYPE}
    spacing         ::Vector{ELTYPE} # 1/β in Monaghan, Kajtar (2009). TODO ELTYPE or hardcoded float?
end

struct SummationDensity end

struct ContinuityDensity end

struct SPHSemidiscretization{NDIMS, ELTYPE<:Real, DC, SE, K, V}
    particles           ::ParticleContainer{NDIMS, ELTYPE}
    boundaries          ::BoundaryParticleContainer{NDIMS, ELTYPE}
    density_calculator  ::DC
    state_equation      ::SE
    smoothing_kernel    ::K
    viscosity           ::V

    function SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                          boundary_masses, boundary_spacings,
                                          density_calculator, state_equation,
                                          smoothing_kernel;
                                          viscosity=NoViscosity()) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        density             = Vector{ELTYPE}(undef, nparticles)
        pressure            = Vector{ELTYPE}(undef, nparticles)

        particles = ParticleContainer{NDIMS, ELTYPE}(
            particle_masses, density, pressure
        )

        boundaries = BoundaryParticleContainer{NDIMS, ELTYPE}(
            boundary_coordinates, boundary_masses, boundary_spacings
        )

        return new{NDIMS, ELTYPE, typeof(density_calculator), typeof(state_equation), typeof(smoothing_kernel), typeof(viscosity)}(
            particles, boundaries, density_calculator, state_equation, smoothing_kernel, viscosity)
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


function semidiscretize(semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity},
                        particle_coordinates, particle_velocities, tspan) where {NDIMS, ELTYPE}
    @unpack particles, boundaries = semi

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(particles), nparticles(particles))

    for particle in eachparticle(particles)
        for dim in 1:ndims(particles)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        for dim in 1:ndims(particles)
            u0[dim + ndims(particles), particle] = particle_velocities[dim, particle]
        end
    end

    compute_quantities!(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function semidiscretize(semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity},
                        particle_coordinates, particle_velocities, particle_densities, tspan) where {NDIMS, ELTYPE}
    @unpack particles, boundaries = semi

    u0 = Array{eltype(particle_coordinates), 2}(undef, 2 * ndims(particles) + 1, nparticles(particles))

    for particle in eachparticle(particles)
        for dim in 1:ndims(particles)
            u0[dim, particle] = particle_coordinates[dim, particle]
        end

        for dim in 1:ndims(particles)
            u0[dim + ndims(particles), particle] = particle_velocities[dim, particle]
        end

        u0[2 * ndims(particles) + 1, particle] = particle_densities[particle]
    end

    compute_quantities!(u0, semi)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function compute_quantities!(u, semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    @unpack particles, smoothing_kernel, state_equation = semi
    @unpack mass, density, pressure = particles

    h = 0.12 # smoothing length TODO

    for particle in eachparticle(particles)
        @pixie_timeit timer() "Compute density" begin
            density[particle] = sum(eachparticle(particles)) do neighbor
                distance = norm(get_particle_coords(u, particles, particle) -
                                get_particle_coords(u, particles, neighbor))

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
    @unpack particles, state_equation = semi
    @unpack mass, density, pressure = particles

    for particle in eachparticle(particles)
        pressure[particle] = state_equation(u[2 * ndims(particles) + 1, particle])
    end
end


function rhs!(du, u, semi, t)
    @unpack particles, boundaries, smoothing_kernel, state_equation, viscosity = semi

    @pixie_timeit timer() "rhs!" begin
        @unpack mass, pressure = particles
        # TODO
        density = view(u, 2 * ndims(particles) + 1, :)

        compute_quantities!(u, semi)

        # u[1:3] = coordinates
        # u[4:6] = velocity
        for particle in eachparticle(particles)
            h = 0.12 # smoothing length TODO

            # dr = v
            for i in 1:ndims(particles)
                du[i, particle] = u[i + ndims(particles), particle]
            end

            # dv (constant smoothing length, Price (31))
            r1 = get_particle_coords(u, particles, particle)
            @pixie_timeit timer() "Compute dv" begin
                dv = sum(eachparticle(particles)) do neighbor
                    m = mass[neighbor]
                    r2 = get_particle_coords(u, particles, neighbor)

                    pos_diff = r1 - r2
                    distance = norm(pos_diff)

                    if eps() < distance <= 2 * h
                        # Viscosity
                        v_diff = get_particle_vel(u, particles, particle) -
                                 get_particle_vel(u, particles, neighbor)
                        density_diff = (density[particle] + density[neighbor]) / 2
                        pi_ij = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                                          distance, density_diff, h)

                        result = -m * (pressure[particle] / density[particle]^2 +
                                      pressure[neighbor] / density[neighbor]^2 + pi_ij) *
                            kernel_deriv(smoothing_kernel, distance, h) * pos_diff / distance
                    else
                        # Don't compute pressure and density terms, just return zero
                        result = zeros(SVector{ndims(particles), eltype(particles.density)})
                    end

                    return result
                end

                # Boundary conditions
                @pixie_timeit timer() "Boundary particles" begin
                    if length(boundaries.mass) > 0
                        dv += sum(eachparticle(boundaries)) do boundary_particle
                            pos_diff = get_particle_coords(u, particles, particle) -
                                       get_boundary_coords(boundaries, boundary_particle)
                            distance = norm(pos_diff)

                            if eps() < distance <= 2 * h
                                m_b = boundaries.mass[boundary_particle]
                                K = 500 # TODO experiment with this constant

                                # Viscosity
                                v_diff = get_particle_vel(u, particles, particle)
                                pi_ij = viscosity(state_equation.sound_speed, v_diff, pos_diff,
                                                  distance, density[particle], h)

                                return K * boundaries.spacing[boundary_particle] * pos_diff / distance^2 *
                                    kernel(smoothing_kernel, distance, h) * 2 * m_b / (mass[particle] + m_b) -
                                    kernel_deriv(smoothing_kernel, distance, h) * m_b * pi_ij * pos_diff / distance
                            else
                                return zeros(SVector{ndims(particles), eltype(particles.density)})
                            end
                        end
                    end
                end
            end

            for i in 1:ndims(particles)
                # Gravity
                du[i + ndims(particles), particle] = i == 2 ? dv[i] - 9.81 : dv[i]

                # du[i + ndims(particles), particle] = dv[i]
            end

        end

        continuity_equation!(du, u, semi)
    end

    return nothing
end


function continuity_equation!(du, u, semi::SPHSemidiscretization{NDIMS, ELTYPE, ContinuityDensity}) where {NDIMS, ELTYPE}
    @unpack particles, smoothing_kernel = semi
    @unpack mass = particles

    for particle in eachparticle(particles)
        h = 0.12 # smoothing length TODO

        r1 = get_particle_coords(u, particles, particle)
        @pixie_timeit timer() "Compute drho" begin
            du[2 * ndims(particles) + 1, particle] = sum(eachparticle(particles)) do neighbor
                m = mass[neighbor]
                r2 = get_particle_coords(u, particles, neighbor)

                diff = r1 - r2
                distance = norm(diff)

                if distance <= 2 * h
                    vdiff = get_particle_vel(u, particles, particle) -
                            get_particle_vel(u, particles, neighbor)

                    result = sum(m * vdiff * kernel_deriv(smoothing_kernel, distance, h) .* diff)

                    if abs(result) > eps()
                        # Avoid dividing by zero
                        # TODO The derivative does not exist for r1 = r2
                        result /= distance
                    end
                else
                    # Don't compute pressure and density terms, just return zero
                    result = 0.0
                end

                return result
            end
        end
    end

    return nothing
end

function continuity_equation!(du, u, semi::SPHSemidiscretization{NDIMS, ELTYPE, SummationDensity}) where {NDIMS, ELTYPE}
    return nothing
end


@inline function get_particle_coords(u, container, particle)
    SVector(ntuple(@inline(dim -> u[dim, particle]), Val(ndims(container))))
end

@inline function get_particle_vel(u, container, particle)
    SVector(ntuple(@inline(dim -> u[dim + ndims(container), particle]), Val(ndims(container))))
end


@inline function get_boundary_coords(boundary_container, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(boundary_container))))
end


@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline nparticles(container) = length(container.mass)
@inline Base.ndims(::ParticleContainer{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = NDIMS
@inline Base.ndims(::BoundaryParticleContainer{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = NDIMS
