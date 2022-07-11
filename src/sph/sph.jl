struct ParticleContainer{NDIMS, ELTYPE<:Real}
    mass            ::Vector{ELTYPE}
    density         ::Vector{ELTYPE}
    pressure        ::Vector{ELTYPE}
    entropy         ::Vector{ELTYPE}
    smoothing_length::Vector{ELTYPE}
    rest_density    ::Vector{ELTYPE}
end

struct BoundaryParticleContainer{NDIMS, ELTYPE<:Real}
    coordinates     ::Array{ELTYPE, 2}
    mass            ::Vector{ELTYPE}
    spacing         ::Vector{ELTYPE} # 1/β in Monaghan, Kajtar (2009). TODO ELTYPE or hardcoded float?
end


struct SPHSemidiscretization{NDIMS, ELTYPE<:Real}
    particles       ::ParticleContainer{NDIMS, ELTYPE}
    boundaries      ::BoundaryParticleContainer{NDIMS, ELTYPE}

    function SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                          boundary_masses, boundary_spacings) where NDIMS
        ELTYPE = eltype(particle_masses)
        nparticles = length(particle_masses)

        density             = Vector{ELTYPE}(undef, nparticles)
        pressure            = Vector{ELTYPE}(undef, nparticles)
        entropy             = Vector{ELTYPE}(undef, nparticles)
        smoothing_length    = Vector{ELTYPE}(undef, nparticles)
        rest_density        = Vector{ELTYPE}(undef, nparticles)

        particles = ParticleContainer{NDIMS, ELTYPE}(
            particle_masses, density, pressure, entropy, smoothing_length, rest_density
        )

        boundaries = BoundaryParticleContainer{NDIMS, ELTYPE}(
            boundary_coordinates, boundary_masses, boundary_spacings
        )

        return new{NDIMS, ELTYPE}(
            particles, boundaries)
    end

    function SPHSemidiscretization(particle_masses, boundary_coordinates,
                                   boundary_masses, boundary_spacings)
        NDIMS = size(boundary_coordinates, 1)

        if NDIMS == 0
            error("boundary_coordinates cannot be empty")
        end

        return SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                            boundary_masses, boundary_spacings)
    end

    function SPHSemidiscretization{NDIMS}(particle_masses) where NDIMS
        boundary_coordinates = Array{Float64, 2}(undef, 0, 0)
        boundary_masses      = Vector{Float64}(undef, 0)
        boundary_spacings    = Vector{Float64}(undef, 0)

        return SPHSemidiscretization{NDIMS}(particle_masses, boundary_coordinates,
                                            boundary_masses, boundary_spacings)
    end
end


function semidiscretize(semi, particle_coordinates, particle_velocities, tspan)
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

    compute_quantities!(u0, particles, compute_rest_density=true)

    return ODEProblem(rhs!, u0, tspan, semi)
end


function compute_quantities!(u, container; compute_rest_density=false)
    @unpack mass, density, pressure, entropy, smoothing_length, rest_density = container

    for particle in eachparticle(container)
        h = 1 # smoothing length TODO

        @pixie_timeit timer() "Compute density" begin
            density[particle] = sum(eachparticle(container)) do neighbor
                distance = norm(get_particle_coords(u, container, particle) -
                                get_particle_coords(u, container, neighbor))
                return mass[neighbor] * smoothing_kernel(distance, h)
            end
        end

        if compute_rest_density
            rest_density[particle] = density[particle]
        end

        entropy[particle] = 1

        gamma = 1
        # pressure[particle] = entropy[particle] * (density[particle] - rest_density[particle])^gamma
        pressure[particle] = rest_density[particle] * 10^2 / 7 * ((density[particle]/rest_density[particle])^7 - 1)

        smoothing_length[particle] = h
    end
end


function smoothing_kernel(r, h)
    q = 0.5 * r / h

    if q > 1
        return 0
    elseif 0.5 <= q <= 1
        return 2 * (1 - q)^3
    else
        return 1 - 6 * q^2 + 6 * q^3
    end
end


function smoothing_kernel_der_r(r, h)
    inner_deriv = 0.5 / h
    q = 0.5 * r / h

    if q > 1
        return 0
    elseif 0.5 <= q <= 1
        return -2 * 3 * (1 - q)^2 * inner_deriv
    else
        return -12 * q + 18 * q^2 * inner_deriv
    end
end


function rhs!(du, u, semi, t)
    @unpack particles, boundaries = semi

    @pixie_timeit timer() "rhs!" begin
        @unpack mass, density, pressure, entropy, smoothing_length = particles

        compute_quantities!(u, particles)

        # u[1:3] = coordinates
        # u[4:6] = velocity
        for particle in eachparticle(particles)
            h = smoothing_length[particle] # TODO constant so far

            # dr = v
            for i in 1:ndims(particles)
                du[i, particle] = u[i + ndims(particles), particle]
            end

            # dv (constant smoothing length, Price (31))
            r1 = get_particle_coords(u, particles, particle)
            @pixie_timeit timer() "Compute dv" begin
                dv = -sum(eachparticle(particles)) do neighbor
                    m = mass[neighbor]
                    r2 = get_particle_coords(u, particles, neighbor)

                    kernel_result = smoothing_kernel_der_r(norm(r1 - r2), h)

                    if kernel_result != 0
                        result = m * (pressure[particle] / density[particle]^2 +
                                      pressure[neighbor] / density[neighbor]^2) *
                            kernel_result * (r1 - r2)
                    else
                        # Don't compute pressure and density terms, just return zero
                        result = zeros(SVector{ndims(particles), eltype(particles.density)})
                    end

                    if norm(result) > eps()
                        # Avoid dividing by zero
                        # TODO The derivative does not exist for r1 = r2
                        result /= norm(r1 - r2)
                    end

                    return result
                end

                # Boundary conditions
                @pixie_timeit timer() "Boundary particles" begin
                    if length(boundaries.mass) > 0
                        dv += sum(eachparticle(boundaries)) do boundary_particle
                            diff = get_particle_coords(u, particles, particle) -
                                get_boundary_coords(boundaries, boundary_particle)
                            distance = norm(diff)

                            kernel_result = smoothing_kernel(distance, h)

                            if kernel_result != 0
                                m_b = boundaries.mass[boundary_particle]
                                K = 100 # TODO experiment with this constant
                                return K * boundaries.spacing[boundary_particle] * diff / distance^2 *
                                    kernel_result * 2 * m_b / (mass[particle] + m_b)
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
            end
        end
    end
end


@inline function get_particle_coords(u, container, particle)
    SVector(ntuple(@inline(dim -> u[dim, particle]), Val(ndims(container))))
end


@inline function get_boundary_coords(boundary_container, particle)
    @unpack coordinates = boundary_container
    SVector(ntuple(@inline(dim -> coordinates[dim, particle]), Val(ndims(boundary_container))))
end


@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline nparticles(container) = length(container.mass)
@inline Base.ndims(::ParticleContainer{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = NDIMS
@inline Base.ndims(::BoundaryParticleContainer{NDIMS, ELTYPE}) where {NDIMS, ELTYPE} = NDIMS
