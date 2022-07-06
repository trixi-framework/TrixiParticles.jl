struct ParticleContainer{ELTYPE<:Real, NDIMS}
    mass            ::Vector{ELTYPE}
    density         ::Vector{ELTYPE}
    pressure        ::Vector{ELTYPE}
    entropy         ::Vector{ELTYPE}
    smoothing_length::Vector{ELTYPE}
    rest_density    ::Vector{ELTYPE}
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
        pressure[particle] = entropy[particle] * (density[particle] - rest_density[particle])^gamma

        smoothing_length[particle] = h
    end
end


function semidiscretize(u0::Array{ELTYPE}, mass, tspan) where ELTYPE
    nparticles = size(u0, 2)
    ndims = size(u0, 1) ÷ 2

    density             = Vector{ELTYPE}(undef, nparticles)
    pressure            = Vector{ELTYPE}(undef, nparticles)
    entropy             = Vector{ELTYPE}(undef, nparticles)
    smoothing_length    = Vector{ELTYPE}(undef, nparticles)
    rest_density        = Vector{ELTYPE}(undef, nparticles)

    container = ParticleContainer{ELTYPE, ndims}(
        mass, density, pressure, entropy, smoothing_length, rest_density
    )

    compute_quantities!(u0, container, compute_rest_density=true)

    return ODEProblem(rhs!, u0, tspan, container)
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


function rhs!(du, u, container, t)
    @pixie_timeit timer() "rhs!" begin
        @unpack mass, density, pressure, entropy, smoothing_length = container

        compute_quantities!(u, container)

        # Boundary conditions
        for particle in eachparticle(container)
            r = get_particle_coords(u, container, particle)

            if r[1] < 0
                u[1, particle] = 0
                u[length(r) + 1, particle] *= -0.5
            end
        end

        # u[1:3] = coordinates
        # u[4:6] = velocity
        for particle in eachparticle(container)
            h = smoothing_length[particle] # TODO constant so far

            # dr = v
            for i in 1:ndims(container)
                du[i, particle] = u[i + ndims(container), particle]
            end

            # dv (constant smoothing length, Price (31))
            r1 = get_particle_coords(u, container, particle)
            @pixie_timeit timer() "Compute dv" begin
                dv = -sum(eachparticle(container)) do neighbor
                    m = mass[neighbor]
                    r2 = get_particle_coords(u, container, neighbor)

                    kernel_result = smoothing_kernel_der_r(norm(r1 - r2), h)

                    if kernel_result != 0
                        result = m * (pressure[particle] / density[particle]^2 +
                                      pressure[neighbor] / density[neighbor]^2) *
                            kernel_result * (r1 - r2)
                    else
                        # Don't compute pressure and density terms, just return zero
                        result = zeros(SVector{ndims(container), eltype(container.density)})
                    end

                    if norm(result) > eps()
                        # Avoid dividing by zero
                        # TODO The derivative does not exist for r1 = r2
                        result /= norm(r1 - r2)
                    end

                    return result
                end
            end

            for i in 1:ndims(container)
                du[i + ndims(container), particle] = dv[i]
            end
        end
    end
end


@inline function get_particle_coords(u, container, particle)
    SVector(ntuple(@inline(dim -> u[dim, particle]), Val(ndims(container))))
end


@inline eachparticle(container) = Base.OneTo(nparticles(container))
@inline nparticles(container) = length(container.density)
@inline Base.ndims(::ParticleContainer{ELTYPE, NDIMS}) where {ELTYPE, NDIMS} = NDIMS
