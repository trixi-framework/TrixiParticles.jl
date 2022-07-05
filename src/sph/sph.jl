struct ParticleContainer{uEltype<:Real}
    mass            ::Vector{uEltype}
    density         ::Vector{uEltype}
    pressure        ::Vector{uEltype}
    entropy         ::Vector{uEltype}
    smoothing_length::Vector{uEltype}
    rest_density    ::Vector{uEltype}
end


function compute_quantities!(u, container; compute_rest_density=false)
    @unpack mass, density, pressure, entropy, smoothing_length, rest_density = container

    for particle in axes(u, 2)
        h = 1 # smoothing length TODO
        r = SVector(u[1, particle], u[2, particle], u[3, particle])

        density[particle] = sum(axes(u, 2)) do neighbor
            distance = norm(r - SVector(u[1, neighbor], u[2, neighbor], u[3, neighbor]))
            return mass[neighbor] * smoothing_kernel(distance, h)
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


function semidiscretize(u0::Array{uEltype}, mass, tspan) where uEltype
    n_particles = size(u0, 2)

    density             = Vector{uEltype}(undef, n_particles)
    pressure            = Vector{uEltype}(undef, n_particles)
    entropy             = Vector{uEltype}(undef, n_particles)
    smoothing_length    = Vector{uEltype}(undef, n_particles)
    rest_density        = Vector{uEltype}(undef, n_particles)

    container = ParticleContainer{uEltype}(
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
    @unpack mass, density, pressure, entropy, smoothing_length = container

    compute_quantities!(u, container)

    # Boundary conditions
    for particle in axes(u, 2)
        r = SVector(u[1, particle], u[2, particle], u[3, particle])

        if r[3] < 0
            u[3, particle] = 0
            u[6, particle] *= -0.5
        end
    end

    # u[1:3] = coordinates
    # u[4:6] = velocity
    for particle in axes(du, 2)
        h = smoothing_length[particle] # TODO constant so far

        # dr = v
        du[1, particle] = u[4, particle]
        du[2, particle] = u[5, particle]
        du[3, particle] = u[6, particle]

        # dv (constant smoothing length, Price (31))
        r1 = SVector(u[1, particle], u[2, particle], u[3, particle])
        dv = -sum(axes(u, 2)) do neighbor
            m = mass[neighbor]
            r2 = SVector(u[1, neighbor], u[2, neighbor], u[3, neighbor])

            result = m * (pressure[particle] / density[particle]^2 +
                          pressure[neighbor] / density[neighbor]^2) *
                smoothing_kernel_der_r(norm(r1 - r2), h) * (r1 - r2)

            if norm(result) > eps()
                # Avoid dividing by zero
                # TODO The derivative does not exist for r1 = r2
                result /= norm(r1 - r2)
            end

            return result
        end

        du[4, particle] = dv[1]
        du[5, particle] = dv[2]
        du[6, particle] = dv[3]
    end
end
