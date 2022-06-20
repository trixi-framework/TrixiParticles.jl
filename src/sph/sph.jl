function semidiscretize(u0, tspan)
    n_particles = size(u0, 2)

    # For mass, entropy, density, pressure
    non_integrated_quantities = Array{Float64, 2}(undef, 4, n_particles)

    for particle in axes(u0, 2)
        non_integrated_quantities[1, particle] = 1 # mass TODO
        non_integrated_quantities[2, particle] = 1 # entropy TODO
    end

    return ODEProblem(rhs!, u0, tspan, non_integrated_quantities)
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



function rhs!(du, u, non_integrated_quantities, t)
    # Compute non-integrated quantites first
    for particle in axes(u, 2)
        r = SVector(u[1, particle], u[2, particle], u[3, particle])

        # Mass and entropy are assumed to be constant per particle
        # Density
        non_integrated_quantities[3, particle] = sum(axes(u, 2)) do neighbor
            distance = r - SVector(u[1, neighbor], u[2, neighbor], u[3, neighbor])
            m = non_integrated_quantities[1, neighbor]
            h = 1 # smoothing length TODO
            return m * smoothing_kernel(distance, h)
        end

        # Pressure
        gamma = 1.4
        non_integrated_quantities[4, particle] = (
            non_integrated_quantities[2, particle] *
            non_integrated_quantities[3, particle]^gamma)
    end

    # u[1:3] = coordinates
    # u[4:6] = velocity
    # u[7] = mass
    for particle in axes(du, 2)
        # dr = v
        du[1, particle] = u[4, particle]
        du[2, particle] = u[5, particle]
        du[3, particle] = u[6, particle]

        # dv (constant smoothing length, Price (31))
        du[4, particle] = -sum(axes(u, 2)) do neighbor
            m = non_integrated_quantities[1, neighbor]
            density1 = non_integrated_quantities[3, particle]
            density2 = non_integrated_quantities[3, neighbor]
            pressure1 = non_integrated_quantities[4, particle]
            pressure2 = non_integrated_quantities[4, neighbor]

            return # TODO what the hell is that derivative!?
        end
    end
end
