function linear_interpolation_clamped(x, y, interpolation_point)
    interpolation_point <= first(x) && return first(y)
    interpolation_point >= last(x) && return last(y)

    i = searchsortedlast(x, interpolation_point)
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    return y0 + (y1 - y0) * (interpolation_point - x0) / (x1 - x0)
end

function carreau_yasuda_kinematic_viscosity(shear_rate, nu0, nu_inf,
                                            time_constant, lambda_exponent,
                                            power_law_index)
    return nu_inf +
           (nu0 - nu_inf) *
           (1.0 + (time_constant * shear_rate)^lambda_exponent)^((power_law_index -
                                                                  1.0) /
                                                                 lambda_exponent)
end

function solve_shear_rate_from_stress(shear_stress, density, nu0, nu_inf,
                                      time_constant, lambda_exponent,
                                      power_law_index)
    shear_stress <= 0 && return 0.0

    residual(shear_rate) = density *
                           carreau_yasuda_kinematic_viscosity(shear_rate, nu0,
                                                              nu_inf,
                                                              time_constant,
                                                              lambda_exponent,
                                                              power_law_index) *
                           shear_rate - shear_stress

    lower = 0.0
    upper = 1.0
    while residual(upper) < 0.0
        upper *= 2.0
        upper > 1.0e12 &&
            error("failed to bracket shear-rate root for shear stress $shear_stress")
    end

    for _ in 1:120
        middle = 0.5 * (lower + upper)
        residual_middle = residual(middle)

        if abs(residual_middle) <= 1.0e-12 * max(shear_stress, 1.0)
            return middle
        elseif residual_middle > 0
            upper = middle
        else
            lower = middle
        end
    end

    return 0.5 * (lower + upper)
end

function analytical_ux_profile(y_positions, power_law_index, channel_height,
                               density, nu0, nu_inf, time_constant,
                               lambda_exponent, pressure_gradient)
    distances_to_centerline = sort(unique(abs.(y_positions .- 0.5 * channel_height)))
    shear_rates = similar(distances_to_centerline)

    for i in eachindex(distances_to_centerline)
        shear_stress = pressure_gradient * distances_to_centerline[i]
        shear_rates[i] = solve_shear_rate_from_stress(shear_stress, density, nu0,
                                                      nu_inf, time_constant,
                                                      lambda_exponent,
                                                      power_law_index)
    end

    velocity_at_distance = zeros(length(distances_to_centerline))
    for i in (lastindex(distances_to_centerline) - 1):-1:firstindex(distances_to_centerline)
        ds = distances_to_centerline[i + 1] - distances_to_centerline[i]
        velocity_at_distance[i] = velocity_at_distance[i + 1] +
                                  0.5 * (shear_rates[i + 1] + shear_rates[i]) * ds
    end

    velocity = Vector{Float64}(undef, length(y_positions))
    for (i, y) in pairs(y_positions)
        distance_to_centerline = abs(y - 0.5 * channel_height)
        velocity[i] = linear_interpolation_clamped(distances_to_centerline,
                                                   velocity_at_distance,
                                                   distance_to_centerline)
    end

    return velocity
end

function velocity_profile_errors(numerical_velocity, analytical_velocity)
    squared_error = sum((numerical_velocity .- analytical_velocity) .^ 2)
    squared_reference = sum(analytical_velocity .^ 2)
    relative_l2_error = sqrt(squared_error / length(numerical_velocity)) /
                        (sqrt(squared_reference / length(analytical_velocity)) + eps())
    max_velocity_error = maximum(abs.(numerical_velocity .- analytical_velocity))

    return relative_l2_error, max_velocity_error
end
