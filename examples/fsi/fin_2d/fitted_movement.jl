
@inline function spectral_value(t, coefficients, frequency, period_start)
    theta = 2pi * frequency * (t - period_start)
    value = coefficients[1]

    for harmonic in 1:div(length(coefficients) - 1, 2)
        sine, cosine = sincos(harmonic * theta)
        value += coefficients[2 * harmonic] * cosine + coefficients[2 * harmonic + 1] * sine
    end

    return value
end

@inline function fitted_movement(frequency, period_start, translation_x_coefficients,
                                 translation_y_coefficients, rotation_coefficients, center)
    function movement(x, t)
        # Smooth startup matching the previous 0.5 s ramp.
        tau = clamp(t / 0.5, 0, 1)
        ramp = tau^3 * (10 + tau * (-15 + 6tau))

        translation = SVector(
            spectral_value(t, translation_x_coefficients, frequency, period_start),
            spectral_value(t, translation_y_coefficients, frequency, period_start),
        )

        angle = spectral_value(t, rotation_coefficients, frequency, period_start)

        sine, cosine = sincos(angle)
        relative_position = x - center
        rotated_position = SVector(
            cosine * relative_position[1] - sine * relative_position[2],
            sine * relative_position[1] + cosine * relative_position[2],
        )
        target_position = center + rotated_position + translation

        # Ramp the complete displacement, as done by `OscillatingMotion2D`.
        return x + ramp * (target_position - x)
    end

    return movement
end
