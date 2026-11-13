function velocity_profile_errors(numerical_velocity, analytical_velocity)
    squared_error = sum((numerical_velocity .- analytical_velocity) .^ 2)
    squared_reference = sum(analytical_velocity .^ 2)
    relative_l2_error = sqrt(squared_error / length(numerical_velocity)) /
                        (sqrt(squared_reference / length(analytical_velocity)) + eps())
    max_velocity_error = maximum(abs.(numerical_velocity .- analytical_velocity))

    return relative_l2_error, max_velocity_error
end
