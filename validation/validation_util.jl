using Interpolations
function calculate_mse(reference_data, simulation_data)
    # Interpolate simulation data
    interp_func = LinearInterpolation(simulation_data["time"], simulation_data["values"])

    # Align with reference data time points
    interpolated_values = interp_func(reference_data.time)

    # Calculate MSE
    mse = mean((interpolated_values .- reference_data.displacement) .^ 2)
    return mse
end
