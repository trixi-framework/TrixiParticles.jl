using Interpolations
using Statistics

function calculate_mse(reference_data, simulation_time, simulation_values)
    interp_func = LinearInterpolation(simulation_time, simulation_values,
                                      extrapolation_bc=Flat())

    # Find the common time range
    common_time_range = filter(t -> t >= maximum([
                                                minimum(simulation_time),
                                                minimum(reference_data.time),
                                            ]) &&
                                   t <= minimum([
                                               maximum(simulation_time),
                                               maximum(reference_data.time),
                                           ]), reference_data.time)

    # Interpolate simulation data at the common time points
    interpolated_values = [interp_func(t) for t in common_time_range]

    # Extract the corresponding reference displacement values
    reference_displacement = [reference_data.displacement[findfirst(==(t),
                                                                    reference_data.time)]
                              for t in common_time_range]

    # Calculate MSE only over the common time range
    mse = mean((interpolated_values .- reference_displacement) .^ 2)
    return mse
end

function extract_number(filename)
    # This regex matches the last sequence of digits in the filename
    m = match(r"(\d+)(?!.*\d)", filename)
    if m !== nothing
        num = parse(Int, m.captures[1])
        return num
    else
        println("No numeric sequence found in filename: ", filename)
        return -1
    end
end
