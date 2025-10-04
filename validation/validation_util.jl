function linear_interpolation(x, y, interpolation_point)
    if !(first(x) <= interpolation_point <= last(x))
        throw(ArgumentError("`interpolation_point` at $interpolation_point is outside the interpolation range"))
    end

    i = searchsortedlast(x, interpolation_point)
    # Handle right boundary
    i == lastindex(x) && return last(y)

    # Linear interpolation
    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return y[i] + slope * (interpolation_point - x[i])
end

function interpolated_mse(reference_time, reference_values, simulation_time,
                          simulation_values)
    if last(simulation_time) > last(reference_time)
        @warn "simulation time range is larger than reference time range. " *
              "Only checking values within reference time range."
    end
    # Remove reference time points outside the simulation time
    start = searchsortedfirst(reference_time, first(simulation_time))
    end_ = searchsortedlast(reference_time, last(simulation_time))
    common_time_range = reference_time[start:end_]

    # Interpolate simulation data at the common time points
    interpolated_values = [linear_interpolation(simulation_time, simulation_values, t)
                           for t in common_time_range]

    filtered_values = reference_values[start:end_]

    # Calculate MSE only over the common time range
    mse = sum((interpolated_values .- filtered_values) .^ 2) / length(common_time_range)
    return mse
end

function extract_number_from_filename(filename)
    # This regex matches the last sequence of digits in the filename
    m = match(r"(\d+)(?!.*\d)", filename)
    if m !== nothing
        return parse(Int, m.captures[1])
    end
    return -1
end
