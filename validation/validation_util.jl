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

# Compute the MSE of spatial profiles (e.g., a velocity profile along a line)
# over all simulation time points, which must be contained in the reference time points.
# Entries that are `NaN` in both the reference and the simulation data
# (e.g., interpolation points outside the fluid domain) are skipped.
function profile_mse(reference_time, reference_values, simulation_time, simulation_values)
    sum_squared_error = 0.0
    n_values = 0

    for (t, values) in zip(simulation_time, simulation_values)
        i = findfirst(t_ref -> isapprox(t_ref, t), reference_time)
        if isnothing(i)
            throw(ArgumentError("simulation time $t is not contained in the reference data"))
        end

        for (value_ref, value) in zip(reference_values[i], values)
            isnan(value_ref) && isnan(value) && continue

            sum_squared_error += (value - value_ref)^2
            n_values += 1
        end
    end

    return sum_squared_error / n_values
end
