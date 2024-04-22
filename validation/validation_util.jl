# Perform linear interpolation to find a value at `interpolation_point` using arrays `x` and `y`.
#
# Arguments:
# - `x`                   : The array of abscissas (e.g., time points).
# - `y`                   : The array of ordinates (e.g., data points corresponding to `x`).
# - `interpolation_point` : The point at which to interpolate the data.
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

# Calculate the mean squared error (MSE) between interpolated simulation values and reference values
# over a common time range.
#
# Arguments:
# - `reference_time`    : Time points for the reference data.
# - `reference_values`  : Data points for the reference data.
# - `simulation_time`   : Time points for the simulation data.
# - `simulation_values` : Data points for the simulation data.
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

# Calculate the mean relative error (MRE) between interpolated simulation values and reference values
# over a common time range.
#
# Arguments:
# - `reference_time`    : Time points for the reference data.
# - `reference_values`  : Data points for the reference data.
# - `simulation_time`   : Time points for the simulation data.
# - `simulation_values` : Data points for the simulation data.
function interpolated_mre(reference_time, reference_values, simulation_time,
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

    # Calculate MRE only over the common time range (adding 10*eps() to prevent NaNs)
    relative_errors = abs.(interpolated_values .- filtered_values) ./
                      (abs.(filtered_values) + 10 * eps())
    valid_relative_errors = filter(!isnan, relative_errors)

    return sum(valid_relative_errors) / length(common_time_range)
end

function extract_number_from_filename(filename)
    # This regex matches the last sequence of digits in the filename
    m = match(r"(\d+)(?!.*\d)", filename)
    if m !== nothing
        return parse(Int, m.captures[1])
    end
    return -1
end

function extract_resolution_from_filename(str)
    str = match(r"\d+(?!.*\d)", str).match

    # Remove leading zeros and count them
    leading_zeros = length(match(r"^0*", str).match)
    str_non_zero = replace(str, r"^0*" => "")

    if isempty(str_non_zero)
        return "0.0"
    end

    # Adjust string to have a decimal point at the correct position
    if leading_zeros > 0
        decimal_str = "0." * "0"^(leading_zeros - 1) * str_non_zero
    else
        decimal_str = str_non_zero
    end

    # Convert integer strings to float strings
    return string(parse(Float64, decimal_str))
end
