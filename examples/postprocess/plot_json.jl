function running_average(data::Vector{Float64}, window_size::Int)
    @assert window_size>=1 "Window size for running average must be >= 1"

    cum_sum = cumsum(data)
    cum_sum = vcat(zeros(window_size - 1), cum_sum)  # prepend zeros

    # See above for an explanation of the parameter choice
    sol = solve(ode, RDPK3SpFSAL49(),
                abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
                reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
                dtmax=1e-2, # Limit stepsize to prevent crashing
                save_everystep=false, callback=callbacks)
    averaged_data = (cum_sum[window_size:end] - cum_sum[1:(end - window_size + 1)]) /
                    window_size
    return averaged_data
end

using GLM
using DataFrames

function calculate_regression(data::Vector{Float64}, t::Vector{Float64})
    @assert length(data)==length(t) "Data and time vectors must have the same length"

    df = DataFrame(Y=data, T=t)
    model = lm(@formula(Y~T), df)  # Perform linear regression

    # Get the regression line values
    trend = predict(model, df)

    # Extract the gradient of the trend line
    gradient = coef(model)[2]

    return trend, gradient
end

function plot_json_data(dir_path::AbstractString="")
    if isempty(dir_path)
        dir_path = pwd()
    end

    files = readdir(dir_path)
    json_files = filter(x -> occursin(r"^values(\d*).json$", x), files)

    if isempty(json_files)
        println("No matching JSON files found.")
        return
    end

    recent_file = sort(json_files, by=x -> stat(joinpath(dir_path, x)).ctime, rev=true)[1]
    file_path = joinpath(dir_path, recent_file)

    json_string = read(file_path, String)
    json_data = JSON.parse(json_string)

    dt_series = json_data["dt"]
    dt_values = Vector{Float64}(dt_series["values"])
    accumulated_dt = cumsum(dt_values)

    function plot_dt()
        xlabel("T")
        ylabel("dt")
        title("dt")
        yscale("log")
        scatter(accumulated_dt, dt_values, s=4, color="blue", label="dt")
        legend()
    end

    subplot_number = 1
    for key in ["dp", "ekin"]
        if haskey(json_data, key)
            subplot_number += 1
        end
    end

    figure(figsize=(10, 5 * subplot_number))

    subplot(subplot_number, 1, 1)
    plot_dt()

    subplot_index = 2
    for key in ["dp", "ekin"]
        if haskey(json_data, key)
            series = json_data[key]
            values = Vector{Float64}(series["values"])

            subplot(subplot_number, 1, subplot_index)
            xlabel("T")
            ylabel(key)
            title(key)
            scatter(accumulated_dt, values, s=4, label=key)

            if key == "ekin"  # calculate and plot running average for ekin
                window_size = 50
                averaged_ekin = running_average(values, window_size)
                # println("Length of accumulated_dt: ", length(accumulated_dt))
                # println("Length of averaged_ekin: ", length(averaged_ekin))
                # println("Window size: ", window_size)
                scatter(accumulated_dt, averaged_ekin, s=4, color="orange",
                        label="ekin (run. avg. windowsize = $window_size)")

                trend_line, gradient = calculate_regression(values, accumulated_dt)
                plot(accumulated_dt, trend_line, label="trend (gradient=$gradient)",
                     color="red", linewidth=2)
            end

            legend()
            subplot_index += 1
        end
    end

    show()
end

plot_json_data()
