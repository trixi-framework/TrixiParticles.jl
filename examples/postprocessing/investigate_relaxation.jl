using TrixiParticles
using JSON
using PythonPlot
using GLM
using DataFrames
using Printf

# Any function can be implemented and will be called after each timestep! See example below:
# a = function(pp, t, system, u, v, system_name) println("test_func ", t) end
# example_cb = PostprocessCallback([a,])

# see also the implementation for the functions calculate_ekin, calculate_total_mass,...
# pp_cb = PostprocessCallback([calculate_ekin, max_pressure, avg_density], interval=10)
pp_cb = PostprocessCallback([calculate_ekin, max_pressure, avg_density], dt=0.01)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"),
              pp_callback=pp_cb, tspan=(0.0, 0.1));

function calculate_regression(data::Vector{Float64}, t::Vector{Float64})
    @assert length(data)==length(t) "Data and time vectors must have the same length"

    df = DataFrame(Y=data, T=t)
    # Perform linear regression
    model = lm(@formula(Y~T), df)

    # Get the regression line values
    trend = predict(model, df)

    # Extract the gradient of the trend line
    gradient = coef(model)[2]

    return trend, gradient
end

file_path = TrixiParticles.get_latest_unique_filename(pwd(), "values", ".json")
if file_path != ""
    json_string = read(file_path, String)
    json_data = JSON.parse(json_string)

    time = Vector{Float64}(json_data["ekin_fluid_1"]["time"])
    ekin = Vector{Float64}(json_data["ekin_fluid_1"]["values"])
    tl_ekin, grad_ekin = calculate_regression(ekin, time)

    p_max = Vector{Float64}(json_data["max_p_fluid_1"]["values"])
    tl_p_max, grad_p_max = calculate_regression(p_max, time)

    avg_rho = Vector{Float64}(json_data["avg_rho_fluid_1"]["values"])
    tl_avg_rho, grad_avg_rho = calculate_regression(avg_rho, time)

    fig, (subplot1, subplot2, subplot3) = subplots(1, 3, figsize=(18, 5))

    subplot1.plot(time, ekin, linestyle="-", label="sim")
    subplot1.plot(time, tl_ekin, label="trend (gradient=$(@sprintf("%.5f", grad_ekin)))",
                  color="red", linewidth=2)
    subplot1.set_xlabel("Time [s]")
    subplot1.set_ylabel("kinetic energy [J]")
    subplot1.set_title("Kinetic Energy of the Fluid")
    subplot1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=true,
                    shadow=true)

    subplot2.plot(time, p_max, linestyle="-", label="sim")
    subplot2.plot(time, tl_p_max, label="trend (gradient=$(@sprintf("%.5f", grad_p_max)))",
                  color="red", linewidth=2)
    subplot2.set_xlabel("Time [s]")
    subplot2.set_ylabel("Max. Pressure [Pa]")
    subplot2.set_title("Maximum Pressure of the Fluid")
    subplot2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=true,
                    shadow=true)

    subplot3.plot(time, avg_rho, linestyle="-", label="With Endpoint")
    subplot3.plot(time, tl_avg_rho,
                  label="trend (gradient=$(@sprintf("%.5f", grad_avg_rho)))",
                  color="red", linewidth=2)
    subplot3.set_xlabel("Time [s]")
    subplot3.set_ylabel("Avg. Density [kg/m^3]")
    subplot3.set_title("Avg. Density of the Fluid")
    subplot3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fancybox=true,
                    shadow=true)

    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9, wspace=0.4, hspace=0.4)

    plotshow()
end
