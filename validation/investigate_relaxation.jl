using TrixiParticles
using JSON
using Plots
using GLM
using DataFrames
using Printf

pp_cb = PostprocessCallback(ekin, max_pressure, avg_density; dt=0.025, filename="relaxation")

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "rectangular_tank_2d.jl"),
              extra_callback=pp_cb, tspan=(0.0, 5.0));

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

file_path = joinpath(pwd(), "out", "relaxation.json")

if file_path != ""
    json_string = read(file_path, String)
    json_data = JSON.parse(json_string)

    time = Vector{Float64}(json_data["ekin_fluid_1"]["time"])
    e_kin = Vector{Float64}(json_data["ekin_fluid_1"]["values"])
    tl_ekin, grad_ekin = calculate_regression(e_kin, time)

    x_position_for_annotation = minimum(time) + (maximum(time) - minimum(time)) * 0.5

    p_max = Vector{Float64}(json_data["max_pressure_fluid_1"]["values"])
    tl_p_max, grad_p_max = calculate_regression(p_max, time)

    avg_rho = Vector{Float64}(json_data["avg_density_fluid_1"]["values"])
    tl_avg_rho, grad_avg_rho = calculate_regression(avg_rho, time)

    plot1 = plot(time, [e_kin, tl_ekin], label=["sim" "trend"], color=[:blue :red],
                 linewidth=[2 2],
                 title="Kinetic Energy of the Fluid", xlabel="Time [s]",
                 ylabel="kinetic energy [J]")
    annotate!(x_position_for_annotation, ylims(plot1)[2] - 0.0001,
              @sprintf("gradient=%.5f", grad_ekin))

    plot2 = plot(time, [p_max, tl_p_max], label=["sim" "trend"], color=[:blue :red],
                 linewidth=[2 2],
                 title="Maximum Pressure of the Fluid", xlabel="Time [s]",
                 ylabel="Max. Pressure [Pa]")
    annotate!(x_position_for_annotation, ylims(plot2)[2] - 5.5,
              @sprintf("gradient=%.5f", grad_p_max))

    plot3 = plot(time, [avg_rho, tl_avg_rho], label=["sim" "trend"], color=[:blue :red],
                 linewidth=[2 2],
                 title="Avg. Density of the Fluid", xlabel="Time [s]",
                 ylabel="Avg. Density [kg/m^3]")
    annotate!(x_position_for_annotation, ylims(plot3)[2] - 0.003,
              @sprintf("gradient=%.5f", grad_avg_rho))

    plot(plot1, plot2, plot3, layout=(1, 3), legend=:bottom, size=(1800, 500))
end
