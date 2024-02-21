using TrixiParticles
using JSON
using Plots
using GLM
using DataFrames
using Printf

pp_cb = PostprocessCallback(; ekin=kinetic_energy, max_pressure, avg_density, dt=0.025,
                            filename="relaxation", write_csv=false)

pp_damped_cb = PostprocessCallback(; ekin=kinetic_energy, max_pressure, avg_density,
                                   dt=0.025,
                                   filename="relaxation_damped", write_csv=false)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=pp_cb, tspan=(0.0, 5.0), saving_callback=nothing,
              fluid_particle_spacing=0.02,
              viscosity_wall=ViscosityAdami(nu=0.5));

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=pp_damped_cb, tspan=(0.0, 5.0),
              source_terms=SourceTermDamping(;
                                             damping_coefficient=2.0),
              saving_callback=nothing, fluid_particle_spacing=0.02,
              viscosity_wall=ViscosityAdami(nu=0.5));

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
file_path_damped = joinpath(pwd(), "out", "relaxation_damped.json")

function read_and_parse(file_path)
    json_string = read(file_path, String)
    json_data = JSON.parse(json_string)

    time = Vector{Float64}(json_data["ekin_fluid_1"]["time"])
    e_kin = Vector{Float64}(json_data["ekin_fluid_1"]["values"])
    p_max = Vector{Float64}(json_data["max_pressure_fluid_1"]["values"])
    avg_rho = Vector{Float64}(json_data["avg_density_fluid_1"]["values"])

    return time, e_kin, p_max, avg_rho
end

if file_path != ""
    time, e_kin, p_max, avg_rho = read_and_parse(file_path)
    time_damped, e_kin_damped, p_max_damped, avg_rho_damped = read_and_parse(file_path_damped)

    tl_ekin, grad_ekin = calculate_regression(e_kin, time)
    tl_ekin_damped, grad_ekin_damped = calculate_regression(e_kin_damped, time_damped)

    tl_p_max, grad_p_max = calculate_regression(p_max, time)
    tl_p_max_damped, grad_p_max_damped = calculate_regression(p_max_damped, time_damped)

    tl_avg_rho, grad_avg_rho = calculate_regression(avg_rho, time)
    tl_avg_rho_damped, grad_avg_rho_damped = calculate_regression(avg_rho_damped,
                                                                  time_damped)

    plot1 = plot(time, [e_kin, tl_ekin], label=["undamped" "trend"],
                 color=[:blue :red], linewidth=[2 2])

    plot!(time_damped, [e_kin_damped, tl_ekin_damped],
          label=["damped" "damped trend"], color=[:green :orange], linewidth=[2 2])

    plot!(title="Kinetic Energy of the Fluid", xlabel="Time [s]",
          ylabel="kinetic energy [J]")

    plot2 = plot(time, [p_max, tl_p_max], label=["sim" "trend"], color=[:blue :red],
                 linewidth=[2 2])

    plot!(time_damped, [p_max_damped, tl_p_max_damped], label=["damped" "damped trend"],
          color=[:green :orange], linewidth=[2 2])

    plot!(title="Maximum Pressure of the Fluid", xlabel="Time [s]",
          ylabel="Max. Pressure [Pa]")

    plot3 = plot(time, [avg_rho, tl_avg_rho], label=["sim" "trend"], color=[:blue :red],
                 linewidth=[2 2])

    plot!(time_damped, [avg_rho_damped, tl_avg_rho_damped], label=["damped" "damped trend"],
          color=[:green :orange], linewidth=[2 2])

    plot!(title="Avg. Density of the Fluid", xlabel="Time [s]",
          ylabel="Avg. Density [kg/m^3]")

    plot(plot1, plot2, plot3, layout=(2, 2), size=(1200, 1200))
end
