# Activate for an interactive plot.
# using GLMakie
using CairoMakie
using Printf
using TrixiParticles
using TrixiParticles.JSON

# Set to true to save the figure instead of returning it for interactive inspection.
save_figure = false
figure_filename = joinpath(validation_dir(), "rigid_body_sliding_2d",
                           "rigid_body_sliding_2d.svg")
plotted_friction_coefficients = (0.2, 0.3, 0.4)
friction_milli(coefficient) = round(Int, 1000 * coefficient)
simulation_files = [joinpath("out",
                             "validation_result_rigid_body_sliding_2d_mu_$(friction_milli(coefficient))_wall_spacing_30.json")
                    for coefficient in plotted_friction_coefficients]
all(isfile, simulation_files) ||
    error("rigid-body sliding results are missing; run the validation script first")

fig = Figure(size=(900, 700))
ax = Axis(fig[1, 1], ylabel="Horizontal displacement [m]",
          title="Rigid-Body Sliding under Coulomb Friction")
error_ax = Axis(fig[2, 1], xlabel="Time [s]", ylabel="Displacement error [m]",
                title="Numerical - Analytical")
linkxaxes!(ax, error_ax)
hidexdecorations!(ax; grid=false)

for (index, simulation_file) in enumerate(simulation_files)
    run_data = JSON.parsefile(simulation_file)
    position_key = only(filter(key -> startswith(key, "frictional_center_of_mass_x_"),
                               keys(run_data)))
    position_data = run_data[position_key]
    displacement = Float64.(position_data["values"]) .- first(position_data["values"])
    analytical_data = run_data["analytical_center_of_mass_x"]
    analytical_displacement = Float64.(analytical_data["values"]) .-
                              first(analytical_data["values"])
    friction_coefficient = plotted_friction_coefficients[index]
    label = "mu_k = $(@sprintf("%.1f", friction_coefficient))"

    lines!(ax, position_data["time"], displacement;
           color=index, colormap=:tab10, colorrange=(1, length(simulation_files)),
           label=label * " numerical")
    lines!(ax, analytical_data["time"], analytical_displacement;
           color=index, colormap=:tab10, colorrange=(1, length(simulation_files)),
           linestyle=:dash, label=label * " analytical")
    lines!(error_ax, position_data["time"], displacement - analytical_displacement;
           color=index, colormap=:tab10, colorrange=(1, length(simulation_files)), label)
end

hlines!(error_ax, 0.0; color=:black, linestyle=:dash)
axislegend(ax; position=:rb, nbanks=2)

if save_figure
    save(figure_filename, fig)
end

fig
