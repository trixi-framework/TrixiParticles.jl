# Activate for an interactive plot.
# using GLMakie
using CairoMakie
using Glob
using Printf
using TrixiParticles
using TrixiParticles.JSON

# Set to true to save the figure instead of returning it for interactive inspection.
save_figure = false
figure_filename = joinpath(validation_dir(), "rigid_body_sliding_2d",
                           "rigid_body_sliding_2d.svg")
simulation_files = sort(glob("validation_result_rigid_body_sliding_2d_wall_spacing_*.json",
                             "out"))
isempty(simulation_files) &&
    error("no rigid-body sliding results found; run the validation script first")

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
    spacing_match = match(r"wall_spacing_(\d+)\.json$", simulation_file)
    wall_spacing = parse(Int, only(spacing_match.captures)) / 1000
    label = "wall spacing = $(@sprintf("%.3f", wall_spacing)) m"

    lines!(ax, position_data["time"], displacement;
           color=index, colormap=:tab10, colorrange=(1, length(simulation_files)), label)
    lines!(error_ax, position_data["time"], displacement - analytical_displacement;
           color=index, colormap=:tab10, colorrange=(1, length(simulation_files)), label)

    if index == 1
        lines!(ax, analytical_data["time"], analytical_displacement;
               color=:black, linestyle=:dash, label="analytical")
    end
end

hlines!(error_ax, 0.0; color=:black, linestyle=:dash)
axislegend(ax; position=:rb)

if save_figure
    save(figure_filename, fig)
end

fig
