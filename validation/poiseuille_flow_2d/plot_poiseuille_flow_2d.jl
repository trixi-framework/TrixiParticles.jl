using TrixiParticles
using Plots

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              sol=nothing)
output_directory = joinpath(examples_dir(), validation_dir(), "poiseuille_flow_2d")

data = TrixiParticles.CSV.read(joinpath(output_directory, "result_vx.csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]

target_time = 0.1
idx = argmin(abs.(times .- target_time))  # Nearest available time point

v_x_arrays = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]]

p = scatter(range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
            label="TrixiP (0.1s)",
            linewidth=3,
            color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = 0.3
idx = argmin(abs.(times .- target_time))  # Nearest available time point
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.3s)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = 0.6
idx = argmin(abs.(times .- target_time))  # Nearest available time point
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.6s)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = 0.9
idx = argmin(abs.(times .- target_time))  # Nearest available time point
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.9s)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = tspan[2]
idx = argmin(abs.(times .- target_time))  # Nearest available time point
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (∞)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

@show p
