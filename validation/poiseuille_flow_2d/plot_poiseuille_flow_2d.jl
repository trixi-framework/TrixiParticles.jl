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

rmsep = Float64[]
for t_target in (0.1, 0.3, 0.6, 0.9, 2.0)
    idx = argmin(abs.(times .- t_target))  # Nearest available time point

    range_ = 10:90
    N = length(range_)
    positions = range(0, wall_distance, length=100)
    res = sum(range_, init=0) do i
        v_x = v_x_arrays[idx][i]

        v_analytical = -poiseuille_velocity(positions[i], t_target)

        v_analytical < sqrt(eps()) && return 0.0

        rel_err = (v_analytical - v_x) / v_analytical

        return rel_err^2 / N
    end

    push!(rmsep, sqrt(res) * 100)
end

p
