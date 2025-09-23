using TrixiParticles
using Plots
using Bessels

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
              sol=nothing)

# output_directory = joinpath(examples_dir(), validation_dir(), "hagen_poiseuille_flow_3d")
output_directory = "out"

data = TrixiParticles.CSV.read(joinpath(output_directory, "result_vx.csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]
v_x_arrays = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]]

# Roots of J_0
const roots_J_0 = [2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391, 14.9309177086]

function hagen_poiseuille_velocity(r, t)
    # Base profile (stationary part)
    base_profile = (pressure_drop / (4 * dynamic_viscosity * flow_length)) *
                   (pipe_radius^2 - r^2)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 1:5
        alpha = roots_J_0[n]
        J_1 = besselj(1, alpha)
        J_2 = besselj(2, alpha)

        coefficient = (pressure_drop * pipe_radius^2 * J_2) /
                      (dynamic_viscosity * flow_length * alpha^2 * J_1^2)

        exp_term = exp(-t * (dynamic_viscosity * alpha^2) /
                       (fluid_density * pipe_radius^2))

        transient_sum += coefficient * besselj0(r * alpha / pipe_radius) * exp_term
    end

    # Total velocity
    v_x = base_profile - transient_sum

    return v_x
end

t_targets = [0.03, 0.05, 0.07, 0.14, 0.3, 1.0]

idx = argmin(abs.(times .- t_targets[1]))  # Nearest available time point
p = scatter(range(-pipe_radius, pipe_radius, length=50), v_x_arrays[idx][1:2:100],
            label="TrixiP (0.03s)", linewidth=3, markersize=5, opacity=0.4)
plot!(p, (y) -> hagen_poiseuille_velocity(y, t_targets[1]),
      xlims=(-pipe_radius, pipe_radius), legendposition=:outerright, size=(750, 400),
      ylims=(-0.001, 0.005), label=nothing, linewidth=3, linestyle=:dash, color=:black)

idx = argmin(abs.(times .- t_targets[2]))  # Nearest available time point
scatter!(p, range(-pipe_radius, pipe_radius, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.05s)", linewidth=3, markersize=5, opacity=0.4)
plot!((y) -> hagen_poiseuille_velocity(y, t_targets[2]), xlims=(-pipe_radius, pipe_radius),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.001, 0.005), label=nothing, linewidth=3, linestyle=:dash, color=:black)

idx = argmin(abs.(times .- t_targets[3]))  # Nearest available time point
scatter!(p, range(-pipe_radius, pipe_radius, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.07s)", linewidth=3, markersize=5, opacity=0.4)
plot!((y) -> hagen_poiseuille_velocity(y, t_targets[3]), xlims=(-pipe_radius, pipe_radius),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.001, 0.005),
      label=nothing, linewidth=3, linestyle=:dash, color=:black)

idx = argmin(abs.(times .- t_targets[4]))  # Nearest available time point
scatter!(p, range(-pipe_radius, pipe_radius, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.14s)", linewidth=3, markersize=5, opacity=0.4)
plot!((y) -> hagen_poiseuille_velocity(y, t_targets[4]), xlims=(-pipe_radius, pipe_radius),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.001, 0.005),
      label=nothing, linewidth=3, linestyle=:dash, color=:black)

idx = argmin(abs.(times .- t_targets[5]))  # Nearest available time point
scatter!(p, range(-pipe_radius, pipe_radius, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.3s)", linewidth=3, markersize=5, opacity=0.4)
plot!((y) -> hagen_poiseuille_velocity(y, t_targets[5]), xlims=(-pipe_radius, pipe_radius),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.001, 0.005),
      label=nothing, linewidth=3, linestyle=:dash, color=:black)

idx = argmin(abs.(times .- t_targets[6]))  # Nearest available time point
scatter!(p, range(-pipe_radius, pipe_radius, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (∞)", linewidth=3, markersize=5, opacity=0.4)
plot!((y) -> hagen_poiseuille_velocity(y, t_targets[6]), xlims=(-pipe_radius, pipe_radius),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.001, 0.005),
      label="analytical", linewidth=3, linestyle=:dash, color=:black)

yaxis!(p, ylabel="x velocity (m/s)")
xaxis!(p, xlabel="y position (m)")
plot!(left_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)



rmsep_run = Float64[]
for t_target in [0.03, 0.05, 0.07, 0.14, 0.3, 1.0]
    idx = argmin(abs.(times .- t_target))  # Nearest available time point

    range_ = 10:90
    N = length(range_)
    positions = range(-pipe_radius, pipe_radius, length=100)
    res = sum(range_, init=0) do i
        v_x = v_x_arrays[idx][i]

        v_analytical = hagen_poiseuille_velocity(positions[i], t_target)

        v_analytical < sqrt(eps()) && return 0.0

        rel_err = (v_analytical - v_x) / v_analytical

        return rel_err^2 / N
    end

    push!(rmsep_run, sqrt(res) * 100)
end
