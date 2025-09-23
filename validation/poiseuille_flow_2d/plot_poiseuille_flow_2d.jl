using TrixiParticles
using Plots

# Import variables into scope
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              sol=nothing)

# Analytical velocity evolution given in eq. 16
function poiseuille_velocity(y, t)

    # Base profile (stationary part)
    base_profile = (pressure_drop / (2 * dynamic_viscosity * flow_length)) * y *
                   (y - wall_distance)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 0:10  # Limit to 10 terms for convergence
        coefficient = (4 * pressure_drop * wall_distance^2) /
                      (dynamic_viscosity * flow_length * pi^3 * (2 * n + 1)^3)

        sine_term = sin(pi * y * (2 * n + 1) / wall_distance)

        exp_term = exp(-((2 * n + 1)^2 * pi^2 * dynamic_viscosity * t) /
                       (fluid_density * wall_distance^2))

        transient_sum += coefficient * sine_term * exp_term
    end

    # Total velocity
    v_x = base_profile + transient_sum

    return v_x
end

# Load results
output_directory = joinpath(validation_dir(), "poiseuille_flow_2d")
data = TrixiParticles.CSV.read(joinpath(output_directory, "result_vx.csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]
times_ref = [0.1, 0.3, 0.6, 0.9, 2.0]
positions = range(0, wall_distance, length=100)
data_range = 1:100
data_indices = findall(t -> t in times_ref, times)
v_x_vector = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]][data_indices]

# Calculate RMSEP error (eq. 17, Zhang et al.)
rmsep_run = Float64[]
for (i, t) in enumerate(times_ref)
    N = length(data_range)
    res = sum(data_range, init=0) do j
        v_x = v_x_vector[i][j]

        v_analytical = -poiseuille_velocity(positions[j], t)

        # Avoid dividing by zero
        v_analytical < sqrt(eps()) && return 0.0

        rel_err = (v_analytical - v_x) / v_analytical

        return rel_err^2 / N
    end

    push!(rmsep_run, sqrt(res) * 100)
end

# RMSEP error (%) from Zhang et al. (2025)
rmsep_reference = [1.81, 0.95, 0.67, 0.86, 1.22]

p_rmsep = scatter(times_ref, rmsep_run, markersize=5, label="TrixiP")
scatter!(p_rmsep, times_ref, rmsep_reference, marker=:x, markersize=5,
         markerstrokewidth=3, label="Zhang et al. (2025)")

yaxis!(p_rmsep, ylabel="RMSEP error (%)", ylims=(0, 2))
xaxis!(p_rmsep, xlabel="t", xlims=(0, 2))
plot!(left_margin=5Plots.mm)
plot!(right_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

@show p_rmsep

plot_range = range(0, wall_distance, length=50)
v_x_plot = view(stack(v_x_vector), 1:2:100, :)
label_ = "TrixiP (" .* ["0.1" "0.3" "0.6" "0.9" "∞"] .* " s)"
line_colors = cgrad(:coolwarm, length(times_ref), categorical=true)

p = scatter(plot_range, v_x_plot, label=label_, linewidth=3, markersize=5, opacity=0.6,
            palette=line_colors.colors, legend_position=:outerright)
for t in times_ref
    label_ = t == 2.0 ? "analytical" : nothing
    plot!(p, (y) -> -poiseuille_velocity(y, t), xlims=(0, wall_distance),
          ylims=(-0.002, 0.014), label=label_, linewidth=3, linestyle=:dash, color=:black)
end

yaxis!(p, ylabel="x velocity (m/s)")
xaxis!(p, xlabel="y position (m)")
plot!(left_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

@show p
