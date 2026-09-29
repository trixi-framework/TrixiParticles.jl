using TrixiParticles
using TrixiParticles.JSON
using Plots

# Reference files are available for 30 and 50.
# Use `particle_spacing_factor = 50` to match the resolution used by Zhang et al. (2025).
particle_spacing_factor = 50

# Set to true to plot the results of a validation run in the `out` folder
# instead of the reference file.
use_sim_results = false

# Import variables into scope
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              particle_spacing_factor=particle_spacing_factor, sol=nothing)

# Analytical velocity evolution given in eq. 16 (Zhang et al., 2025)
function poiseuille_velocity(y, t)

    # Base profile (stationary part)
    base_profile = (imposed_pressure_drop / (2 * dynamic_viscosity * channel_length)) * y *
                   (y - channel_height)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 0:10  # Limit to 10 terms for convergence
        coefficient = (4 * imposed_pressure_drop * channel_height^2) /
                      (dynamic_viscosity * channel_length * pi^3 * (2 * n + 1)^3)

        sine_term = sin(pi * y * (2 * n + 1) / channel_height)

        exp_term = exp(-((2 * n + 1)^2 * pi^2 * dynamic_viscosity * t) /
                       (fluid_density * channel_height^2))

        transient_sum += coefficient * sine_term * exp_term
    end

    # Total velocity
    v_x = base_profile + transient_sum

    return v_x
end

# Load results
input_file = use_sim_results ?
             joinpath("out",
                      "validation_run_poiseuille_flow_2d_$particle_spacing_factor.json") :
             joinpath(validation_dir(), "poiseuille_flow_2d",
                      "validation_reference_$particle_spacing_factor.json")
data = JSON.parsefile(input_file, allownan=true)["v_x_fluid_1"]

times = data["time"]
times_ref = [0.1, 0.3, 0.6, 0.9, 2.0]
positions = range(0, channel_height, length=100)
data_range = 2:98
data_indices = [findfirst(t -> isapprox(t, t_ref), times) for t_ref in times_ref]
v_x_vector = [Float64.(data["values"][i]) for i in data_indices]

# Calculate RMSEP error (eq. 17, Zhang et al., 2025)
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

# RMSEP error (%) received by Zhang et al. (2025)
rmsep_reference = [1.81, 0.95, 0.67, 0.86, 1.22]

p_rmsep = scatter(times_ref, rmsep_run, markersize=5, label="TrixiP",
                  title="Poiseuille Flow 2D")
scatter!(p_rmsep, times_ref, rmsep_reference, marker=:x, markersize=5,
         markerstrokewidth=3, label="Zhang et al. (2025)", dpi=200)

yaxis!(p_rmsep, ylabel="RMSEP error (%)", ylims=(0, 4))
xaxis!(p_rmsep, xlabel="t", xlims=(0, 2.05))
plot!(left_margin=5Plots.mm)
plot!(right_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

display(p_rmsep)

plot_range = range(0, channel_height, length=50)
v_x_plot = view(stack(v_x_vector), 1:2:100, :)
label_ = "TrixiP (" .* ["0.1" "0.3" "0.6" "0.9" "∞"] .* " s)"
line_colors = cgrad(:coolwarm, length(times_ref), categorical=true)

p = scatter(plot_range, v_x_plot, label=label_, linewidth=3, markersize=5, opacity=0.6,
            palette=line_colors.colors, legend_position=:outerright, size=(750, 400),
            title="Poiseuille Flow 2D")
for t in times_ref
    label__ = t == 2.0 ? "analytical" : nothing
    plot!(p, (y) -> -poiseuille_velocity(y, t), xlims=(0, channel_height),
          ylims=(-0.002, 0.014), label=label__, linewidth=3, linestyle=:dash, color=:black)
end

yaxis!(p, ylabel="x velocity (m/s)")
xaxis!(p, xlabel="y position (m)")
plot!(left_margin=5Plots.mm, bottom_margin=5Plots.mm, dpi=200)

display(p)
