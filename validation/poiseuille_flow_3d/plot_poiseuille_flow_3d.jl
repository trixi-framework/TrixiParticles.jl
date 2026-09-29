using TrixiParticles
using TrixiParticles.JSON
using Plots
using Bessels

# Reference files are available for 10, 20, 30, 40 and 50.
# Use `particle_spacing_factor = 50` to match the resolution used by Zhang et al. (2025).
particle_spacing_factor = 50

# Set to true to plot the results of a validation run in the `out` folder
# instead of the reference file.
use_sim_results = false

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_3d.jl"),
              particle_spacing_factor=particle_spacing_factor, sol=nothing)

# First five roots of the Bessel function of the first kind, J₀(x),
# required for the analytical solution of the transient velocity profile in axisymmetric pipe flow.
roots_J_0 = [2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391, 14.9309177086]

# Analytical velocity evolution given in eq. 18 (Zhang et al., 2025)
function hagen_poiseuille_velocity(r, t)
    # Base profile (stationary part)
    base_profile = (imposed_pressure_drop / (4 * dynamic_viscosity * channel_length)) *
                   (channel_radius^2 - r^2)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in eachindex(roots_J_0)
        alpha = roots_J_0[n]
        J_1 = besselj(1, alpha)
        J_2 = besselj(2, alpha)

        coefficient = (imposed_pressure_drop * channel_radius^2 * J_2) /
                      (dynamic_viscosity * channel_length * alpha^2 * J_1^2)

        exp_term = exp(-t * (dynamic_viscosity * alpha^2) /
                       (fluid_density * channel_radius^2))

        transient_sum += coefficient * besselj0(r * alpha / channel_radius) * exp_term
    end

    # Total velocity
    v_x = base_profile - transient_sum

    return v_x
end

input_file = use_sim_results ?
             joinpath("out",
                      "validation_run_poiseuille_flow_3d_$particle_spacing_factor.json") :
             joinpath(validation_dir(), "poiseuille_flow_3d",
                      "validation_reference_$particle_spacing_factor.json")
data = JSON.parsefile(input_file, allownan=true)["v_x_fluid_1"]

times = data["time"]
times_ref = [0.03, 0.05, 0.07, 0.14, 0.3, 1.0]
positions = range(-channel_radius, channel_radius, length=100)
data_range = 10:90
data_indices = [findfirst(t -> isapprox(t, t_ref), times) for t_ref in times_ref]
v_x_vector = [Float64.(data["values"][i]) for i in data_indices]

# Calculate RMSEP error (eq. 17, Zhang et al., 2025)
rmsep_run = Float64[]
for (i, t) in enumerate(times_ref)
    N = length(data_range)
    res = sum(data_range, init=0) do j
        v_x = v_x_vector[i][j]

        v_analytical = hagen_poiseuille_velocity(positions[j], t)

        # Avoid dividing by zero
        v_analytical < sqrt(eps()) && return 0.0

        rel_err = (v_analytical - v_x) / v_analytical

        return rel_err^2 / N
    end

    push!(rmsep_run, sqrt(res) * 100)
end

# RMSEP error (%) received by Zhang et al. (2025)
rmsep_reference = [2.97, 1.88, 1.61, 1.5, 0.74, 0.89]

p_rmsep = scatter(collect(times_ref), rmsep_run, markersize=5, label="TrixiP",
                  title="Poiseuille Flow 3D")
scatter!(p_rmsep, collect(times_ref), rmsep_reference, marker=:x, markersize=5,
         markerstrokewidth=3, label="Zhang et al. (2025)", dpi=200)

yaxis!(p_rmsep, ylabel="RMSEP error (%)", ylims=(0, 4))
xaxis!(p_rmsep, xlabel="t", xlims=(0, 1.1))
plot!(left_margin=5Plots.mm)
plot!(right_margin=5Plots.mm)
plot!(bottom_margin=5Plots.mm)

display(p_rmsep)

plot_range = range(-channel_radius, channel_radius, length=50)
v_x_plot = view(stack(v_x_vector), 1:2:100, :)
label_ = "TrixiP (" .* ["0.03" "0.05" "0.07" "0.14" "0.3" "∞"] .* " s)"
line_colors = cgrad(:coolwarm, length(times_ref), categorical=true)

p = scatter(plot_range, v_x_plot, label=label_, linewidth=3, markersize=5, opacity=0.6,
            palette=line_colors.colors, legend_position=:outerright, size=(750, 400),
            title="Poiseuille Flow 3D")
for t in times_ref
    label__ = t == 1.0 ? "analytical" : nothing
    plot!(p, (y) -> hagen_poiseuille_velocity(y, t),
          xlims=(-channel_radius, channel_radius),
          ylims=(-0.001, 0.005), label=label__, linewidth=3, linestyle=:dash, color=:black)
end

yaxis!(p, ylabel="x velocity (m/s)")
xaxis!(p, xlabel="y position (m)")
plot!(left_margin=5Plots.mm, bottom_margin=5Plots.mm, dpi=200)

display(p)
