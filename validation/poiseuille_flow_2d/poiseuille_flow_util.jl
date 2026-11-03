# Shared helpers for plotting the Poiseuille flow validation results.
using TrixiParticles.JSON
using Plots

# Load the x velocity profiles at the times `times_ref`.
function load_velocity_profiles(input_file, times_ref)
    data = JSON.parsefile(input_file, allownan=true)["v_x_fluid_1"]

    times = data["time"]
    data_indices = [findfirst(t -> isapprox(t, t_ref), times) for t_ref in times_ref]

    return [Float64.(data["values"][i]) for i in data_indices]
end

# Calculate the RMSEP in percent (eq. 17, Zhang et al., 2025).
function rmsep(v_x_vector, times_ref, positions, data_range, v_analytical)
    N = length(data_range)

    return map(zip(times_ref, v_x_vector)) do (t, v_x)
        res = sum(data_range, init=0.0) do j
            v_x_analytical = v_analytical(positions[j], t)

            # Avoid dividing by zero
            v_x_analytical < sqrt(eps()) && return 0.0

            rel_err = (v_x_analytical - v_x[j]) / v_x_analytical

            return rel_err^2 / N
        end

        return sqrt(res) * 100
    end
end

function plot_rmsep(times_ref, rmsep_run, rmsep_reference; title, xlims)
    p = scatter(times_ref, rmsep_run, markersize=5, label="TrixiParticles", title=title)
    scatter!(p, times_ref, rmsep_reference, marker=:x, markersize=5,
             markerstrokewidth=3, label="Zhang et al. (2025)", dpi=200)

    yaxis!(p, ylabel="RMSEP (%)", ylims=(0, 4))
    xaxis!(p, xlabel="t", xlims=xlims)
    plot!(p, left_margin=5Plots.mm, right_margin=5Plots.mm, bottom_margin=5Plots.mm)

    return p
end

# Plot every second point of the simulated profiles together with the analytical solution.
function plot_velocity_profiles(positions, v_x_vector, times_ref, v_analytical;
                                title, ylims)
    plot_indices = 1:2:length(positions)
    v_x_plot = view(stack(v_x_vector), plot_indices, :)
    simulation_labels = permutedims(["TrixiParticles ($t s)" for t in times_ref])
    line_colors = cgrad(:coolwarm, length(times_ref), categorical=true)

    p = scatter(positions[plot_indices], v_x_plot, label=simulation_labels, linewidth=3,
                markersize=5, opacity=0.6, palette=line_colors.colors,
                legend_position=:outerright, size=(750, 400), title=title)

    for t in times_ref
        # Only add one legend entry for all analytical profiles.
        analytical_label = t == last(times_ref) ? "analytical" : nothing
        plot!(p, y -> v_analytical(y, t), xlims=extrema(positions), ylims=ylims,
              label=analytical_label, linewidth=3, linestyle=:dash, color=:black)
    end

    yaxis!(p, ylabel="x velocity (m/s)")
    xaxis!(p, xlabel="y position (m)")
    plot!(p, left_margin=5Plots.mm, bottom_margin=5Plots.mm, dpi=200)

    return p
end
