using CSV: CSV
using DataFrames: DataFrame, groupby
using Plots
using Statistics: median

output_directory = joinpath("out", "corrections_periodic", "tgv_2d")
data = CSV.read(joinpath(output_directory, "tgv_decay.csv"), DataFrame)
styles = Dict("none" => ("Uncorrected", "#333333", :solid),
              "shepard" => ("Shepard", "#BC4570", :dot),
              "gradient" => ("Gradient", "#009E73", :dash),
              "shepard_gradient" => ("Shepard + gradient", "#0072B2", :dashdot),
              "mixed" => ("Mixed", "#804998", :solid),
              "shepard_mixed" => ("Shepard + mixed", "#D55E00", :dash))
panels = Plots.Plot[]
for runs in groupby(data, :resolution; sort=true)
    n = first(runs.resolution)
    epsilon = only(unique(runs.epsilon))
    energy = plot(; title="$(n)×$n · Energy decay (ε=$epsilon)", ylabel="K(t) / K(0)",
                  legend=:topright)
    residual = plot(; title="Energy deviation", ylabel="(K / Kexact − 1) [%]", legend=false)
    velocity = plot(; title="Velocity field error", ylabel="Relative L₂ error [%]",
                    legend=:outerright)
    for series in groupby(runs, :method)
        label, color, linestyle = styles[first(series.method)]
        plot!(energy, series.time, series.energy_ratio; color, linestyle, label=false)
        plot!(residual, series.time,
              100 .* (series.energy_ratio ./ series.energy_ratio_exact .- 1);
              color, linestyle)
        plot!(velocity, series.time, 100 .* series.velocity_l2_error; color, linestyle,
              label)
    end
    ref = first(groupby(runs, :method))
    plot!(energy, ref.time, ref.energy_ratio_exact; color=:black, linestyle=:dash,
          label="Exact (2D)")
    hline!(residual, [0.0]; color=:gray, linewidth=0.8)
    append!(panels, (energy, residual, velocity))
end
rows = length(unique(data.resolution))
figure = plot(panels...; layout=(rows, 3), size=(1550, 360 * rows),
              xlabel="Time",
              linewidth=2, legendfontsize=10, tickfontsize=10, guidefontsize=12,
              foreground_color_legend=nothing, left_margin=9Plots.mm,
              top_margin=4Plots.mm, bottom_margin=6Plots.mm)
plot_file = joinpath(output_directory, "tgv_decay.png")
savefig(figure, plot_file)
println("Wrote $plot_file")

# Compare the same final time and the same saved times across all grids. Logarithmic
# axes make a slope of -1 in particles per axis correspond to first order in Δx.
levels = sort(unique(data.resolution))
markers = Dict("none" => :circle, "shepard" => :star5, "gradient" => :diamond,
               "shepard_gradient" => :utriangle, "mixed" => :rect, "shepard_mixed" => :cross
               )
function scaling_panel(title, ylabel; kwargs...)
    plot(; title, ylabel, xlabel="Particles per axis (N)",
         xscale=:log2, xticks=(levels, string.(levels)), legend=:outerright, kwargs...)
end
comparison_layout = (; layout=(2, 1), size=(850, 780), linewidth=2, guidefontsize=12,
                     tickfontsize=10, legendfontsize=10, foreground_color_legend=nothing,
                      left_margin=7Plots.mm, top_margin=3Plots.mm, bottom_margin=5Plots.mm)

times = sort(unique(data.time))
endpoint = scaling_panel("Velocity error at t=$(last(times))", "Relative L₂ error [%]";
                         yscale=:log10)
peak = scaling_panel("Largest velocity error over $(length(times)) saved times",
                     "Relative L₂ error [%]"; yscale=:log10)
for runs in groupby(data, :method)
    method = first(runs.method)
    label, color, linestyle = styles[method]
    groups = groupby(runs, :resolution; sort=true)
    ns = [first(s.resolution) for s in groups]
    all(s -> sort(s.time) == times, groups) || error("Mismatched output times for $method")
    final_errors = [100 * s.velocity_l2_error[argmax(s.time)] for s in groups]
    peak_errors = [100 * maximum(s.velocity_l2_error) for s in groups]
    for (panel, errs) in ((endpoint, final_errors), (peak, peak_errors))
        plot!(panel, ns, errs; label, color, linestyle, marker=markers[method])
        if method == "gradient" && length(ns) > 1
            plot!(panel, ns, (first(errs)/2) .* first(ns) ./ ns; color=:gray,
                  linestyle=:dot,
                  label="First order (1/N)")
        end
    end
end
for panel in (endpoint, peak)
    lo, hi = ylims(panel)
    ticks = exp2.(ceil(log2(lo)):floor(log2(hi)))
    yticks!(panel, ticks, string.(ticks))
end
resolution_plot = joinpath(output_directory, "tgv_error_vs_resolution.png")
savefig(plot(endpoint, peak; comparison_layout...), resolution_plot)
println("Wrote $resolution_plot")

timings = CSV.read(joinpath(output_directory, "tgv_timings.csv"), DataFrame)
threads = only(unique(timings.threads))
only(unique(timings.t_end)) == last(times) || error("Mismatched benchmark end times")
Set(zip(data.method, data.resolution, data.epsilon)) ==
Set(zip(timings.method, timings.resolution, timings.epsilon)) ||
    error("Timing and error results must use the same cases")
baseline = Dict(first(s.resolution) => median(s.seconds)
                for s in groupby(timings[timings.method .== "none", :], :resolution))
runtime = scaling_panel("Full solve to t=$(last(times)) · $threads threads",
                        "Median wall time [s]"; yscale=:log10)
relative = scaling_panel("Cost relative to uncorrected at the same N",
                         "Time / uncorrected time")
for runs in groupby(timings, :method)
    method = first(runs.method)
    label, color, linestyle = styles[method]
    groups = groupby(runs, :resolution; sort=true)
    ns = [first(s.resolution) for s in groups]
    med = [median(s.seconds) for s in groups]
    span = (med - [minimum(s.seconds) for s in groups],
            [maximum(s.seconds) for s in groups] - med)
    plot!(runtime, ns, med; yerror=span, label, color, linestyle, marker=markers[method])
    plot!(relative, ns, med ./ [baseline[n] for n in ns]; label, color, linestyle,
          marker=markers[method])
end
performance_plot = joinpath(output_directory, "tgv_performance.png")
savefig(plot(runtime, relative; comparison_layout...), performance_plot)
println("Wrote $performance_plot")
