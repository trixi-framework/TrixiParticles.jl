using Base64: base64encode
using CSV
using DataFrames
using Plots

dimension = !isempty(ARGS) && (@__FILE__) == abspath(PROGRAM_FILE) ?
            parse(Int, only(ARGS)) : 2
output_directory = joinpath("out", "corrections_periodic",
                            dimension == 2 ? "" : "$(dimension)d")
csv_file = joinpath(output_directory, "correction_convergence.csv")
report_file = joinpath(output_directory, "correction_validation.html")
plot_file = joinpath(output_directory, "correction_validation.png")

data = CSV.read(csv_file, DataFrame)
all(==("periodic"), data.region) || throw(ArgumentError("expected periodic study results"))
unique(data.dimension) == [dimension] ||
    throw(ArgumentError("expected only $(dimension)D results"))

categories = (("Interpolation and density",
               op -> op in ("interpolation", "summation_density",
                            "density_reinitialization")),
              ("Direct and difference gradients", op -> endswith(op, "_gradient")),
              ("Pressure acceleration", op -> startswith(op, "pressure_")),
              ("Constant-pressure response", op -> startswith(op, "constant_pressure_")))
levels = sort(unique(data.resolution))
# Use the same key for corrections in all three gradient/pressure panels.
# Different marker shapes and sizes help distinguish coincident curves.
styles = Dict("none" => ("Uncorrected", "#333333", :circle, :solid, 7),
              "kernel" => ("Kernel", "#0072B2", :rect, :dash, 6),
              "gradient" => ("Gradient", "#009E73", :diamond, :solid, 8),
              "blended" => ("Blended", "#D55E00", :utriangle, :dashdot, 5),
              "mixed" => ("Mixed", "#804998", :circle, :dash, 4),
              "shepard" => ("Shepard", "#BC4570", :cross, :dot, 5),
              "shepard_mixed" => ("Shepard + mixed", "#76521A", :xcross, :dot, 3))
density_styles = Dict(
    ("none",
     "interpolation") => ("Raw interpolation", "#333333", :circle, :solid, 7),
    ("shepard",
     "interpolation") => ("Shepard interpolation", "#0072B2", :diamond, :dash,
                          4),
    ("none",
     "summation_density") => ("Raw density sum", "#D55E00", :rect, :solid, 7),
    ("shepard",
     "summation_density") => ("Shepard density sum", "#009E73", :utriangle,
                              :dashdot, 5),
    ("shepard",
     "density_reinitialization") => ("Density reset", "#804998", :circle,
                                     :dot, 4))
panels = map(enumerate(categories)) do (index, (title, matches))
    subset = filter(row -> matches(row.operator), data)
    panel = plot(; title="$(dimension)D · $title", titlelocation=:left, titlefontsize=16,
                 size=(750, 400),
                 xscale=:log2, xticks=(levels, string.(levels)),
                 xlabel="Particles per axis",
                 ylabel=index == 4 ? "RMS acceleration (×10⁻¹⁵)" : "Relative L₂ error",
                 yscale=index == 4 ? :identity : :log10, legend=:outerright,
                 foreground_color_legend=nothing, legendfontsize=10, tickfontsize=10,
                 guidefontsize=12, gridalpha=0.12, margin=5Plots.mm, widen=true)
    # For the gradient and pressure variants show the full range for each correction.
    for group in groupby(subset, index == 1 ? [:method, :operator] : [:method])
        method = first(group.method)
        bounds = combine(groupby(group, :resolution; sort=true),
                         :l2_error => minimum => :low,
                         :l2_error => maximum => :high)
        label, color, marker, linestyle,
        markersize = index == 1 ?
                     density_styles[(method, first(group.operator))] : styles[method]
        scale = index == 4 ? 1.0e15 : 1.0
        plot!(panel, bounds.resolution, scale * bounds.high;
              ribbon=(scale * (bounds.high - bounds.low), zero(bounds.high)),
              fillalpha=0.12,
              linewidth=2, markerstrokewidth=0, color, marker, markersize, linestyle, label)
    end
    if index != 4
        # Anchor the guide to the coarse corrected measurement; only the slope matters.
        method = index == 1 ? "shepard" : "mixed"
        reference = filter(row -> row.method == method, subset)
        coarse = maximum(reference.l2_error[reference.resolution .== first(levels)])
        plot!(panel, levels, 0.5 * coarse .* (first(levels) ./ levels) .^ 2;
              linestyle=:dash, color=:gray, label="O(h²) guide")
    else
        ylims!(panel, 0, 1.08 * max(1e-4, 1e15 * maximum(subset.l2_error)))
    end
    panel
end
figure = plot(panels...; layout=(2, 2), size=(1560, 900), left_margin=8Plots.mm,
              right_margin=5Plots.mm, top_margin=5Plots.mm, bottom_margin=6Plots.mm)
savefig(figure, plot_file)
open(report_file, "w") do io
    csv = base64encode(read(csv_file))
    println(io,
            """<!doctype html><meta charset='utf-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1'>
        <title>$(dimension)D periodic correction study</title>
        <style>body{font:1rem system-ui;margin:1rem auto;padding:0 1rem;max-width:96rem}
        .plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,40rem),1fr))}
        img{width:100%;height:auto}</style>
        <h1>$(dimension)D periodic correction consistency</h1><div class='plots'>""")
    for ((title, _), panel) in zip(categories, panels)
        svg = base64encode(repr(MIME"image/svg+xml"(), panel))
        println(io, "<img alt='$title' src='data:image/svg+xml;base64,$svg'>")
    end
    println(io,
            "</div><p>For gradients and pressure, lines show the largest error per correction; shading spans the measured variants. The constant-pressure target is zero.</p>")
    println(io,
            "<a download='correction_convergence.csv' href='data:text/csv;base64,$csv'>Download all measurements (CSV)</a>")
end
println("Wrote $plot_file and $report_file")
