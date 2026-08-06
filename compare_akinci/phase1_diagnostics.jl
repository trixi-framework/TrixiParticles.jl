using CairoMakie

function phase1_diagnostics(output_path)
    particle_counts = [389, 739, 1503, 2969, 6031]
    css_phase0 = [1.0333, 1.0026, 0.9970, 0.9557, 1.0079]
    css_phase1 = [1.03091, 0.99925, 0.99265, 0.95259, 1.00437]

    radii = [0.00498, 0.00617, 0.00782]
    inverse_radius = 2 ./ radii
    morris_pressure = [460.464, 363.865, 275.497]
    css_pressure = [468.614, 352.515, 250.727]
    fit_x = range(minimum(inverse_radius), maximum(inverse_radius); length=100)
    morris_fit = -48.792 .+ 1.270360 .* fit_x
    css_fit = -132.249 .+ 1.497079 .* fit_x

    figure = Figure(; size=(1200, 500), fontsize=16)
    static_axis = Axis(figure[1, 1];
                       title="Static CSS balance",
                       xlabel="particle count",
                       ylabel="inferred sigma / input sigma",
                       xscale=log10,
                       xticks=(particle_counts, string.(particle_counts)))
    band!(static_axis, particle_counts, fill(0.95, length(particle_counts)),
          fill(1.05, length(particle_counts)); color=(:seagreen, 0.13),
          label="+/-5% acceptance")
    hlines!(static_axis, [1.0]; color=:gray35, linestyle=:dash)
    lines!(static_axis, particle_counts, css_phase0; color=:gray45, linewidth=2,
           label="Phase 0")
    scatter!(static_axis, particle_counts, css_phase0; color=:gray45, markersize=11)
    lines!(static_axis, particle_counts, css_phase1; color=:dodgerblue3, linewidth=3,
           label="Phase 1")
    scatter!(static_axis, particle_counts, css_phase1; color=:dodgerblue3,
             markersize=12)
    ylims!(static_axis, 0.93, 1.06)
    axislegend(static_axis; position=:rb)

    laplace_axis = Axis(figure[1, 2];
                        title="Dynamic three-radius Laplace fit",
                        xlabel="2 / R [1/m]",
                        ylabel="median interior pressure [Pa]")
    lines!(laplace_axis, fit_x, morris_fit; color=:darkorange2, linewidth=2,
           label="Morris fit: sigma=1.270")
    scatter!(laplace_axis, inverse_radius, morris_pressure; color=:darkorange2,
             markersize=13)
    lines!(laplace_axis, fit_x, css_fit; color=:purple3, linewidth=2,
           label="CSS fit: sigma=1.497")
    scatter!(laplace_axis, inverse_radius, css_pressure; color=:purple3,
             marker=:diamond, markersize=14)
    axislegend(laplace_axis; position=:lt)
    text!(laplace_axis, maximum(inverse_radius), minimum(css_pressure) + 12;
          text="Both series finish below 5 min\nMorris: 137 s | CSS: 119 s",
          align=(:right, :bottom), color=:gray25)

    Label(figure[0, :],
          "Phase 1 surface-tension diagnostics: static preservation and dynamic bias",
          fontsize=21, font=:bold)
    Label(figure[2, :],
          "Static CSS remains within 5%. Dynamic slope overprediction is retained as a Phase 2 convergence target.",
          fontsize=14, color=:gray30)

    save(output_path, figure; px_per_unit=1.5)
    println("Wrote Phase 1 diagnostics to $output_path")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    output_path = isempty(ARGS) ?
                  joinpath(@__DIR__, "phase1_surface_tension_diagnostic.png") : ARGS[1]
    phase1_diagnostics(output_path)
end
