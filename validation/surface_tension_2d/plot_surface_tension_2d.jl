using CairoMakie
using CSV
using DataFrames
using JSON

reference = JSON.parsefile(joinpath(@__DIR__, "validation_reference.json"))
young_laplace = reference["young_laplace"]["results"]
rayleigh = reference["rayleigh_mode_2"]
stability = CSV.read(joinpath(@__DIR__, "rayleigh_tensile_stability.csv"), DataFrame)

figure = Figure(size=(1450, 430))
young_laplace_axis = Axis(figure[1, 1];
                          title="2D Young-Laplace convergence",
                          xlabel="particle count", ylabel="fitted sigma error [%]",
                          xscale=log10, yscale=log10)
rayleigh_axis = Axis(figure[1, 2]; title="Rayleigh mode-2 stiffness",
                     xlabel="particle count", ylabel="frequency error [%]",
                     xscale=log10)
stability_axis = Axis(figure[1, 3]; title="Free Rayleigh tensile stability",
                      xlabel="applicable shipped option", ylabel="periods before collapse",
                      xticks=(1:3, ["baseline", "EOS background", "tangential PST"]),
                      limits=(nothing, nothing, 0, 5.4))

young_laplace_particles = getindex.(young_laplace, "particle_count")
young_laplace_errors = 100 .* getindex.(young_laplace, "relative_error")
rayleigh_particles = getindex.(rayleigh, "particle_count")
rayleigh_errors = 100 .* getindex.(rayleigh, "frequency_error")

scatterlines!(young_laplace_axis, young_laplace_particles, young_laplace_errors;
              color=:navy, marker=:circle, label="CSS operator fit")
hlines!(young_laplace_axis, [5.0]; color=:firebrick, linestyle=:dash,
        label="5% acceptance")
scatterlines!(rayleigh_axis, rayleigh_particles, rayleigh_errors;
              color=:darkgreen, marker=:diamond, label="linear stiffness")
hlines!(rayleigh_axis, [5.0]; color=:firebrick, linestyle=:dash,
        label="5% acceptance")
applicable = stability[stability.admissible, :]
barplot!(stability_axis, 1:nrow(applicable), applicable.periods_completed;
         color=[:gray45, :darkorange, :dodgerblue3])
hlines!(stability_axis, [5.0]; color=:firebrick, linestyle=:dash,
        label="5-period gate")

axislegend(young_laplace_axis; position=:rt)
axislegend(rayleigh_axis; position=:rt)
axislegend(stability_axis; position=:rt)
save(joinpath(@__DIR__, "surface_tension_2d_validation.png"), figure)
figure
