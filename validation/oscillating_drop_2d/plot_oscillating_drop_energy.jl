include("../validation_util.jl")

# Activate for interactive plot
# using GLMakie
using CairoMakie
using Printf
using TrixiParticles
using TrixiParticles.CSV
using TrixiParticles.DataFrames
using TrixiParticles.JSON

save_figures = true
include_sim_results = true
include_trixiparticles_reference = true
include_paper_reference = true

# Particle spacing used in the filenames, e.g. 0.005 -> `dx_0p0050`.
# This setting does not affect the resolution-independent paper reference.
requested_resolution = 0.005

period = 4.827343
case_dir = joinpath(validation_dir(), "oscillating_drop_2d")

formatted_resolution = replace(@sprintf("%.4f", requested_resolution), "." => "p")
result_filename = "validation_result_oscillating_drop_2d_dx_$formatted_resolution.json"

fig = Figure(size=(900, 500))
ax = Axis(fig[1, 1],
          xlabel=L"t / T",
          ylabel="Relative energy",
          title="Oscillating Drop Energy Components")

lines!(ax, [NaN], [NaN]; color=:red, linewidth=2.5,
       label=L"\left(\mathcal{E}_{\text{tot}} - \mathcal{E}_{\text{tot}}^0\right) / \mathcal{E}_{\text{M}}^0")
lines!(ax, [NaN], [NaN]; color=:purple, linewidth=2.5,
       label=L"\left(\mathcal{E}_{\text{C}} - \mathcal{E}_{\text{C}}^0\right) / \mathcal{E}_{\text{M}}^0")
lines!(ax, [NaN], [NaN]; color=:blue, linewidth=2.5,
       label=L"\mathcal{E}_{\text{M}} / \mathcal{E}_{\text{M}}^0 - 1")
lines!(ax, [NaN], [NaN]; color=:green, linewidth=2.5,
       label=L"-Q_{\delta} / \mathcal{E}_{\text{M}}^0")

function plot_energy(filename, linestyle)
    json_data = JSON.parsefile(filename)

    time = json_data["kinetic_energy_fluid_1"]["time"]
    kinetic = json_data["kinetic_energy_fluid_1"]["values"]
    potential = json_data["potential_energy_fluid_1"]["values"]
    compressible = json_data["compressible_energy_fluid_1"]["values"]
    q_delta = json_data["q_delta_fluid_1"]["values"]

    mechanical = kinetic .+ potential
    mechanical_initial = first(mechanical)
    compressible_initial = first(compressible)

    internal_energy = compressible .- compressible_initial .+ q_delta
    total = mechanical .+ internal_energy
    total_initial = first(total)

    t_over_period = time ./ period
    total_relative = (total .- total_initial) ./ mechanical_initial
    compressible_relative = (compressible .- compressible_initial) ./ mechanical_initial
    mechanical_relative = mechanical ./ mechanical_initial .- 1
    q_delta_relative = -q_delta ./ mechanical_initial

    lines!(ax, t_over_period, total_relative;
           color=:red, linestyle, linewidth=2.5)
    lines!(ax, t_over_period, compressible_relative;
           color=:purple, linestyle, linewidth=2.5)
    lines!(ax, t_over_period, mechanical_relative;
           color=:blue, linestyle, linewidth=2.5)
    lines!(ax, t_over_period, q_delta_relative;
           color=:green, linestyle, linewidth=2.5)
end

if include_sim_results
    if isfile(joinpath("out", result_filename))
        plot_energy(joinpath("out", result_filename), :solid)
        lines!(ax, [NaN], [NaN]; color=:black, linestyle=:solid, linewidth=2.5,
               label="Simulation")
    else
        @warn "Simulation result file not found: out/$result_filename"
    end
end
if include_trixiparticles_reference
    if isfile(joinpath(case_dir, result_filename))
        plot_energy(joinpath(case_dir, result_filename), :dash)
        lines!(ax, [NaN], [NaN]; color=:black, linestyle=:dash, linewidth=2.5,
               label="TrixiParticles reference")
    else
        @warn "TrixiParticles reference file not found: $case_dir/$result_filename"
    end
end

if include_paper_reference
    reference_file = joinpath(case_dir, "reference_antuono_2015.csv")
    reference = CSV.read(reference_file, DataFrame; delim=';', normalizenames=true)

    lines!(ax, reference.t_T, reference.E_C;
           color=:purple, linestyle=:dot, linewidth=2.5)
    lines!(ax, reference.t_T, reference.E_M;
           color=:blue, linestyle=:dot, linewidth=2.5)
    lines!(ax, reference.t_T, reference.Q_delta;
           color=:green, linestyle=:dot, linewidth=2.5)
    lines!(ax, [NaN], [NaN]; color=:black, linestyle=:dot, linewidth=2.5,
           label="Antuono et al. (2015)")
end

xlims!(ax, 0, 12)
ylims!(ax, -0.018, 0.005)
axislegend(ax; position=:lb)

if save_figures
    output_file = "oscillating_drop_energy.svg"
    save(output_file, fig)
    println("Saved $output_file")
else
    display(fig)
end

fig
