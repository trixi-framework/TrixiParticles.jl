include("../validation_util.jl")

# Activate for interactive plot
# using GLMakie
using CairoMakie
using Glob
using TrixiParticles
using TrixiParticles.JSON

save_figures = true
include_sim_results = true

period = 4.827343
case_dir = joinpath(validation_dir(), "oscillating_drop_2d")

function find_series(json_data, quantity)
    matching_keys = sort(collect(filter(key -> startswith(key, quantity * "_"),
                                        keys(json_data))))
    isempty(matching_keys) && error("No series found for quantity `$quantity`")

    return json_data[first(matching_keys)]
end

function series_values(json_data, quantity)
    data = find_series(json_data, quantity)
    return Float64.(data["time"]), Float64.(data["values"])
end

input_files = isempty(ARGS) ?
              vcat(glob("validation_reference_oscillating_drop_2d*.json", case_dir),
                   include_sim_results ?
                   glob("validation_result_oscillating_drop_2d*.json", "out") : []) :
              ARGS

isempty(input_files) && error("No oscillating-drop validation files found")

input_file = isempty(ARGS) ? last(sort(input_files; by=mtime)) : first(input_files)
println("Plotting $input_file")

json_data = JSON.parsefile(input_file)

time, kinetic = series_values(json_data, "kinetic_energy")
_, potential = series_values(json_data, "potential_energy")
_, compressible = series_values(json_data, "compressible_energy")
_, q_delta = series_values(json_data, "q_delta")

mechanical = kinetic .+ potential
mechanical_initial = first(mechanical)
compressible_initial = first(compressible)

internal_energy = compressible .- compressible_initial .+ q_delta
total = mechanical .+ internal_energy
total_initial = first(total)

t_over_period = time ./ period
mechanical_relative = mechanical ./ mechanical_initial .- 1
compressible_relative = (compressible .- compressible_initial) ./ mechanical_initial
q_delta_relative = -q_delta ./ mechanical_initial
total_relative = (total .- total_initial) ./ mechanical_initial

fig = Figure(size=(1200, 650))
ax = Axis(fig[1, 1],
          xlabel="t / T",
          ylabel="Relative energy",
          title="Oscillating Drop Energy Components")

lines!(ax, t_over_period, total_relative;
       color=:red, linewidth=2.5,
       label="(E_tot - E_tot0) / E_M0")
lines!(ax, t_over_period, compressible_relative;
       color=:purple, linewidth=2.5,
       label="(E_C - E_C0) / E_M0")
lines!(ax, t_over_period, mechanical_relative;
       color=:blue, linestyle=:dash, linewidth=2.5,
       label="E_M / E_M0 - 1")
lines!(ax, t_over_period, q_delta_relative;
       color=:green, linewidth=2.5,
       label="-Q_delta / E_M0")

xlims!(ax, 0, maximum(t_over_period))
ylims!(ax, -0.018, 0.005)
axislegend(ax; position=:rt)

if save_figures
    output_file = "oscillating_drop_energy.svg"
    save(output_file, fig)
    println("Saved $output_file")
else
    display(fig)
end

fig
