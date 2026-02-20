include("../validation_util.jl")
using CairoMakie
using CairoMakie.Colors
using JSON
using Glob
using TrixiParticles

# ==========================================================================================
# ==== Color adjustment functions
function lighten(c::Colorant, amount::Real)
    c_hsl = convert(HSL, c)
    new_l = clamp(c_hsl.l + amount, 0, 1)
    return HSL(c_hsl.h, c_hsl.s, new_l)
end

function darken(c::Colorant, amount::Real)
    c_hsl = convert(HSL, c)
    new_l = clamp(c_hsl.l - amount, 0, 1)
    return HSL(c_hsl.h, c_hsl.s, new_l)
end

# ==========================================================================================
# ==== JSON helpers
function require_key(json_data, key, json_file)
    if !haskey(json_data, key)
        available = sort(collect(keys(json_data)))
        available_list = join(available, ", ")
        error("Missing key \"$(key)\" in $json_file. Available keys: $available_list.")
    end
    return json_data[key]
end

function require_fields(data, fields, key, json_file)
    missing_fields = [field for field in fields if !haskey(data, field)]
    if !isempty(missing_fields)
        available = sort(collect(keys(data)))
        missing_list = join(missing_fields, ", ")
        available_list = join(available, ", ")
        error("Key \"$(key)\" in $json_file missing fields $missing_list. " *
              "Available fields: $available_list.")
    end
end

# ==========================================================================================
# ==== Directories, files, and global y-limits for simulation plots
include_reference_files = true
include_out_files = true

case_dir = joinpath(validation_dir(), "hydrostatic_water_column_2d")
edac_reference_files = include_reference_files ?
                       sort(glob("validation_reference_edac*.json", case_dir),
                            by=extract_number_from_filename) : String[]
wcsph_reference_files = include_reference_files ?
                        sort(glob("validation_reference_wcsph*.json", case_dir),
                             by=extract_number_from_filename) : String[]
edac_out_files = include_out_files ?
                 sort(glob("validation_result_hyd_edac*.json", "out/"),
                      by=extract_number_from_filename) : String[]
wcsph_out_files = include_out_files ?
                  sort(glob("validation_result_hyd_wcsph*.json", "out/"),
                       by=extract_number_from_filename) : String[]

edac_files = vcat(edac_reference_files, edac_out_files)
wcsph_files = vcat(wcsph_reference_files, wcsph_out_files)
all_files = vcat(edac_files, wcsph_files)

global_ymin = -Inf
global_ymax = Inf
for file in all_files
    json_data = JSON.parsefile(file)
    deflection_data = require_key(json_data, "y_deflection_structure_1", file)
    require_fields(deflection_data, ["time", "values"], "y_deflection_structure_1", file)
    time_vals = deflection_data["time"]
    inds = findall(t -> t <= 0.5, time_vals)
    global global_ymin,
           global_ymax = extrema(deflection_data["values"][inds])
    # Analytical solution is constant
    analytical_data = require_key(json_data, "analytical_solution", file)
    require_fields(analytical_data, ["values"], "analytical_solution", file)
    global_ymin = min(global_ymin, analytical_data["values"][1])
    global_ymax = max(global_ymax, analytical_data["values"][1])
end

# ==========================================================================================
# ==== Map resolution labels to colors
resolution_labels = map(file -> extract_number_from_filename(file), all_files)
unique_resolution = sort(unique(resolution_labels))
cmap = cgrad(:tab10, length(unique_resolution))
sim_color_map = Dict{Int, RGB}()
avg_color_map = Dict{Int, RGB}()
for (i, resolution) in enumerate(unique_resolution)
    base_color = cmap[i]
    sim_color_map[resolution] = lighten(base_color, 0.15)
    # Make the average markers slightly darker than the simulation plots
    avg_color_map[resolution] = darken(base_color, 0.15)
end

# ==========================================================================================
# ==== Compute error metrics (averaging over t in [0.25, 0.5])
D = 67.5e9 * (0.05)^3 / (12 * (1 - 0.3^2))
analytical_value = -0.0026 * 9.81 * (1000 * 2.0 + 2700 * 0.05) / D

function compute_errors(json_file)
    json_data = JSON.parsefile(json_file)
    deflection_data = require_key(json_data, "y_deflection_structure_1", json_file)
    require_fields(deflection_data, ["time", "values"], "y_deflection_structure_1",
                   json_file)
    time_vals = deflection_data["time"]
    sim_vals = deflection_data["values"]
    inds = findall(t -> 0.25 <= t <= 1.0, time_vals)
    avg_sim = sum(sim_vals[inds]) / length(sim_vals[inds])
    abs_err = abs(avg_sim - analytical_value)
    rel_err = abs_err / abs(analytical_value)
    res_int = extract_number_from_filename(json_file)
    res_val = res_int == -1 ? missing : Float64(res_int)
    return res_val, abs_err, rel_err
end

function collect_errors(files)
    resolutions = Float64[]
    abs_errs = Float64[]
    rel_errs = Float64[]
    for file in files
        res_val, abs_err, rel_err = compute_errors(file)
        push!(resolutions, res_val !== missing ? res_val : length(resolutions) + 1.0)
        push!(abs_errs, abs_err)
        push!(rel_errs, rel_err)
    end

    sorted = sort(collect(zip(resolutions, abs_errs, rel_errs)),
                  lt=(a, b) -> a[1] < b[1])
    res_sorted = [x for (x, _, _) in sorted]
    abs_sorted = [y for (_, y, _) in sorted]
    rel_sorted = [z for (_, _, z) in sorted]
    return res_sorted, abs_sorted, rel_sorted
end

edac_ref_res, edac_ref_abs_err, edac_ref_rel_err = collect_errors(edac_reference_files)
edac_out_res, edac_out_abs_err, edac_out_rel_err = collect_errors(edac_out_files)
wcsph_ref_res, wcsph_ref_abs_err, wcsph_ref_rel_err = collect_errors(wcsph_reference_files)
wcsph_out_res, wcsph_out_abs_err, wcsph_out_rel_err = collect_errors(wcsph_out_files)

# Rescale resolution for error plots (example: nondimensionalize by 0.05)
edac_ref_res = 0.05 ./ edac_ref_res
edac_out_res = 0.05 ./ edac_out_res
wcsph_ref_res = 0.05 ./ wcsph_ref_res
wcsph_out_res = 0.05 ./ wcsph_out_res

# ==========================================================================================
# ==== Create figure and layout (3 rows x 2 columns)
# Row 1: Reference simulation plots (data for t in [0, 0.5])
# Row 2: Out simulation plots (data for t in [0, 0.5])
# Row 3: Error plots
fig = Figure(size=(1200, 1000), padding=(10, 10, 10, 10))
ax_edac_ref = Axis(fig[1, 1], title="Hydrostatic Water Column: EDAC (Reference)",
                   height=260)
ax_wcsph_ref = Axis(fig[1, 2], title="Hydrostatic Water Column: WCSPH (Reference)",
                    height=260)
ax_edac_out = Axis(fig[2, 1], title="Hydrostatic Water Column: EDAC (Out)",
                   height=260)
ax_wcsph_out = Axis(fig[2, 2], title="Hydrostatic Water Column: WCSPH (Out)",
                    height=260)
ax_abs = Axis(fig[3, 1], title="Absolute Error", xlabel="Resolution",
              ylabel="Absolute Error", height=200)
ax_rel = Axis(fig[3, 2], title="Relative Error", xlabel="Resolution",
              ylabel="Relative Error", height=200)

# ==========================================================================================
# ==== Simulation plot function
reference_sim_linestyle = :solid
out_sim_linestyle = :dashdot
avg_line_style = (:dot, :loose)

function plot_dataset!(ax, json_file; source=:reference, show_sim_label=false,
                       show_avg_label=false, show_analytic=false)
    json_data = JSON.parsefile(json_file)
    deflection_data = require_key(json_data, "y_deflection_structure_1", json_file)
    require_fields(deflection_data, ["time", "values"], "y_deflection_structure_1",
                   json_file)
    time_vals = deflection_data["time"]
    sim_vals = deflection_data["values"]
    # Analytical solution is constant
    analytical_data = require_key(json_data, "analytical_solution", json_file)
    require_fields(analytical_data, ["values"], "analytical_solution", json_file)
    analytic_val = analytical_data["values"][1]

    inds_plot = findall(t -> t <= 0.5, time_vals)
    time_plot = time_vals[inds_plot]
    sim_plot = sim_vals[inds_plot]
    analytic_plot = fill(analytic_val, length(time_plot))

    resolution = extract_number_from_filename(json_file)
    sim_col = sim_color_map[resolution]
    sim_linestyle = source == :reference ? reference_sim_linestyle : out_sim_linestyle

    inds_avg = findall(t -> 0.25 <= t <= 0.5, time_vals)
    avg_sim = sum(sim_vals[inds_avg]) / length(sim_vals[inds_avg])
    avg_col = source == :reference ? darken(sim_col, 0.2) : darken(sim_col, 0.35)

    sim_label = show_sim_label ? "t/dp=$(resolution)" : nothing
    avg_label = show_avg_label ? "Sim avg (0.25≤t≤0.5)" : nothing
    analytic_label = show_analytic ? "Analytical" : nothing

    lines!(ax, time_plot, sim_plot; color=sim_col, linestyle=sim_linestyle, linewidth=2,
           label=sim_label)
    if show_analytic
        lines!(ax, time_plot, analytic_plot; color=:black, linestyle=:dash, linewidth=3,
               label=analytic_label)
    end
    lines!(ax, [time_plot[1], time_plot[end]], [avg_sim, avg_sim];
           color=avg_col, linestyle=avg_line_style, linewidth=4, label=avg_label)
end

# Plot simulation datasets
for (i, file) in enumerate(edac_reference_files)
    plot_dataset!(ax_edac_ref, file; source=:reference, show_sim_label=true,
                  show_avg_label=i == 1, show_analytic=i == 1)
end
for (i, file) in enumerate(edac_out_files)
    plot_dataset!(ax_edac_out, file; source=:out, show_analytic=i == 1)
end
for (i, file) in enumerate(wcsph_reference_files)
    plot_dataset!(ax_wcsph_ref, file; source=:reference, show_sim_label=true,
                  show_avg_label=i == 1, show_analytic=i == 1)
end
for (i, file) in enumerate(wcsph_out_files)
    plot_dataset!(ax_wcsph_out, file; source=:out, show_analytic=i == 1)
end

for ax in (ax_edac_ref, ax_wcsph_ref, ax_edac_out, ax_wcsph_out)
    xlims!(ax, 0, 0.5)
    ylims!(ax, global_ymin, global_ymax)
    ax.xlabel = "Time [s]"
    ax.ylabel = "Vertical Deflection [m]"
end

axislegend(ax_edac_ref; position=:rb, nbanks=2)

# ==========================================================================================
function plot_error_series!(ax_abs, ax_rel, res, abs_err, rel_err;
                            color, marker, linestyle, label)
    isempty(res) && return
    scatter!(ax_abs, res, abs_err; marker, markersize=10, color, label=nothing)
    lines!(ax_abs, res, abs_err; color, linestyle, linewidth=2, label)
    scatter!(ax_rel, res, rel_err; marker, markersize=10, color, label=nothing)
    lines!(ax_rel, res, rel_err; color, linestyle, linewidth=2, label=nothing)
end

# ==========================================================================================
# ==== Reference data
#
# Fig. 8 in:
#   "A fluid-structure interaction model for free-surface flows and flexible structures
#   using smoothed particle hydrodynamics on a GPU".
#   J. O'Connor and B.D. Rogers, Journal of Fluids and Structures.
#   https://doi.org/10.1016/j.jfluidstructs.2021.103312
reference_res = [0.0025, 0.005, 0.01]
reference_error = [8e-7, 6e-6, 1e-5]
scatter!(ax_abs, reference_res, reference_error; marker=:diamond, markersize=10,
         color=:black, label=nothing)
lines!(ax_abs, reference_res, reference_error; color=:black, linestyle=:solid, linewidth=2,
       label="Literature")

# Plot EDAC errors (blue, circle)
plot_error_series!(ax_abs, ax_rel, edac_ref_res, edac_ref_abs_err, edac_ref_rel_err;
                   color=:blue, marker=:circle, linestyle=reference_sim_linestyle,
                   label="EDAC (reference)")
plot_error_series!(ax_abs, ax_rel, edac_out_res, edac_out_abs_err, edac_out_rel_err;
                   color=:blue, marker=:circle, linestyle=out_sim_linestyle,
                   label="EDAC (out)")

# Plot WCSPH errors (red, xcross)
plot_error_series!(ax_abs, ax_rel, wcsph_ref_res, wcsph_ref_abs_err, wcsph_ref_rel_err;
                   color=:red, marker=:xcross, linestyle=reference_sim_linestyle,
                   label="WCSPH (reference)")
plot_error_series!(ax_abs, ax_rel, wcsph_out_res, wcsph_out_abs_err, wcsph_out_rel_err;
                   color=:red, marker=:xcross, linestyle=out_sim_linestyle,
                   label="WCSPH (out)")

axislegend(ax_abs; position=:rb, nbanks=2)

# ==========================================================================================
# ==== Display and save figure
save("hydrostatic_water_column_validation.svg", fig)
fig
