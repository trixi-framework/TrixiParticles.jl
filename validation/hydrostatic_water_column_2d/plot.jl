include("../validation_util.jl")
using CairoMakie, JSON, Glob, Statistics, Printf, TrixiParticles, Colors

# --- Define color adjustment functions ---
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

# === Directories and Files ===============================================
case_dir = joinpath(validation_dir(), "hydrostatic_water_column_2d")
edac_files = sort(glob("validation_reference_edac*.json", case_dir),
                  by=extract_number_from_filename)
wcsph_files = sort(glob("validation_reference_wcsph*.json", case_dir),
                   by=extract_number_from_filename)


# edac_sim_files = sort(glob("validation_result_hyd_edac*.json", "out/"),
#                    by=extract_number_from_filename)
# wcsph_sim_files = sort(glob("validation_result_hyd_wcsph*.json", "out/"),
#                     by=extract_number_from_filename)

# === Global y-axis limits for simulation plots ===========================
all_y = Float64[]
for file in vcat(edac_files, wcsph_files)
    json_data = JSON.parsefile(file)
    time_vals = json_data["y_deflection_solid_1"]["time"]
    inds = findall(t -> t <= 0.5, time_vals)
    append!(all_y, json_data["y_deflection_solid_1"]["values"][inds])
    # Push the constant analytical solution value (do not index with inds)
    push!(all_y, json_data["analytical_solution"]["values"][1])
end
global_ymin = minimum(all_y)
global_ymax = maximum(all_y)

# === Build mapping from resolution label to color =========================
res_labels = map(file -> extract_resolution_from_filename(file), edac_files)
unique_res = sort(unique(res_labels))
cmap = cgrad(:tab10, length(unique_res))
sim_color_map = Dict{String, RGB}()
avg_color_map = Dict{String, RGB}()
for (i, res) in enumerate(unique_res)
    base_color = cmap[i]
    # Use a small lighten factor so red remains red.
    light_color = lighten(base_color, 0.15)
    dark_color  = darken(base_color, 0.15)
    sim_color_map[res] = light_color
    avg_color_map[res] = dark_color
end

# === Compute error metrics for EDAC and WCSPH =============================
# Errors are computed using the simulation average over t in [0.25, 0.5].
D = 67.5e9 * (0.05)^3 / (12 * (1 - 0.3^2))   # parameters from simulation script
analytical_value = -0.0026 * 9.81 * (1000*2.0 + 2700*0.05) / D

function compute_errors(json_file)
    json_data = JSON.parsefile(json_file)
    time_vals = json_data["y_deflection_solid_1"]["time"]
    sim_vals  = json_data["y_deflection_solid_1"]["values"]
    inds = findall(t -> (0.25 <= t <= 0.5), time_vals)
    avg_sim = mean(sim_vals[inds])
    abs_err = abs(avg_sim - analytical_value)
    rel_err = abs_err / abs(analytical_value)
    res_str = extract_resolution_from_filename(json_file)
    res_val = try parse(Float64, res_str) catch missing end
    return res_val, abs_err, rel_err
end

edac_res = Float64[]; edac_abs_err = Float64[]; edac_rel_err = Float64[]
for file in edac_files
    res_val, abs_err, rel_err = compute_errors(file)
    push!(edac_res, res_val !== missing ? res_val : length(edac_res)+1.0)
    push!(edac_abs_err, abs_err)
    push!(edac_rel_err, rel_err)
end

wcsph_res = Float64[]; wcsph_abs_err = Float64[]; wcsph_rel_err = Float64[]
for file in wcsph_files
    res_val, abs_err, rel_err = compute_errors(file)
    push!(wcsph_res, res_val !== missing ? res_val : length(wcsph_res)+1.0)
    push!(wcsph_abs_err, abs_err)
    push!(wcsph_rel_err, rel_err)
end

edac_sorted = sort(collect(zip(edac_res, edac_abs_err, edac_rel_err)), lt = (a, b) -> a[1] < b[1])
edac_res_sorted = [x for (x, _, _) in edac_sorted]
edac_abs_err_sorted = [y for (_, y, _) in edac_sorted]
edac_rel_err_sorted = [z for (_, _, z) in edac_sorted]

wcsph_sorted = sort(collect(zip(wcsph_res, wcsph_abs_err, wcsph_rel_err)), lt = (a, b) -> a[1] < b[1])
wcsph_res_sorted = [x for (x, _, _) in wcsph_sorted]
wcsph_abs_err_sorted = [y for (_, y, _) in wcsph_sorted]
wcsph_rel_err_sorted = [z for (_, _, z) in wcsph_sorted]

# === Create the figure and layout ==========================================
# Layout: 4 rows, 2 columns.
# Row 1: Simulation plots (we display data only for t in [0,0.5]).
# Row 2: Simulation legend.
# Row 3: Error plots.
# Row 4: Error legend.
#
# We set the height of row 1 (simulation axes) to 300 and row 3 (error axes) to 200.
fig = Figure(size = (1200, 800), padding = (10, 10, 10, 10))
ax_edac   = Axis(fig[1, 1], title = "Hydrostatic Water Column: EDAC", height = 300)
ax_wcsph  = Axis(fig[1, 2], title = "Hydrostatic Water Column: WCSPH", height = 300)
# Restrict x-axis for simulation plots to [0,0.5].
xlims!(ax_edac, 0, 0.5)
xlims!(ax_wcsph, 0, 0.5)

# === Function to plot simulation datasets (for t in [0,0.5]) ===================
function plot_dataset!(ax, json_file)
    json_data = JSON.parsefile(json_file)
    time_vals  = json_data["y_deflection_solid_1"]["time"]
    sim_vals   = json_data["y_deflection_solid_1"]["values"]
    # Create a constant vector from the analytical solution value.
    anal_val = json_data["analytical_solution"]["values"][1]

    # Only plot data for t <= 0.5.
    inds_plot = findall(t -> t <= 0.5, time_vals)
    time_plot = time_vals[inds_plot]
    sim_plot  = sim_vals[inds_plot]
    anal_plot = fill(anal_val, length(time_plot))

    res = extract_resolution_from_filename(json_file)
    sim_col = sim_color_map[res]

    # Compute simulation average using data in [0.25, 0.5].
    inds_avg = findall(t -> (0.25 <= t <= 0.5), time_vals)
    avg_sim = mean(sim_vals[inds_avg])

    lines!(ax, time_plot, sim_plot; color = sim_col, linestyle = :solid, linewidth = 2)
    lines!(ax, time_plot, anal_plot; color = :black, linestyle = :solid, linewidth = 4)
    lines!(ax, [time_plot[1], time_plot[end]], [avg_sim, avg_sim];
           color = darken(sim_col, 0.2), linestyle = :dot, linewidth = 4)
end

# Plot simulation datasets.
for file in edac_files
    plot_dataset!(ax_edac, file)
end
for file in wcsph_files
    plot_dataset!(ax_wcsph, file)
end

ylims!(ax_edac, global_ymin, global_ymax)
ylims!(ax_wcsph, global_ymin, global_ymax)
for ax in (ax_edac, ax_wcsph)
    ax.xlabel = "Time [s]"
    ax.ylabel = "Vertical Deflection [m]"
end

# === Row 2: Build common simulation legend ================================
# Create dummy plots for the legend on a dummy scene.
sim_dummy = Scene()
analytical_ref = lines!(sim_dummy, [0.0, 1.0], [0.0, 0.0]; color = :black, linestyle = :solid, linewidth = 4)
avg_ref = lines!(sim_dummy, [0.0, 1.0], [0.0, 0.0]; color = :black, linestyle = :dot, linewidth = 4)
sim_refs = [ lines!(sim_dummy, [0.0, 1.0], [0.0, 0.0]; color = sim_color_map[res], linestyle = :solid, linewidth = 2)
             for res in unique_res ]
group_entries = [ [analytical_ref], sim_refs, [avg_ref] ]
group_labels  = [ ["Analytical"], [ "Simulation (t/dp=$(res))" for res in unique_res ], ["Sim avg (0.25≤t≤0.5)"] ]
group_titles  = [ "Line Style", "Color", "Mean Style" ]
sim_leg = Legend(fig, group_entries, group_labels, group_titles; orientation = :horizontal, tellwidth = false)
fig[2, 1:2] = sim_leg

# === Row 3: Error plots ====================================================
ax_abs = Axis(fig[3, 1], title = "Absolute Error", xlabel = "Resolution", ylabel = "Absolute Error", height = 200)
ax_rel = Axis(fig[3, 2], title = "Relative Error", xlabel = "Resolution", ylabel = "Relative Error", height = 200)

wcsph_res_sorted = 0.05 ./ wcsph_res_sorted
edac_res_sorted = 0.05 ./ edac_res_sorted



############################################################################################
# Data extracted from Fig. 8 in
# "A fluid–structure interaction model for free-surface flows and flexible structures
# using smoothed particle hydrodynamics on a GPU" by J. O'Connor and B.D. Rogers
# published in Journal of Fluids and Structures
# https://doi.org/10.1016/j.jfluidstructs.2021.103312
############################################################################################
reference_res = [0.0025, 0.005, 0.01]
reference_error = [8E-7, 6E-6 ,1E-5]


scatter!(ax_abs, reference_res, reference_error; marker = :diamond, markersize = 10, color = :black)
lines!(ax_abs, reference_res, reference_error; color = :black, linestyle = :solid, linewidth = 2)

scatter!(ax_abs, edac_res_sorted, edac_abs_err_sorted; marker = :circle, markersize = 10, color = :blue)
lines!(ax_abs, edac_res_sorted, edac_abs_err_sorted; color = :blue, linestyle = :solid, linewidth = 2)
scatter!(ax_abs, wcsph_res_sorted, wcsph_abs_err_sorted; marker = :xcross, markersize = 10, color = :red)
lines!(ax_abs, wcsph_res_sorted, wcsph_abs_err_sorted; color = :red, linestyle = :solid, linewidth = 2)

scatter!(ax_rel, edac_res_sorted, edac_rel_err_sorted; marker = :circle, markersize = 10, color = :blue)
lines!(ax_rel, edac_res_sorted, edac_rel_err_sorted; color = :blue, linestyle = :solid, linewidth = 2)
scatter!(ax_rel, wcsph_res_sorted, wcsph_rel_err_sorted; marker = :xcross, markersize = 10, color = :red)
lines!(ax_rel, wcsph_res_sorted, wcsph_rel_err_sorted; color = :red, linestyle = :solid, linewidth = 2)

# Row 4: Build error legend.
dummy_err = Scene()
edac_marker = scatter!(dummy_err, [0.0], [0.0]; marker = :circle, markersize = 10, color = :blue)
wcsph_marker = scatter!(dummy_err, [0.0], [0.0]; marker = :xcross, markersize = 10, color = :red)
reference_marker = scatter!(dummy_err, [0.0], [0.0]; marker = :diamond, markersize = 10, color = :black)

err_group_entries = [ [edac_marker, wcsph_marker, reference_marker]]
err_group_labels  = [ ["EDAC", "WCSPH", "Reference"]]
err_group_titles  = [ "Method"]
err_leg = Legend(fig, err_group_entries, err_group_labels, err_group_titles; orientation = :horizontal, tellwidth = false)
fig[4, 1:2] = err_leg

fig
save("hydrostatic_water_column_validation.svg", fig)
