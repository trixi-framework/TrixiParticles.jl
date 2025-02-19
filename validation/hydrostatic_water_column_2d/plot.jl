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
    append!(all_y, json_data["y_deflection_solid_1"]["values"])
    append!(all_y, json_data["analytical_solution"]["values"])
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
# (Errors computed from "y_deflection_solid_1" for t > 0.5 versus constant analytical value.)
D = 67.5e9 * (0.05)^3 / (12 * (1 - 0.3^2))   # using parameters from simulation script
analytical_value = -0.0026 * 9.81 * (1000*2.0 + 2700*0.05) / D

function compute_errors(json_file)
    json_data = JSON.parsefile(json_file)
    time_vals = json_data["y_deflection_solid_1"]["time"]
    sim_vals  = json_data["y_deflection_solid_1"]["values"]
    inds = findall(t -> t > 0.5, time_vals)
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
# Row 1: Simulation plots; Row 2: Simulation legend; Row 3: Error plots; Row 4: Error legend.
fig = Figure(size = (1200, 800), padding = (10, 10, 10, 10))
# Row 1: Simulation plots.
ax_edac   = Axis(fig[1, 1], title = "Hydrostatic Water Column: EDAC")
ax_wcsph  = Axis(fig[1, 2], title = "Hydrostatic Water Column: WCSPH")

# === Function to plot simulation datasets =================================
function plot_dataset!(ax, json_file)
    json_data = JSON.parsefile(json_file)
    time_vals  = json_data["y_deflection_solid_1"]["time"]
    sim_vals   = json_data["y_deflection_solid_1"]["values"]
    anal_vals  = json_data["analytical_solution"]["values"]

    res = extract_resolution_from_filename(json_file)
    sim_col = sim_color_map[res]

    inds = findall(t -> t > 0.5, time_vals)
    avg_sim = mean(sim_vals[inds])

    lines!(ax, time_vals, sim_vals; color = sim_col, linestyle = :solid, linewidth = 2)
    lines!(ax, time_vals, anal_vals; color = :black, linestyle = :solid, linewidth = 4)
    lines!(ax, [time_vals[1], last(time_vals)], [avg_sim, avg_sim];
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
    ax.xlabel = "Time"
    ax.ylabel = "Vertical Deflection"
end

# Row 2: Build common simulation legend.
dummy_scene = Scene()
analytical_ref = lines!(dummy_scene, [0.0, 1.0], [0.0, 0.0]; color = :black, linestyle = :solid, linewidth = 4)
avg_ref = lines!(dummy_scene, [0.0, 1.0], [0.0, 0.0]; color = :black, linestyle = :dot, linewidth = 4)
sim_refs = [ lines!(dummy_scene, [0.0, 1.0], [0.0, 0.0]; color = sim_color_map[res], linestyle = :solid, linewidth = 2)
             for res in unique_res ]

group_entries = [ [analytical_ref], sim_refs, [avg_ref] ]
group_labels  = [ ["Analytical"], [ "Simulation (dp=$(res))" for res in unique_res ], ["Sim avg (t>0.5)"] ]
group_titles  = [ "Line Style", "Color", "Mean Style" ]

sim_leg = Legend(fig, group_entries, group_labels, group_titles; orientation = :horizontal, tellwidth = false)
fig[2, 1:2] = sim_leg

# Row 3: Error plots.
ax_abs = Axis(fig[3, 1], title = "Absolute Error", xlabel = "Resolution", ylabel = "Absolute Error")
ax_rel = Axis(fig[3, 2], title = "Relative Error", xlabel = "Resolution", ylabel = "Relative Error")

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

err_group_entries = [ [edac_marker, wcsph_marker]]
err_group_labels  = [ ["EDAC", "WCSPH"]]
err_group_titles  = [ "Method"]
err_leg = Legend(fig, err_group_entries, err_group_labels, err_group_titles; orientation = :horizontal, tellwidth = false)
fig[4, 1:2] = err_leg

fig
save("hydrostatic_water_column_validation.svg", fig)
