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


# === Global y-axis limits ===============================================
all_y = Float64[]
for file in vcat(edac_files, wcsph_files)
    json_data = JSON.parsefile(file)
    append!(all_y, json_data["y_deflection_solid_1"]["values"])
    append!(all_y, json_data["analytical_solution"]["values"])
end
global_ymin = minimum(all_y)
global_ymax = maximum(all_y)

# === Build a mapping from resolution label to color ======================
res_labels = map(file -> extract_resolution_from_filename(file), edac_files)
unique_res = sort(unique(res_labels))

# Create a colormap using :tab10.
cmap = cgrad(:tab10, length(unique_res))

# For each resolution, assign a light color (for simulation curves)
# and a dark color (for the average line). Use a smaller lighten factor to avoid white.
sim_color_map = Dict{String, RGB}()
avg_color_map = Dict{String, RGB}()
for (i, res) in enumerate(unique_res)
    base_color = cmap[i]
    light_color = lighten(base_color, 0.15)  # e.g. lighten by 0.15
    dark_color  = darken(base_color, 0.15)   # darken by 0.15
    sim_color_map[res] = light_color
    avg_color_map[res] = dark_color
end

# === Create the figure and axes ===============================================
fig = Figure(size = (1200, 600))
ax_edac   = Axis(fig[1, 1], title="Hydrostatic Water Column: EDAC")
ax_wcsph  = Axis(fig[1, 2], title="Hydrostatic Water Column: WCSPH")

# === Function to plot one dataset ======================================
function plot_dataset!(ax, json_file)
    json_data = JSON.parsefile(json_file)
    time_vals  = json_data["y_deflection_solid_1"]["time"]
    sim_vals   = json_data["y_deflection_solid_1"]["values"]
    anal_vals  = json_data["analytical_solution"]["values"]

    # Determine the resolution label for this file.
    res = extract_resolution_from_filename(json_file)
    sim_col = sim_color_map[res]

    # Compute simulation average for times > 0.5.
    inds = findall(t -> t > 0.5, time_vals)
    avg_sim = mean(sim_vals[inds])

    # Plot simulation curve (light color).
    lines!(ax, time_vals, sim_vals; color = sim_col, linestyle = :solid, linewidth = 2)
    # Plot analytical solution (black solid).
    lines!(ax, time_vals, anal_vals; color = :black, linestyle = :solid, linewidth = 4)
    # Plot horizontal line for simulation average (t>0.5) in darkened color.
    lines!(ax, [time_vals[1], last(time_vals)], [avg_sim, avg_sim];
           color = darken(sim_col, 0.2), linestyle = :dot, linewidth = 4)
end

# === Plot datasets on each axis ======================================
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

# === Build a common multigroup legend ======================================
# Create dummy line plots for legend entries on a separate (dummy) scene.
dummy_scene = Scene()  # separate scene to create dummy plots

# Group 1: "Line Style": Analytical (black solid) and Simulation Average (black dotted)
analytical_ref = lines!(dummy_scene, [0.0, 1.0], [0.0, 0.0]; color = :black, linestyle = :solid, linewidth = 4)
avg_ref        = lines!(dummy_scene, [0.0, 1.0], [0.0, 0.0]; color = :black, linestyle = :dot, linewidth = 4)

# Group 2: "Color": One entry per resolution (simulation curves using light colors)
sim_refs = [ lines!(dummy_scene, [0.0, 1.0], [0.0, 0.0]; color = sim_color_map[res], linestyle = :solid, linewidth = 2)
             for res in unique_res ]

group_entries = [ [analytical_ref], sim_refs, [avg_ref] ]
group_labels  = [ ["Analytical"], [ "Simulation (dp=$(res))" for res in unique_res ], ["Sim avg (t>0.5)"] ]
group_titles  = [ "Line Style", "Color", "Mean Style" ]

# Create the Legend and place it in an empty cell (e.g. fig[3,1]).
leg = Legend(fig, group_entries, group_labels, group_titles; orientation = :horizontal, tellwidth = false)
fig[3, 1] = leg

fig
save("hydrostatic_water_column_validation.svg", fig)
