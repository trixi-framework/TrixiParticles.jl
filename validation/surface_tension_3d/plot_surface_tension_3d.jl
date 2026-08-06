using CairoMakie
using CSV
using DataFrames
using JSON
using Statistics

reference = JSON.parsefile(joinpath(@__DIR__, "validation_reference.json"))
young_laplace = reference["young_laplace"]["results"]
sessile = CSV.read(joinpath(@__DIR__, "sessile_drop_matrix.csv"), DataFrame)
scorecard = CSV.read(joinpath(@__DIR__, "contact_angle_scorecard.csv"), DataFrame)
perturbation = CSV.read(joinpath(@__DIR__, "contact_angle_perturbation.csv"), DataFrame)
cost = CSV.read(joinpath(@__DIR__, "contact_angle_cost.csv"), DataFrame)
normal_components = CSV.read(joinpath(@__DIR__, "contact_angle_normal_components.csv"),
                             DataFrame)
force_sign = CSV.read(joinpath(@__DIR__, "contact_angle_force_sign.csv"), DataFrame)
ghost_sign = CSV.read(joinpath(@__DIR__, "contact_angle_force_sign_ghost_geometric.csv"),
                      DataFrame)
wall_energy_sign = CSV.read(joinpath(@__DIR__, "contact_angle_force_sign_wall_energy.csv"),
                            DataFrame)
cap_transfer = CSV.read(joinpath(@__DIR__, "contact_line_cap_transfer.csv"), DataFrame)
wetted_area = CSV.read(joinpath(@__DIR__, "wetted_area_measure.csv"), DataFrame)
wetted_area_sign = CSV.read(joinpath(@__DIR__,
                                     "contact_angle_force_sign_wetted_area.csv"), DataFrame)
measure_protocol = CSV.read(joinpath(@__DIR__, "contact_measure_protocol.csv"), DataFrame)
corrected_wetted_area = CSV.read(joinpath(@__DIR__, "wetted_area_corrected.csv"),
                                 DataFrame)
extended_recovery = CSV.read(joinpath(@__DIR__, "contact_angle_recovery_extended.csv"),
                             DataFrame)
extended_sign = CSV.read(joinpath(@__DIR__, "contact_angle_force_sign_extended.csv"),
                         DataFrame)
r4_perturbation = CSV.read(joinpath(@__DIR__,
                                    "contact_angle_perturbation_r4_wetted_area_extended_production.csv"),
                           DataFrame)
r4_cost = CSV.read(joinpath(@__DIR__,
                            "contact_angle_cost_r4_wetted_area_active_production.csv"),
                   DataFrame)

figure = Figure(size=(1450, 1250))
young_laplace_axis = Axis(figure[1, 1];
                          title="3D Young-Laplace convergence",
                          xlabel="particle count", ylabel="fitted sigma error [%]",
                          xscale=log10, yscale=log10)
geometric_axis = Axis(figure[1, 2]; title="Geometric contact angle",
                      xlabel="target angle [deg]", ylabel="local-fit angle [deg]",
                      limits=(20, 160, 20, 160))
clf_axis = Axis(figure[1, 3]; title="Contact-line force",
                xlabel="target angle [deg]", ylabel="local-fit angle [deg]",
                limits=(20, 160, 20, 160))
mae_axis = Axis(figure[2, 1]; title="Static angle error",
                xlabel="particle count", ylabel="mean absolute error [deg]",
                xscale=log10)
response_axis = Axis(figure[2, 2]; title="Off-target restoring response",
                     xlabel="target <- initial [deg]", ylabel="error reduction [%]",
                     xticks=(1:4, ["60 <- 90", "90 <- 60", "90 <- 120", "120 <- 90"]))
cost_axis = Axis(figure[2, 3]; title="Repeated solver cost",
                 xlabel="mechanism", ylabel="runtime / no-contact runtime",
                 xticks=(1:4, ["none", "geometric", "CLF", "R4-W"]))
normal_axis = Axis(figure[3, 1]; title="CLF normal diagnostics",
                   xlabel="local-circle angle [deg]", ylabel="weighted normal angle [deg]",
                   limits=(20, 160, 20, 160))
line_axis = Axis(figure[3, 2]; title="Contact-line normalization",
                 xlabel="target angle [deg]", ylabel="line-length error [%]")
sign_axis = Axis(figure[3, 3]; title="Fixed signs (measure-gated separately)",
                 xlabel="validation-only variant", ylabel="passing cases",
                 xticks=(1:9,
                         ["geometric", "CLF", "ghost", "wall 1x", "wall 2x",
                             "wetted R6", "wetted R7", "geometry", "Young BC"]),
                 xticklabelrotation=pi / 5,
                 limits=(nothing, nothing, 0, 4.4))

particles = getindex.(young_laplace, "particle_count")
errors = 100 .* getindex.(young_laplace, "relative_error")
scatterlines!(young_laplace_axis, particles, errors;
              color=:navy, marker=:circle, label="CSS operator fit")
hlines!(young_laplace_axis, [5.0]; color=:firebrick, linestyle=:dash,
        label="5% acceptance")
axislegend(young_laplace_axis; position=:rt)

colors = (:dodgerblue, :darkorange, :seagreen)
markers = (:circle, :diamond, :utriangle)
for (index, resolution) in enumerate((750, 1500, 3000))
    for (mechanism, axis) in (("geometric", geometric_axis),
         ("contact_line_force", clf_axis))
        rows = sessile[(sessile.requested_particles .== resolution) .& (sessile.mechanism .== mechanism),
                       :]
        sort!(rows, :target)
        scatterlines!(axis, rows.target, rows.circle_angle;
                      color=colors[index], marker=markers[index],
                      label="$resolution particles")
    end
end
for axis in (geometric_axis, clf_axis)
    lines!(axis, [20, 160], [20, 160]; color=:black, linestyle=:dash,
           label="target")
    axislegend(axis; position=:lt)
end

mechanism_colors = Dict("none" => :gray45, "geometric" => :dodgerblue,
                        "contact_line_force" => :darkorange,
                        "r4_wetted_area" => :firebrick)
mechanism_markers = Dict("none" => :rect, "geometric" => :circle,
                         "contact_line_force" => :diamond,
                         "r4_wetted_area" => :star5)
for mechanism in ("geometric", "contact_line_force")
    row = only(eachrow(scorecard[scorecard.mechanism .== mechanism, :]))
    scatterlines!(mae_axis, [750, 1500, 3000],
                  [row.mae_750, row.mae_1500, row.mae_3000];
                  color=mechanism_colors[mechanism], marker=mechanism_markers[mechanism],
                  label=replace(mechanism, "contact_line_force" => "CLF"))
end
hlines!(mae_axis, [5.0]; color=:firebrick, linestyle=:dash, label="5 deg gate")
axislegend(mae_axis; position=:rt)

for mechanism in ("none", "geometric", "contact_line_force")
    rows = perturbation[perturbation.mechanism .== mechanism, :]
    scatterlines!(response_axis, 1:4, 100 .* rows.error_reduction;
                  color=mechanism_colors[mechanism], marker=mechanism_markers[mechanism],
                  label=replace(mechanism, "contact_line_force" => "CLF"))
end
r4_response = r4_perturbation[r4_perturbation.mechanism .== "wetted_area_production", :]
scatterlines!(response_axis, 1:4, 100 .* r4_response.error_reduction;
              color=mechanism_colors["r4_wetted_area"],
              marker=mechanism_markers["r4_wetted_area"], label="production (0.02 s)")
hlines!(response_axis, [0.0]; color=:firebrick, linestyle=:dash)
axislegend(response_axis; position=:lb)

cost_mechanisms = ("none", "geometric", "contact_line_force")
cost_medians = [median(cost[cost.mechanism .== mechanism, :normalized_runtime])
                for mechanism in cost_mechanisms]
push!(cost_medians,
      median(r4_cost[r4_cost.mechanism .== "wetted_area_production",
                     :normalized_runtime]))
barplot!(cost_axis, 1:4, cost_medians;
         color=[mechanism_colors[mechanism]
                for mechanism in (cost_mechanisms..., "r4_wetted_area")])
hlines!(cost_axis, [1.0]; color=:black, linestyle=:dash)

for (variant, color, marker, label) in (("baseline_total", :darkorange, :circle,
      "wall-completed"),
     ("fluid_only", :purple, :diamond, "fluid-only"))
    rows = normal_components[(normal_components.variant .== variant) .& (normal_components.requested_particles .== 1500) .& (normal_components.target .== normal_components.initial_angle),
                             :]
    scatterlines!(normal_axis, rows.local_angle, rows.angle_mean;
                  color, marker, label)
end
lines!(normal_axis, [20, 160], [20, 160]; color=:black, linestyle=:dash,
       label="reference")
axislegend(normal_axis; position=:lt)

line_rows = normal_components[(normal_components.variant .== "baseline_total") .& (normal_components.requested_particles .== 1500) .& (normal_components.target .== normal_components.initial_angle),
                              :]
scatterlines!(line_axis, line_rows.target, 100 .* line_rows.line_length_error;
              color=:darkorange, marker=:circle, label="current divergence")
scatterlines!(line_axis, line_rows.target, 100 .* line_rows.raw_cross_error_4x;
              color=:purple, marker=:diamond, label="raw coarea, one-sided")
scatterlines!(line_axis, line_rows.target, 100 .* line_rows.corrected_cross_error_4x;
              color=:seagreen, marker=:utriangle, label="coarea / support moment")
compatible_rows = cap_transfer[(cap_transfer.variant .== "compatible_indicator") .& (cap_transfer.requested_particles .== 1500),
                               :]
scatterlines!(line_axis, compatible_rows.target, 100 .* compatible_rows.line_length_error;
              color=:steelblue, marker=:rect, label="compatible continuation")
wetted_rows = wetted_area[wetted_area.requested_particles .== 1500, :]
scatterlines!(line_axis, wetted_rows.target, 100 .* wetted_rows.area_error;
              color=:gray55, marker=:cross, linestyle=:dot, label="wetted area R6")
control_rows = measure_protocol[(measure_protocol.series .== "production_resolution") .& (measure_protocol.requested_particles .== 1500),
                                :]
scatterlines!(line_axis, control_rows.target, 100 .* control_rows.line_length_error;
              color=:black, marker=:star5, linestyle=:dash,
              label="phase-averaged exact control")
for (variant, color, marker, label) in
    (("compatible_geometry_wall", :purple, :hexagon, "compatible + geometry"),
     ("young_color_boundary", :darkorange, :pentagon, "Young color BC"))
    rows = extended_recovery[(extended_recovery.variant .== variant) .& (extended_recovery.requested_particles .== 1500),
                             :]
    scatterlines!(line_axis, rows.target, 100 .* rows.line_length_error;
                  color, marker, label)
end
corrected_rows = corrected_wetted_area[corrected_wetted_area.requested_particles .== 1500,
                                       :]
scatterlines!(line_axis, corrected_rows.target,
              100 .* corrected_rows.corrected_area_error;
              color=:firebrick, marker=:xcross, label="corrected wetted area")
hlines!(line_axis, [20.0]; color=:firebrick, linestyle=:dash, label="20% gate")
axislegend(line_axis; position=:lt)

sign_counts = [count(force_sign[(force_sign.mechanism .== "geometric"), :sign_pass]),
    count(force_sign[(force_sign.mechanism .== "contact_line_force"), :sign_pass]),
    count(ghost_sign.sign_pass),
    count(wall_energy_sign[wall_energy_sign.variant .== "wall_energy_1x", :sign_pass]),
    count(wall_energy_sign[wall_energy_sign.variant .== "wall_energy_2x", :sign_pass]),
    count(wetted_area_sign.sign_pass),
    count(extended_sign[(extended_sign.variant .== "corrected_wetted_area"), :sign_pass]),
    count(extended_sign[(extended_sign.variant .== "compatible_geometry_wall"), :sign_pass]),
    count(extended_sign[(extended_sign.variant .== "young_color_boundary"), :sign_pass])]
barplot!(sign_axis, 1:9, sign_counts;
         color=[:dodgerblue, :darkorange, :seagreen, :gray55, :purple, :steelblue,
             :firebrick, :mediumpurple, :goldenrod])
hlines!(sign_axis, [4.0]; color=:black, linestyle=:dash)

save(joinpath(@__DIR__, "surface_tension_3d_validation.png"), figure)
figure
