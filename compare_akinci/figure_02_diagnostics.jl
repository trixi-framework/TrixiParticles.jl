using CairoMakie
using Serialization

include(joinpath(@__DIR__, "figure_02_metrics.jl"))

function nearest_frame(snapshot, target_time)
    _, index = findmin(abs.(snapshot.times .- target_time))
    return snapshot.frames[index]
end

function center_slice(frame; system_index=1)
    system = frame.systems[system_index]
    coordinates = system.coordinates
    center_y = (minimum(coordinates[2, :]) + maximum(coordinates[2, :])) / 2
    mask = abs.(coordinates[2, :] .- center_y) .<= 0.75 * system.particle_spacing
    return coordinates[:, mask], system.particle_spacing
end

function top_slice(frame; system_index=1)
    system = frame.systems[system_index]
    coordinates = system.coordinates
    center_z = (minimum(coordinates[3, :]) + maximum(coordinates[3, :])) / 2
    mask = abs.(coordinates[3, :] .- center_z) .<= 0.75 * system.particle_spacing
    return coordinates[:, mask], system.particle_spacing
end

function figure_02_diagnostic(akinci_path, css_path, output_path)
    akinci = open(deserialize, akinci_path)
    css = open(deserialize, css_path)
    figure = Figure(; size=(1760, 720), fontsize=17)
    models = (("Akinci", akinci, :darkorange2), ("CSS", css, :dodgerblue3))
    final_time = min(last(akinci.times), last(css.times))
    panel_times = (0.05, final_time)

    for (column, time) in enumerate(panel_times)
        slices = map(models) do (_, snapshot, _)
            center_slice(nearest_frame(snapshot, time))
        end
        x_min = minimum(minimum(coordinates[1, :]) for (coordinates, _) in slices)
        x_max = maximum(maximum(coordinates[1, :]) for (coordinates, _) in slices)
        z_min = minimum(minimum(coordinates[3, :]) for (coordinates, _) in slices)
        z_max = maximum(maximum(coordinates[3, :]) for (coordinates, _) in slices)
        margin = 2maximum(last, slices)

        for (row, ((label, _, color), (coordinates, _))) in enumerate(zip(models, slices))
            axis = Axis(figure[row, column];
                        title="$label, t = $(time) s", xlabel="x [m]", ylabel="z [m]",
                        aspect=DataAspect())
            scatter!(axis, coordinates[1, :], coordinates[3, :]; color, markersize=3)
            limits!(axis, x_min - margin, x_max + margin,
                    min(-margin, z_min - margin), z_max + margin)
        end
    end

    final_slices = map(models) do (_, snapshot, _)
        top_slice(nearest_frame(snapshot, final_time))
    end
    final_coordinates = first.(final_slices)
    final_spacing = last.(final_slices)
    x_min = minimum(minimum(coordinates[1, :]) for coordinates in final_coordinates)
    x_max = maximum(maximum(coordinates[1, :]) for coordinates in final_coordinates)
    y_min = minimum(minimum(coordinates[2, :]) for coordinates in final_coordinates)
    y_max = maximum(maximum(coordinates[2, :]) for coordinates in final_coordinates)
    margin = 2 * maximum(final_spacing)
    for (row, ((label, snapshot, color), coordinates)) in
        enumerate(zip(models, final_coordinates))
        metrics = figure_02_metrics(snapshot)
        final = metric_at_time(metrics, final_time)
        axis = Axis(figure[row, 3];
                    title="$label top slice, asymmetry = $(round(100 * final.planar_asymmetry; digits=2))%",
                    xlabel="x [m]", ylabel="y [m]", aspect=DataAspect())
        scatter!(axis, coordinates[1, :], coordinates[2, :]; color,
                 markersize=3)
        limits!(axis, x_min - margin, x_max + margin,
                y_min - margin, y_max + margin)
    end

    sphere_axis = Axis(figure[1, 4]; title="Sphere formation",
                       xlabel="time [s]", ylabel="radial asphericity", yscale=log10)
    spread_axis = Axis(figure[2, 4]; title="Impact spreading (not settled)",
                       xlabel="time [s]", ylabel="height / width")
    for (label, snapshot, color) in models
        metrics = figure_02_metrics(snapshot)
        sphere_metrics = filter(row -> row.time <= 0.05, metrics)
        lines!(sphere_axis, getproperty.(sphere_metrics, :time),
               getproperty.(sphere_metrics, :asphericity); label, color, linewidth=3)
        scatter!(sphere_axis, getproperty.(sphere_metrics, :time),
                 getproperty.(sphere_metrics, :asphericity); color, markersize=8)
        lines!(spread_axis, getproperty.(metrics, :time),
               getproperty.(metrics, :height_to_width); label, color, linewidth=3)
        scatter!(spread_axis, getproperty.(metrics, :time),
                 getproperty.(metrics, :height_to_width); color, markersize=8)
    end
    vlines!(sphere_axis, [0.05]; color=:gray45, linestyle=:dash,
            label="gravity release")
    vlines!(spread_axis, [0.05]; color=:gray45, linestyle=:dash)
    axislegend(sphere_axis; position=:lb)
    save(output_path, figure; px_per_unit=1.5)
    println("Wrote Figure 2 CSS/Akinci diagnostic to $output_path")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 ||
        error("pass Akinci snapshot, CSS snapshot, and output PNG")
    figure_02_diagnostic(ARGS...)
end
