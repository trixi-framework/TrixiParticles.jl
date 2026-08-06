using CairoMakie
using Serialization
using Statistics

function center_slice(snapshot_path; system_index=1)
    snapshot = open(deserialize, snapshot_path)
    frame = last(snapshot.frames)
    system = frame.systems[system_index]
    coordinates = system.coordinates
    spacing = system.particle_spacing
    slice_center = median(coordinates[2, :])
    mask = abs.(coordinates[2, :] .- slice_center) .<= 0.75 * spacing
    width = max(maximum(coordinates[1, :]) - minimum(coordinates[1, :]),
                maximum(coordinates[2, :]) - minimum(coordinates[2, :])) + spacing
    height = maximum(coordinates[3, :]) - minimum(coordinates[3, :]) + spacing
    below = coordinates[3, :] .< 0
    below_wall = count(below)
    return (; coordinates=coordinates[:, mask], spacing, time=last(snapshot.times), width,
            height, below_wall, below_coordinates=coordinates[:, below],
            particle_count=size(coordinates, 2))
end

function wetting_diagnostics(entries, output_path)
    slices = map(entry -> center_slice(entry.snapshot), entries)
    x_min = minimum(minimum(slice.coordinates[1, :]) for slice in slices)
    x_max = maximum(maximum(slice.coordinates[1, :]) for slice in slices)
    z_min = minimum(minimum(slice.coordinates[3, :]) for slice in slices)
    z_max = maximum(maximum(slice.coordinates[3, :]) for slice in slices)
    margin = 2 * maximum(slice.spacing for slice in slices)

    figure = Figure(; size=(360 * length(entries), 340), fontsize=17)
    for (index, (entry, slice)) in enumerate(zip(entries, slices))
        (; coordinates, time, width, height, below_wall, below_coordinates,
         particle_count) = slice
        status = if iszero(below_wall)
            "h/w=$(round(height / width; digits=2))"
        else
            "invalid: $below_wall/$particle_count below plane"
        end
        axis = Axis(figure[1, index]; title=entry.label,
                    subtitle="t=$(round(time; digits=3)) s, $status",
                    xlabel="x [m]", ylabel=index == 1 ? "z [m]" : "",
                    yticks=WilkinsonTicks(5), aspect=DataAspect())
        scatter!(axis, coordinates[1, :], coordinates[3, :]; color=:dodgerblue3,
                 markersize=3)
        if !isempty(below_coordinates)
            scatter!(axis, below_coordinates[1, :], below_coordinates[3, :]; color=:crimson,
                     markersize=5)
        end
        hlines!(axis, 0; color=:gray35, linewidth=2)
        limits!(axis, x_min - margin, x_max + margin, z_min - margin, z_max + margin)

        println(entry.label, ": t=", round(time; digits=3), " s, width=",
                round(1.0e3 * width; digits=3), " mm, height=",
                round(1.0e3 * height; digits=3), " mm, h/w=",
                round(height / width; digits=4), ", below plane=", below_wall, "/",
                particle_count)
    end

    save(output_path, figure)
    println("Wrote wetting diagnostics to $output_path")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 3 ||
        error("pass an output path followed by at least two label=snapshot entries")
    entries = map(ARGS[2:end]) do argument
        label, snapshot = split(argument, '='; limit=2)
        return (; label, snapshot)
    end
    wetting_diagnostics(entries, ARGS[1])
end
