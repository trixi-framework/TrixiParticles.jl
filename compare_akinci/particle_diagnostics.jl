using CairoMakie
using Serialization

function fluid_slice(frame, system_index, slice_center, slice_half_width)
    system = frame.systems[system_index]
    coordinates = system.coordinates
    mask = abs.(coordinates[2, :] .- slice_center) .<= slice_half_width
    pressure = isnothing(system.pressure) ? nothing : system.pressure[mask]
    return (; coordinates=coordinates[:, mask], pressure)
end

function particle_diagnostics(snapshot_path, output_path; system_index=1,
                              color_by_pressure=false)
    snapshot = open(deserialize, snapshot_path)
    isempty(snapshot.frames) && error("snapshot contains no frames")

    system = snapshot.frames[1].systems[system_index]
    spacing = system.particle_spacing
    y_min = minimum(system.coordinates[2, :])
    y_max = maximum(system.coordinates[2, :])
    slice_center = (y_min + y_max) / 2
    slice_half_width = 0.75 * spacing
    slices = [fluid_slice(frame, system_index, slice_center, slice_half_width)
              for frame in snapshot.frames]

    x_min = minimum(minimum(slice.coordinates[1, :]) for slice in slices)
    x_max = maximum(maximum(slice.coordinates[1, :]) for slice in slices)
    z_min = minimum(minimum(slice.coordinates[3, :]) for slice in slices)
    z_max = maximum(maximum(slice.coordinates[3, :]) for slice in slices)
    margin = 2 * spacing

    pressure_max = if color_by_pressure
        available_pressures = [slice.pressure
                               for slice in slices
                               if !isnothing(slice.pressure)]
        pressures = isempty(available_pressures) ? Float64[] :
                    reduce(vcat, available_pressures)
        isempty(pressures) ? 1.0 : max(maximum(pressures), eps(eltype(pressures)))
    else
        1.0
    end

    columns = min(4, length(snapshot.frames))
    rows = cld(length(snapshot.frames), columns)
    figure = Figure(; size=(360 * columns, 310 * rows), fontsize=18)

    for (index, (time, frame)) in enumerate(zip(snapshot.times, snapshot.frames))
        row = div(index - 1, columns) + 1
        column = mod1(index, columns)
        axis = Axis(figure[row, column]; title="t = $(round(time; digits=4)) s",
                    xlabel="x [m]", ylabel="z [m]", aspect=DataAspect())
        slice = slices[index]
        color = color_by_pressure && !isnothing(slice.pressure) ? slice.pressure :
                :dodgerblue3
        scatter!(axis, slice.coordinates[1, :], slice.coordinates[3, :]; color,
                 colormap=:Reds, colorrange=(0, pressure_max), markersize=3)
        limits!(axis, x_min - margin, x_max + margin, z_min - margin, z_max + margin)
    end

    save(output_path, figure)
    println("Wrote raw particle diagnostics to $output_path")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 3 ||
        error("pass a snapshot path, output path, and optionally 'pressure'")
    color_by_pressure = length(ARGS) == 3 && ARGS[3] == "pressure"
    particle_diagnostics(ARGS[1], ARGS[2]; color_by_pressure)
end
