using CairoMakie
using LinearAlgebra
using Serialization
using Statistics

vector_magnitudes(values) = vec(sqrt.(sum(abs2, values; dims=1)))

const COMPONENTS = ((; label="scaled normal", field=:scaled_normal, units=""),
                    (; label="pressure", field=:pressure, units="m/s^2"),
                    (; label="viscosity", field=:viscosity, units="m/s^2"),
                    (; label="cohesion", field=:cohesion, units="m/s^2"),
                    (; label="normal difference", field=:curvature, units="m/s^2"),
                    (; label="adhesion", field=:adhesion, units="m/s^2"),
                    (; label="total", field=:total, units="m/s^2"))

function component_values(frame, field)
    field == :pressure && return frame.pressure_fluid + frame.pressure_boundary
    field == :viscosity && return frame.viscosity_fluid + frame.viscosity_boundary
    return getproperty(frame, field)
end

function selected_frames(analysis, requested_times)
    isnothing(requested_times) && return analysis.frames
    return map(requested_times) do requested_time
        index = argmin(abs.([frame.time for frame in analysis.frames] .- requested_time))
        frame = analysis.frames[index]
        isapprox(frame.time, requested_time; atol=1.0e-10) ||
            error("analysis has no frame at t=$requested_time")
        frame
    end
end

function center_slice(frame)
    slice_center = median(frame.coordinates[2, :])
    return abs.(frame.coordinates[2, :] .- slice_center) .<=
           0.75 * frame.particle_spacing
end

function reference_magnitude(frames, field)
    magnitudes = reduce(vcat,
                        vector_magnitudes(component_values(frame, field))
                        for frame in frames)
    reference = quantile(magnitudes, 0.95)
    return max(reference, eps(eltype(magnitudes)))
end

function vector_segments(coordinates, values, mask, spacing, reference)
    points = Point2f[]
    colors = Float64[]
    for particle in findall(mask)
        vector = values[[1, 3], particle]
        magnitude = norm(vector)
        magnitude <= eps(eltype(values)) && continue
        direction = vector / magnitude
        length_ = 1.5 * spacing * min(magnitude / reference, 1)
        origin = Point2f(coordinates[1, particle], coordinates[3, particle])
        endpoint = origin + Point2f(direction[1] * length_, direction[2] * length_)
        push!(points, origin, endpoint)
        push!(colors, magnitude, magnitude)
    end
    return points, colors
end

function force_diagnostics(analysis_path, output_path; requested_times=nothing)
    analysis = open(deserialize, analysis_path)
    frames = selected_frames(analysis, requested_times)
    references = Dict(component.field => reference_magnitude(frames, component.field)
                      for component in COMPONENTS)

    x_min = minimum(minimum(frame.coordinates[1, :]) for frame in frames)
    x_max = maximum(maximum(frame.coordinates[1, :]) for frame in frames)
    z_min = minimum(minimum(frame.coordinates[3, :]) for frame in frames)
    z_max = maximum(maximum(frame.coordinates[3, :]) for frame in frames)
    margin = 2 * maximum(frame.particle_spacing for frame in frames)

    figure = Figure(; size=(310 * length(COMPONENTS), 310 * length(frames)),
                    fontsize=15)
    for (row, frame) in enumerate(frames)
        mask = center_slice(frame)
        for (column, component) in enumerate(COMPONENTS)
            values = component_values(frame, component.field)
            magnitudes = vector_magnitudes(values)
            reference = references[component.field]
            unit_suffix = isempty(component.units) ? "" : " $(component.units)"
            title = "$(component.label) | t=$(round(frame.time; digits=3)) s\n" *
                    "median=$(round(median(magnitudes); sigdigits=3)), " *
                    "color cap=$(round(reference; sigdigits=3))$unit_suffix"
            axis = Axis(figure[row, column]; title, xlabel="x [m]",
                        ylabel=column == 1 ? "z [m]" : "", aspect=DataAspect())
            points,
            colors = vector_segments(frame.coordinates, values, mask,
                                     frame.particle_spacing, reference)
            isempty(points) || linesegments!(axis, points; color=colors,
                          colorrange=(0, reference), colormap=:viridis,
                          linewidth=1.5)
            scatter!(axis, frame.coordinates[1, mask], frame.coordinates[3, mask];
                     color=magnitudes[mask], colorrange=(0, reference), colormap=:viridis,
                     markersize=5, strokecolor=:black, strokewidth=0.35)
            hlines!(axis, 0; color=:gray35, linewidth=1.5)
            limits!(axis, x_min - margin, x_max + margin, z_min - margin, z_max + margin)
        end
    end

    Label(figure[0, :],
          "$(analysis.case_name): per-particle vectors; " *
          "line length is clipped at each component's p95",
          fontsize=20, font=:bold)
    save(output_path, figure)
    println("Wrote per-particle force diagnostics to $output_path")
    return output_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    2 <= length(ARGS) <= 3 ||
        error("pass an analysis snapshot, output path, and optional comma-separated times")
    requested_times = length(ARGS) == 3 ? parse.(Float64, split(ARGS[3], ',')) : nothing
    force_diagnostics(ARGS[1], ARGS[2]; requested_times)
end
