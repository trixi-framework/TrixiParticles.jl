using GPUSelect

const DEVICE = GPUSelect.Backend(:Lava; fallback=false)

using FileIO
using GeometryBasics
using Hikari
using Makie
using Meshing
using RayMakie
using Serialization

const IMAGE_SIZE = (640, 480)
const FLUID_ISO_FRACTION = 0.35f0
const SOLID_ISO_FRACTION = 0.45f0
const WATER_MATERIAL = Hikari.Dielectric(;
                                         Kr=Hikari.RGBSpectrum(1.0f0, 1.0f0, 1.0f0),
                                         Kt=Hikari.RGBSpectrum(0.7f0, 0.85f0, 0.95f0),
                                         roughness=0.01f0,
                                         index=1.33f0)
const GROUND_MATERIAL = Hikari.Plastic(; color=(0.38f0, 0.40f0, 0.43f0),
                                       roughness=0.3f0)

include("cases.jl")

function load_snapshot(config)
    project = joinpath(@__DIR__, "simulation")
    simulator = joinpath(@__DIR__, "simulate.jl")

    return mktemp() do path, io
        close(io)
        run(`$(Base.julia_cmd()) --project=$project $simulator $(config.name) $path`)
        open(deserialize, path)
    end
end

function system_coordinates(frame, system_index)
    return frame.systems[system_index].coordinates
end

function solution_bounds(frame, system_indices)
    minimum_corner = fill(Inf32, 3)
    maximum_corner = fill(-Inf32, 3)

    for system_index in system_indices
        coordinates = system_coordinates(frame, system_index)
        for dimension in axes(coordinates, 1)
            minimum_corner[dimension] = min(minimum_corner[dimension],
                                            minimum(coordinates[dimension, :]))
            maximum_corner[dimension] = max(maximum_corner[dimension],
                                            maximum(coordinates[dimension, :]))
        end
    end

    center = Makie.Point3f((minimum_corner + maximum_corner) / 2)
    extent = maximum(maximum_corner - minimum_corner)
    return center, max(extent, 0.05f0)
end

function system_color(config, system_index)
    if hasproperty(config, :system_colors)
        color_index = findfirst(pair -> first(pair) == system_index, config.system_colors)
        !isnothing(color_index) &&
            return Makie.RGBf(last(config.system_colors[color_index])...)
    end

    system_index == 1 && return Makie.RGBf(0.015, 0.24, 0.82)
    return Makie.RGBf(0.48, 0.52, 0.58)
end

function particle_surface(coordinates, particle_spacing; fluid=false,
                          iso_fraction=fluid ? FLUID_ISO_FRACTION : SOLID_ISO_FRACTION)
    spacing = Float32(particle_spacing)
    sigma = (fluid ? 1.1f0 : 0.7f0) * spacing
    grid_step = 0.5f0 * spacing
    cutoff = 3.0f0 * sigma

    minimum_corner = Float32.(vec(minimum(coordinates; dims=2))) .- cutoff
    maximum_corner = Float32.(vec(maximum(coordinates; dims=2))) .+ cutoff
    grid_size = ntuple(dimension -> max(4,
                                        ceil(Int,
                                             (maximum_corner[dimension] -
                                              minimum_corner[dimension]) / grid_step) + 1),
                       3)
    ranges = ntuple(dimension -> range(minimum_corner[dimension], maximum_corner[dimension];
                                       length=grid_size[dimension]),
                    3)
    field = zeros(Float32, grid_size)
    cutoff_squared = cutoff^2
    inverse_two_sigma_squared = inv(2.0f0 * sigma^2)

    for particle in axes(coordinates, 2)
        position = Float32.(view(coordinates, :, particle))
        index_ranges = ntuple(3) do dimension
            lower = max(1,
                        searchsortedfirst(ranges[dimension],
                                          position[dimension] - cutoff))
            upper = min(grid_size[dimension],
                        searchsortedlast(ranges[dimension],
                                         position[dimension] + cutoff))
            lower:upper
        end

        for k in index_ranges[3], j in index_ranges[2], i in index_ranges[1]
            distance_squared = (ranges[1][i] - position[1])^2 +
                               (ranges[2][j] - position[2])^2 +
                               (ranges[3][k] - position[3])^2
            distance_squared > cutoff_squared && continue
            field[i, j, k] += exp(-distance_squared * inverse_two_sigma_squared)
        end
    end

    field[[1, end], :, :] .= 0
    field[:, [1, end], :] .= 0
    field[:, :, [1, end]] .= 0
    interior_value = (sqrt(2.0f0 * Float32(pi)) * sigma / spacing)^3
    target_iso = Float32(iso_fraction) * interior_value
    iso = min(target_iso, 0.5f0 * maximum(field))

    points, faces = Meshing.isosurface(field, Meshing.MarchingTetrahedra(; iso),
                                       ranges...)
    mesh = GeometryBasics.Mesh(GeometryBasics.Point3f.(points),
                               GeometryBasics.GLTriangleFace.(faces))
    return GeometryBasics.normal_mesh(mesh)
end

function system_material(color)
    return Hikari.Plastic(; color=Tuple(color), roughness=0.24f0)
end

function particle_colors(system, config, panel)
    color_field = panel_setting(panel, config, :particle_color, :constant)
    color_field == :constant && return Makie.RGBf(0.88, 0.52, 0.48)
    color_field == :pressure || error("unsupported particle color field: $color_field")
    isnothing(system.pressure) && error("snapshot does not contain fluid pressure")

    pressure_max = max(maximum(system.pressure), eps(eltype(system.pressure)))
    return map(system.pressure) do pressure
        fraction = Float32(clamp(pressure / pressure_max, 0, 1))
        Makie.RGBf(0.92f0, 0.78f0 * (1 - fraction), 0.76f0 * (1 - fraction))
    end
end

function panel_setting(panel, config, name)
    return hasproperty(panel, name) ? getproperty(panel, name) : getproperty(config, name)
end

function panel_setting(panel, config, name, default)
    hasproperty(panel, name) && return getproperty(panel, name)
    hasproperty(config, name) && return getproperty(config, name)
    return default
end

function snapshot_frame(snapshot, time)
    frame_index = argmin(abs.(snapshot.times .- time))
    isapprox(snapshot.times[frame_index], time; atol=1.0e-10, rtol=1.0e-10) ||
        error("snapshot does not contain requested time $time")
    return snapshot.frames[frame_index]
end

function render_props!(axis, panel, config)
    props = panel_setting(panel, config, :props, ())
    for prop in props
        prop.kind == :box || error("unsupported scene prop kind: $(prop.kind)")
        geometry = GeometryBasics.Rect3f(Makie.Vec3f(prop.origin), Makie.Vec3f(prop.widths))
        color = Makie.RGBf(prop.color...)
        material = Hikari.Plastic(; color=Tuple(color), roughness=0.24f0)
        Makie.mesh!(axis, geometry; color, material)
    end
    return axis
end

function render_system!(axis, config, panel, system, system_index)
    is_fluid = system.kind == :fluid
    color = is_fluid ? Makie.RGBf(0.6, 0.8, 0.95) :
            system_color(config, system_index)
    material = is_fluid ? WATER_MATERIAL : system_material(color)
    style = panel_setting(panel, config, :render_style, :surface)

    if style == :particles && is_fluid
        coordinates = system.coordinates
        points = [Makie.Point3f(coordinates[:, particle])
                  for particle in axes(coordinates, 2)]
        marker = GeometryBasics.Sphere(Makie.Point3f(0), 0.5f0)
        scale = panel_setting(panel, config, :particle_scale, 0.82f0)
        particle_color = particle_colors(system, config, panel)
        Makie.meshscatter!(axis, points; marker,
                           markersize=Float32(scale * system.particle_spacing),
                           color=particle_color,
                           material=system_material(Makie.RGBf(1.0, 1.0, 1.0)))
        return axis
    elseif style != :surface
        error("unsupported render style: $style")
    end

    iso_fraction = if is_fluid
        panel_setting(panel, config, :fluid_iso_fraction, FLUID_ISO_FRACTION)
    else
        panel_setting(panel, config, :solid_iso_fraction, SOLID_ISO_FRACTION)
    end
    surface = particle_surface(system.coordinates, system.particle_spacing;
                               fluid=is_fluid, iso_fraction)
    Makie.mesh!(axis, surface; color, material)
    return axis
end

function render_panel(config, snapshot, panel, integrator)
    frame = snapshot_frame(snapshot, panel.time)
    focus_indices = panel_setting(panel, config, :focus_indices)
    center, extent = solution_bounds(frame, focus_indices)
    if hasproperty(panel, :view_center)
        center = Makie.Point3f(panel.view_center)
    elseif hasproperty(config, :view_center)
        center = Makie.Point3f(config.view_center)
    end
    if hasproperty(panel, :view_extent)
        extent = Float32(panel.view_extent)
    elseif hasproperty(config, :view_extent)
        extent = Float32(config.view_extent)
    end

    direction = Makie.Vec3f(panel_setting(panel, config, :camera))
    direction /= sqrt(sum(abs2, direction))
    eye = center + panel_setting(panel, config, :camera_scale) * extent * direction

    lights = [Makie.SunSkyLight(Makie.Vec3f(0.4, -0.3, 0.8);
                                intensity=1.0f0, turbidity=3.0f0,
                                ground_enabled=false)]
    figure = Makie.Figure(; size=IMAGE_SIZE)
    axis = Makie.LScene(figure[1, 1]; show_axis=false,
                        scenekw=(; lights,
                                 backgroundcolor=Makie.RGBf(0.035, 0.045, 0.065)))

    for system_index in panel_setting(panel, config, :system_indices)
        system = frame.systems[system_index]
        isempty(system.coordinates) && continue
        render_system!(axis, config, panel, system, system_index)
    end

    render_props!(axis, panel, config)

    show_ground = panel_setting(panel, config, :show_ground, true)
    if show_ground
        ground_size = 3.0f0 * extent
        ground_height = center[3] - 0.75f0 * extent
        ground = GeometryBasics.Rect3f(Makie.Vec3f(center[1] - ground_size / 2,
                                                   center[2] - ground_size / 2,
                                                   ground_height),
                                       Makie.Vec3f(ground_size, ground_size,
                                                   max(0.01f0 * extent, 1.0f-4)))
        Makie.mesh!(axis, ground; color=Makie.RGBf(0.38, 0.40, 0.43),
                    material=GROUND_MATERIAL)
    end

    camera = Makie.cameracontrols(axis.scene)
    camera.eyeposition[] = eye
    camera.lookat[] = center
    camera.upvector[] = Makie.Vec3f(0, 0, 1)
    camera.fov[] = panel_setting(panel, config, :fov)
    camera.near[] = max(extent / 100, 1.0f-4)
    camera.far[] = 100.0f0 * extent
    Makie.update_cam!(axis.scene, camera)

    output = isabspath(panel.output) ? panel.output : joinpath(@__DIR__, panel.output)
    mkpath(dirname(output))
    image = Makie.colorbuffer(figure; backend=RayMakie, device=DEVICE, integrator,
                              tonemap=:aces, gamma=2.2f0, update=false)
    FileIO.save(output, image)
    println("Wrote $(config.title), t=$(panel.time), to $output")
    return output
end

function render(config; snapshot_path=nothing)
    snapshot = isnothing(snapshot_path) ? load_snapshot(config) :
               open(deserialize, snapshot_path)
    samples = parse(Int, get(ENV, "TRIXIPARTICLES_RAY_SAMPLES", "128"))
    max_depth = parse(Int, get(ENV, "TRIXIPARTICLES_RAY_MAX_DEPTH", "12"))
    integrator = Hikari.VolPath(; samples, max_depth, hw_accel=true,
                                max_component_value=10.0f0, regularize=true)
    RayMakie.activate!(; device=DEVICE, integrator, tonemap=:aces, gamma=2.2f0)

    panels = hasproperty(config, :panels) ? config.panels :
             ((; time=last(snapshot.times), output=config.output),)
    return map(panel -> render_panel(config, snapshot, panel, integrator), panels)
end

if abspath(PROGRAM_FILE) == @__FILE__
    1 <= length(ARGS) <= 2 ||
        error("pass one comparison case and optionally a snapshot path: " *
              join(getproperty.(CASES, :name), ", "))
    snapshot_path = length(ARGS) == 2 ? ARGS[2] : nothing
    render(case_config(ARGS[1]); snapshot_path)
end
