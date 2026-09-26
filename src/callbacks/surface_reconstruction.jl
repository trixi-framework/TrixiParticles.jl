@doc raw"""
    SurfaceReconstructionCallback(reconstruction, semi; fluid_systems=nothing,
                                  boundaries=nothing, interval::Integer=0, dt=0.0,
                                  save_times=Float64[], output_directory="out",
                                  append_timestamp=false, prefix="surface",
                                  formats=(:vtp,), verbose=false,
                                  save_initial_surface=true, save_final_surface=true,
                                  overwrite=false, compress=1, write_statistics=true,
                                  write_boundaries=false, interpolated_quantities=())

Callback to reconstruct closed planar contours (2D) or triangle surfaces (3D) from the
live fluid state using the shared [`SurfaceReconstruction`](@ref) engine. Use at most one of
`interval`, `dt`, and `save_times`: pass `interval` to reconstruct every `interval`
accepted time steps, `dt` to reconstruct in intervals of `dt` in terms of integration
time by adding additional `tstops` (note that this may change the solution), or
`save_times` to reconstruct at specific times.

Each fluid system is reconstructed separately. The callback writes to `output_directory`:
- `<prefix>_<system>_<iter>.vtp`: the surface as VTK PolyData with per-vertex `Normals`,
  collected per system in a `<prefix>_<system>.pvd` time series (e.g.
  `surface_fluid_1.pvd`), optionally with interpolated SPH quantities as point data.
  `formats` additionally accepts `:ply`. Planar output uses VTK lines or PLY edges at z=0.
- `<prefix>_statistics.csv` / `.json`: per-event time series of every fluid system's
  reconstruction statistics (volume, particle volume, relative volume error, effective
  isovalue, region and topology-defect counts, correction evaluations, run time), in the
  file format of the [`PostprocessCallback`](@ref).
- `meta_<prefix>.json`: simulation and system metadata like the
  [`SolutionSavingCallback`](@ref), plus the reconstruction configuration.
- With `write_boundaries=true`, the tracked boundary surfaces of boundary and structure
  systems in the same formats (e.g. `surface_structure_1_<iter>.vtp`).

Only active particles contribute (inactive buffer particles are excluded). The relevant
GPU state and systems are transferred to the CPU before reconstruction; the
neighborhood-search handler is transferred only when interpolation needs it. The latest
meshes and statistics are stored in `latest_meshes` and `latest_statistics` for
programmatic access (``cb.affect!.latest_meshes`` for the returned `DiscreteCallback`);
`latest_mesh` and `latest_stats` hold the first fluid system's results.

# Arguments
- `reconstruction`: A [`SurfaceReconstruction`](@ref). Pass `tank_size` there to pin the
                    grid and reuse the internal workspace across invocations. Unless it
                    configures its own `parallelization_backend`, the reconstruction
                    uses the simulation's CPU threading backend (GPU simulations use
                    `PolyesterBackend()` on the CPU copies). Its `particle_spacing`
                    should match the fluid systems'; a mismatch warns at construction.
                    Its dimension must match the selected systems. Set `ndims=2` for an
                    adaptive planar contour, or supply a two-component `tank_size`.
- `semi`:           The [`Semidiscretization`](@ref TrixiParticles.Semidiscretization).
                    System indices are resolved immediately.

# Keywords
- `fluid_systems=nothing`:   Systems to reconstruct. By default, all
                             `AbstractFluidSystem`s. Can be a system index, a system
                             object, or a collection of either.
- `boundaries=nothing`:       Walls and structures that bound the fluid and clip the
                             reconstructed surface. Either boundary systems (e.g.
                             `WallBoundarySystem`) or structure systems (e.g.
                             `TotalLagrangianSPHSystem`, `RigidBodySystem`) whose
                             particles form complete lattices — the surface topology is
                             built once from the initial configuration and tracked
                             through the current coordinates — or static geometries
                             ([`TrixiParticles.TriangleMesh`](@ref) in 3D or `Polygon{2}`
                             in 2D, e.g. from
                             [`load_geometry`](@ref), or a [`BoundaryMesh`](@ref)). Can
                             be a single entry or a collection.
- `interval=0`:                Reconstruct every `interval` accepted time steps;
                              `0` disables interval-triggered frames (initial and final
                              output remain enabled by default).
- `dt=0.0`:                    Reconstruct in regular intervals of `dt` in terms of
                               integration time.
- `save_times=Float64[]`:      Specific times at which to reconstruct.
- `output_directory="out"`:    Directory to write the mesh files to.
- `append_timestamp=false`:    Append current timestamp to the output directory.
- `prefix="surface"`:          Prefix of the written files
                               (`surface_fluid_1_000010.vtp`).
- `formats=(:vtp,)`:           Output formats, a combination of `:vtp` and `:ply`.
- `verbose=false`:             Print to standard IO when files are written.
- `save_initial_surface=true`: Reconstruct at initialization.
- `save_final_surface=true`:   Reconstruct at the end of the simulation.
- `overwrite=false`:           If `true`, reuse one `_current` file per system instead
                               of numbered files.
- `compress=1`:                Compress VTK data with zlib (levels `0`–`9` or booleans;
                             `true` selects level 6, which is much slower for
                             barely smaller files).
- `write_statistics=true`:     Write the statistics time series (CSV and JSON).
- `write_boundaries=false`:    Write the tracked surfaces of boundary and structure
                               systems (not of static geometries) with each event.
- `interpolated_quantities=()`: SPH quantities interpolated onto the surface vertices
                               and written as VTK point data, a combination of
                               `:velocity`, `:pressure`, and `:density`. Vertices are
                               interpolated from fluid particles only; vertices without
                               fluid neighbors get `NaN`. Requires `:vtp` in `formats`.

# Examples
```julia
tank = RectangularTank(0.05, (0.25, 0.25, 0.25), (0.5, 0.6, 0.5), 1000.0)
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid;
                                           smoothing_kernel=WendlandC2Kernel{3}(),
                                           smoothing_length=1.5 * 0.05,
                                           density_calculator=SummationDensity(),
                                            state_equation=StateEquationCole(;
                                                                            sound_speed=10.0,
                                                                            reference_density=1000.0,
                                                                            exponent=7))
semi = Semidiscretization(fluid_system)

# Reconstruct every 100 time steps
reconstruction = SurfaceReconstruction(particle_spacing=0.05; tank_size=(0.5, 0.6, 0.5))
surface_callback = SurfaceReconstructionCallback(reconstruction, semi, interval=100)
```
"""
mutable struct SurfaceReconstructionCallback{I, F, R}
    interval::I
    save_times::Vector{Float64}
    save_initial_surface::Bool
    save_final_surface::Bool
    overwrite::Bool
    compress::Union{Bool, Integer}
    formats::F
    fluid_indices::Vector{Int}
    boundary_indices::Vector{Int}
    reconstruction::R
    reconstructions::Vector{R}
    output_directory::String
    prefix::String
    verbose::Bool
    boundary_topologies::Dict{Int, BoundaryTopology}
    static_boundaries::Vector{BoundaryMesh}
    write_statistics::Bool
    write_boundaries::Bool
    interpolated_quantities::Vector{Symbol}
    git_hash::Ref{String}
    statistics_times::Vector{Float64}
    statistics_data::Dict{String, Vector{Float64}}
    latest_meshes::Vector{Union{Nothing, SurfaceMesh}}
    latest_statistics::Vector{Union{Nothing, SurfaceReconstructionStatistics}}
    latest_mesh::Union{Nothing, SurfaceMesh}
    latest_stats::Union{Nothing, SurfaceReconstructionStatistics}
    collection_initialized::Bool
    latest_saved_iter::Int
    # CPU neighborhood-search handler reused across interpolation events. The first
    # event with `interpolated_quantities` transfers it from the GPU; later events
    # refresh it in place with `update_nhs!` instead of re-adapting GPU cell lists.
    # On the CPU this holds the simulation's own handler. Nothing when no interpolation
    # has run yet.
    cpu_nhs_handler::Ref{Any}
end

function SurfaceReconstructionCallback(reconstruction::SurfaceReconstruction, semi;
                                       fluid_systems=nothing, boundaries=nothing,
                                       interval::Integer=0, dt=0.0,
                                       save_times=Float64[],
                                       output_directory="out", append_timestamp=false,
                                       prefix="surface", formats=(:vtp,), verbose=false,
                                       save_initial_surface=true,
                                       save_final_surface=true, overwrite=false,
                                       compress=1, write_statistics=true,
                                       write_boundaries=false,
                                       interpolated_quantities=())
    interval = validate_save_schedule(interval, dt, save_times,
                                      "setting `interval`, `dt` and `save_times` simultaneously is not supported. Use either `interval`, `dt` or `save_times`.")

    save_times = sort!(collect(Float64.(save_times)))
    all(format -> format in (:vtp, :ply), formats) ||
        throw(ArgumentError("`formats` must be a combination of `:vtp` and `:ply`"))
    formats = Tuple(formats)

    interpolated_quantities = collect(Symbol, interpolated_quantities)
    all(quantity -> quantity in INTERPOLATED_SURFACE_QUANTITIES,
        interpolated_quantities) ||
        throw(ArgumentError("`interpolated_quantities` must be a combination of " *
                            join(repr.(INTERPOLATED_SURFACE_QUANTITIES), ", ")))
    isempty(interpolated_quantities) || :vtp in formats ||
        throw(ArgumentError("interpolated quantities are written to VTK files; " *
                            "include `:vtp` in `formats`"))

    if append_timestamp
        output_directory *= string("_", Dates.format(now(), "YY-mm-ddTHHMMSS"))
    end

    fluid_indices = resolve_fluid_indices(fluid_systems, semi)
    boundary_indices, static_boundaries = resolve_boundaries(boundaries, semi)
    all(index -> ndims(semi.systems[index]) == ndims(reconstruction), fluid_indices) ||
        throw(ArgumentError("fluid dimensions must match the reconstruction; use `ndims=2` for planar contours"))
    all(index -> ndims(semi.systems[index]) == ndims(reconstruction), boundary_indices) &&
    all(boundary -> ndims(boundary) == ndims(reconstruction), static_boundaries) ||
        throw(ArgumentError("boundary dimensions must match the reconstruction"))

    # The grid, the Gaussian width, and the deposited volumes all scale with the
    # particle spacing; warn when the reconstruction was configured inconsistently.
    reference_spacing = Float64(particle_spacing(semi.systems[first(fluid_indices)], 1))
    if !isapprox(reconstruction.particle_spacing, reference_spacing)
        @warn "`SurfaceReconstruction` particle spacing " *
              "$(reconstruction.particle_spacing) does not match the fluid system " *
              "spacing $reference_spacing"
    end

    n_fluids = length(fluid_indices)
    # One warm-start state per fluid system (sharing the workspace buffers); the first
    # system uses the passed object itself, so its cache reflects the latest frame.
    reconstructions = [position == 1 ? reconstruction :
                       independent_warm_start(reconstruction)
                       for position in 1:n_fluids]
    surface_callback = SurfaceReconstructionCallback(interval, save_times,
                                                     save_initial_surface,
                                                     save_final_surface, overwrite,
                                                     compress, formats, fluid_indices,
                                                     boundary_indices, reconstruction,
                                                     reconstructions,
                                                     output_directory, String(prefix),
                                                     verbose,
                                                     Dict{Int, BoundaryTopology}(),
                                                     static_boundaries, write_statistics,
                                                     write_boundaries,
                                                     interpolated_quantities,
                                                     Ref(""), Float64[],
                                                     Dict{String, Vector{Float64}}(),
                                                     Vector{Union{Nothing,
                                                                  SurfaceMesh}}(nothing,
                                                                                n_fluids),
                                                     Vector{Union{Nothing,
                                                                  SurfaceReconstructionStatistics}}(nothing,
                                                                                                    n_fluids),
                                                     nothing, nothing, false, -1,
                                                     Ref{Any}(nothing))

    if length(save_times) > 0
        return PresetTimeCallback(copy(save_times), surface_callback;
                                  initialize=(initialize_surface_reconstruction_times_cb!),
                                  save_positions=(false, false))
    elseif dt > 0
        # Add a `tstop` every `dt`, and reconstruct at the final time
        return PeriodicCallback(surface_callback, dt,
                                initialize=(initialize_surface_reconstruction_cb!),
                                save_positions=(false, false),
                                final_affect=save_final_surface)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(surface_callback, surface_callback,
                                save_positions=(false, false),
                                initialize=(initialize_surface_reconstruction_cb!))
    end
end

function resolve_fluid_indices(fluid_systems, semi)
    systems = semi.systems

    if fluid_systems === nothing
        indices = findall(system -> system isa AbstractFluidSystem, systems)
        isempty(indices) &&
            throw(ArgumentError("no fluid system found; specify `fluid_systems`"))
    elseif fluid_systems isa Union{Integer, AbstractSystem}
        indices = [system_index_spec(fluid_systems, semi)]
    else
        indices = [system_index_spec(spec, semi) for spec in fluid_systems]
    end

    for system_index in indices
        systems[system_index] isa AbstractFluidSystem ||
            throw(ArgumentError("system $system_index is not a fluid system"))
        ndims(systems[system_index]) in (2, 3) ||
            throw(ArgumentError("surface reconstruction requires a 2D or 3D fluid system"))
    end

    return indices
end

# Boundaries are systems (tracked through their current coordinates) or static geometries
# (`TriangleMesh`, e.g. from `load_geometry`, or a prebuilt `BoundaryMesh`).
function resolve_boundaries(boundaries, semi)
    systems = semi.systems
    single_specification = Union{Integer, AbstractSystem, TriangleMesh, Polygon,
                                 BoundaryMesh}
    specifications = boundaries === nothing ? () :
                     boundaries isa single_specification ? (boundaries,) : boundaries

    indices = Int[]
    static_boundaries = BoundaryMesh[]
    for specification in specifications
        if specification isa BoundaryMesh
            push!(static_boundaries, specification)
        elseif specification isa TriangleMesh
            ndims(specification) == 3 ||
                throw(ArgumentError("surface reconstruction requires 3D boundary geometries"))
            push!(static_boundaries, BoundaryMesh(specification))
        elseif specification isa Polygon{2}
            push!(static_boundaries, BoundaryMesh(specification))
        else
            system_index = system_index_spec(specification, semi)
            system = systems[system_index]
            system isa Union{AbstractBoundarySystem, AbstractStructureSystem} ||
                throw(ArgumentError("system $system_index is not a boundary or structure system"))
            ndims(system) in (2, 3) ||
                throw(ArgumentError("surface reconstruction requires 2D or 3D boundary and structure systems"))
            push!(indices, system_index)
        end
    end

    return indices, static_boundaries
end

function initialize_surface_reconstruction_cb!(cb, u, t, integrator)
    # The `SurfaceReconstructionCallback` is either `cb.affect!` (with `DiscreteCallback`
    # or `PresetTimeCallback`) or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initialize_surface_reconstruction_cb!(cb.affect!, u, t, integrator)
end

function initialize_surface_reconstruction_cb!(surface_callback::SurfaceReconstructionCallback,
                                               u, t, integrator)
    initialize_surface_reconstruction_cb!(surface_callback, u, t, integrator,
                                          surface_callback.save_initial_surface)
end

function initialize_surface_reconstruction_cb!(surface_callback::SurfaceReconstructionCallback,
                                               u, t, integrator, save_initial)
    semi = integrator.p.semi
    set_callbacks_used!(semi, integrator)

    fill!(surface_callback.latest_meshes, nothing)
    fill!(surface_callback.latest_statistics, nothing)
    surface_callback.latest_mesh = nothing
    surface_callback.latest_stats = nothing
    surface_callback.collection_initialized = false
    surface_callback.latest_saved_iter = -1
    # Caches belong to a particular solve and its system ordering. A reused callback
    # must not carry a previous solve's CPU search, topology, or correction seed.
    surface_callback.cpu_nhs_handler[] = nothing
    empty!(surface_callback.boundary_topologies)
    for reconstruction in surface_callback.reconstructions
        reconstruction.cache.previous_isovalue[] = reconstruction.options.isovalue
        reconstruction.cache.last_mesh_analysis[] = nothing
    end
    empty!(surface_callback.statistics_times)
    empty!(surface_callback.statistics_data)
    surface_callback.git_hash[] = compute_git_hash()

    mkpath(surface_callback.output_directory)
    write_surface_meta_data(surface_callback, integrator)

    # Reconstruct the initial surface
    if save_initial
        surface_callback(integrator)
    end

    return nothing
end

function initialize_surface_reconstruction_times_cb!(cb, u, t, integrator)
    surface_callback = cb.affect!
    reset_save_times!(cb.condition.tstops, surface_callback.save_times)
    add_final_save_time!(cb.condition.tstops, surface_callback, integrator)

    # `PresetTimeCallback` calls `affect!` after this initializer when `t` is in
    # `tstops`. Avoid reconstructing the initial surface twice when it is also a save time.
    save_initial = surface_callback.save_initial_surface &&
                   !insorted(t, cb.condition.tstops)
    initialize_surface_reconstruction_cb!(surface_callback, u, t, integrator,
                                          save_initial)
end

function add_final_save_time!(save_times, surface_callback::SurfaceReconstructionCallback,
                              integrator)
    if surface_callback.save_final_surface
        push!(save_times, last(integrator.sol.prob.tspan))
        sort!(unique!(save_times))
    end

    return nothing
end

# `condition`
function (surface_callback::SurfaceReconstructionCallback)(u, t, integrator)
    (; interval, save_final_surface) = surface_callback

    return condition_integrator_interval(integrator, interval;
                                         save_final_solution=save_final_surface)
end

# `affect!`
function (surface_callback::SurfaceReconstructionCallback)(integrator)
    (; output_directory, verbose, fluid_indices, boundary_indices, boundary_topologies,
     static_boundaries, interpolated_quantities) = surface_callback

    @trixi_timeit timer() "reconstruct surface" begin
        semi = integrator.p.semi
        t = integrator.t
        v_ode, u_ode = integrator.u.x

        # Update quantities that are stored in the systems (e.g. the density for
        # `SummationDensity`), which still hold the values from the last stage of the
        # previous step otherwise.
        @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)

        # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
        # Without interpolation, skip the neighborhood-search handler: adapting its GPU
        # cell lists dominates the transfer cost and nothing below uses it.
        if isempty(interpolated_quantities)
            v_ode_cpu, u_ode_cpu, semi_cpu = transfer2cpu_system_state(v_ode, u_ode,
                                                                       semi)
        else
            cached_handler = surface_callback.cpu_nhs_handler[]
            if isnothing(cached_handler)
                # First interpolation event: full transfer, then keep the CPU handler
                v_ode_cpu, u_ode_cpu, semi_cpu = transfer2cpu(v_ode, u_ode, semi)
                surface_callback.cpu_nhs_handler[] = semi_cpu.neighborhood_search_handler
            else
                v_ode_cpu, u_ode_cpu,
                semi_cpu = transfer2cpu_system_state(v_ode, u_ode,
                                                     semi)
                semi_cpu = @set semi_cpu.neighborhood_search_handler = cached_handler
                # `interpolate_points` refreshes the handler below with `update_nhs!`
            end
        end

        system_boundaries = map(boundary_indices) do system_index
            boundary_system = semi_cpu.systems[system_index]
            u_boundary = wrap_u(u_ode_cpu, boundary_system, semi_cpu)
            current = Matrix{Float64}(active_coordinates(u_boundary, boundary_system))
            topology = get!(boundary_topologies, system_index) do
                reference = Matrix{Float64}(initial_coordinates(boundary_system))
                return lattice_surface_topology(reference)
            end
            return BoundaryMesh(current, topology)
        end
        boundaries = vcat(static_boundaries, system_boundaries)

        iter = get_iter(surface_callback.interval, integrator)

        if iter == surface_callback.latest_saved_iter
            # This should only happen at the end of the simulation when using `dt` and the
            # final time is not a multiple of the reconstruction interval.
            @assert isfinished(integrator)

            # Avoid overwriting the previous file
            iter += 1
        end

        if verbose
            println("Writing reconstructed surfaces to $output_directory at t = $(integrator.t)")
        end

        names = system_names(semi_cpu.systems)
        for (position, fluid_index) in enumerate(fluid_indices)
            fluid_system = semi_cpu.systems[fluid_index]

            # Skip systems without active particles, like empty VTK systems are skipped
            if isempty(each_active_particle(fluid_system))
                surface_callback.latest_meshes[position] = nothing
                surface_callback.latest_statistics[position] = nothing
                surface_callback.reconstructions[position].cache.last_mesh_analysis[] = nothing
                if position == 1
                    surface_callback.latest_mesh = nothing
                    surface_callback.latest_stats = nothing
                end
                continue
            end

            v_fluid = wrap_v(v_ode_cpu, fluid_system, semi_cpu)
            u_fluid = wrap_u(u_ode_cpu, fluid_system, semi_cpu)
            points = active_surface_points(fluid_system, u_fluid)
            volumes = particle_volumes(fluid_system, v_fluid)

            mesh,
            stats = reconstruct_surface!(surface_callback.reconstructions[position],
                                         points, volumes, boundaries;
                                         parallelization_backend=semi_cpu.parallelization_backend)
            # The writers recompute the liquid analysis for normals and winding; reuse
            # the pipeline's analysis when it describes this exact mesh object
            cached = surface_callback.reconstructions[position].cache.last_mesh_analysis[]
            mesh_analysis = (!isnothing(cached) && cached[1] === mesh) ? cached[2] :
                            nothing
            surface_callback.latest_meshes[position] = mesh
            surface_callback.latest_statistics[position] = stats
            if position == 1
                surface_callback.latest_mesh = mesh
                surface_callback.latest_stats = stats
            end

            point_data = nothing
            if !isempty(interpolated_quantities) && !isempty(mesh.vertices)
                # SPH interpolation from fluid neighbors only (`cut_off_bnd=false`), so
                # vertices touching walls keep fluid values
                vertex_coordinates = Matrix{Float64}(undef, ndims(mesh),
                                                     length(mesh.vertices))
                for (index, vertex) in enumerate(mesh.vertices)
                    vertex_coordinates[:, index] = vertex
                end
                interpolated = interpolate_points(vertex_coordinates, semi_cpu,
                                                  fluid_system, v_ode_cpu, u_ode_cpu;
                                                  cut_off_bnd=false)
                point_data = Dict(string(quantity) => getproperty(interpolated, quantity)
                                  for quantity in interpolated_quantities)
            end

            write_surface_files(surface_callback, mesh, names[fluid_index], iter, t;
                                point_data, analysis=mesh_analysis)
        end

        if surface_callback.write_boundaries
            for (system_index, boundary) in zip(boundary_indices, system_boundaries)
                write_surface_files(surface_callback, boundary_surface_mesh(boundary),
                                    names[system_index], iter, t)
            end
        end

        if surface_callback.write_statistics
            record_surface_statistics!(surface_callback, t, names)
            write_surface_statistics(surface_callback, integrator)
        end

        # Reset the collection on the first write of each callback run to drop stale
        # PVD entries. Later writes append, including writes after an initial surface.
        surface_callback.collection_initialized = true
        surface_callback.latest_saved_iter = iter
    end

    # This callback only processes results and does not change the result of the right-hand side.
    derivative_discontinuity!(integrator, false)

    return nothing
end

const INTERPOLATED_SURFACE_QUANTITIES = (:velocity, :pressure, :density)

# Quantities recorded per fluid system and reconstruction event
const SURFACE_STATISTICS_QUANTITIES = ("effective_isovalue", "volume", "particle_volume",
                                       "relative_volume_error", "surface_area",
                                       "n_vertices", "n_faces", "n_connected_regions",
                                       "n_cavity_regions", "n_boundary_edges",
                                       "n_nonmanifold_edges", "n_correction_evaluations",
                                       "n_excluded_particles", "n_enclosed_particles",
                                       "reconstruction_time")

const PLANAR_SURFACE_STATISTICS_QUANTITIES = ("effective_isovalue", "area", "particle_area",
                                              "relative_area_error", "perimeter",
                                              "n_vertices", "n_segments",
                                              "n_connected_regions",
                                              "n_cavity_regions",
                                              "n_correction_evaluations",
                                              "n_excluded_particles",
                                              "n_enclosed_particles",
                                              "reconstruction_time")

function surface_statistics_quantities(callback)
    ndims(callback.reconstruction) == 2 ?
    PLANAR_SURFACE_STATISTICS_QUANTITIES :
    SURFACE_STATISTICS_QUANTITIES
end

function write_surface_files(surface_callback, mesh, system_name, iter, t;
                             point_data=nothing, analysis=nothing)
    (; output_directory, prefix, formats, overwrite, compress) = surface_callback

    for format in formats
        if format === :vtp
            trixi2vtk(mesh; output_directory, prefix, filename=system_name,
                      iter=overwrite ? nothing : iter, overwrite,
                      append_collection=surface_callback.collection_initialized,
                      t, compress, point_data, analysis)
        else
            base = joinpath(output_directory,
                            add_underscore_to_optional_prefix(prefix) * system_name)
            path = overwrite ? base * "_current.ply" :
                   base * "_" * lpad(iter, 6, '0') * ".ply"
            write_ply(mesh, path; analysis)
        end
    end

    return nothing
end

function surface_statistics_values(mesh, stats)
    details = statistics_dict(stats)
    timings = get(details, "timings", nothing)
    reconstruction_time = timings isa AbstractDict ? get(timings, "total", NaN) : NaN
    relative_volume_error = (stats.volume - stats.particle_volume) / stats.particle_volume

    values = Dict("effective_isovalue" => stats.effective_isovalue,
                  "volume" => stats.volume,
                  "particle_volume" => stats.particle_volume,
                  "relative_volume_error" => relative_volume_error,
                  "surface_area" => stats.surface_area,
                  "n_vertices" => length(mesh.vertices),
                  "n_faces" => length(mesh.faces),
                  "n_connected_regions" => stats.n_connected_regions,
                  "n_cavity_regions" => stats.n_cavity_regions,
                  "n_boundary_edges" => stats.n_boundary_edges,
                  "n_nonmanifold_edges" => stats.n_nonmanifold_edges,
                  "n_correction_evaluations" => stats.n_correction_evaluations,
                  "n_excluded_particles" => get(details, "n_excluded_particles", 0),
                  "n_enclosed_particles" => get(details, "n_enclosed_particles", 0),
                  "reconstruction_time" => reconstruction_time)
    if ndims(mesh) == 2
        merge!(values,
               Dict("area" => stats.volume, "particle_area" => stats.particle_volume,
                    "relative_area_error" => relative_volume_error,
                    "perimeter" => stats.surface_area, "n_segments" => length(mesh.faces)))
    end
    return values
end

# Append one event to the statistics time series. Skipped systems (no active particles)
# are recorded as `NaN`, so all series keep the length of the time vector.
function record_surface_statistics!(surface_callback, t, names)
    (; fluid_indices, latest_meshes, latest_statistics, statistics_data) = surface_callback

    push!(surface_callback.statistics_times, t)
    for (position, fluid_index) in enumerate(fluid_indices)
        mesh = latest_meshes[position]
        stats = latest_statistics[position]
        values = stats === nothing ? nothing : surface_statistics_values(mesh, stats)
        for quantity in surface_statistics_quantities(surface_callback)
            series = get!(statistics_data, quantity * "_" * names[fluid_index],
                          fill(NaN, length(surface_callback.statistics_times) - 1))
            push!(series, values === nothing ? NaN : Float64(values[quantity]))
        end
    end

    return surface_callback
end

# The files follow the conventions of the `PostprocessCallback` and are rewritten after
# every event, so they are complete even if the simulation stops early.
function write_surface_statistics(surface_callback, integrator)
    (; output_directory, prefix, statistics_times, statistics_data) = surface_callback

    data = Dict{String, Any}()
    data["meta"] = surface_meta_data(surface_callback, integrator)
    for (key, values) in statistics_data
        # Keys are `<quantity>_<system name>`, e.g. `volume_fluid_1`
        quantity = first(filter(quantity -> startswith(key, quantity * "_"),
                                surface_statistics_quantities(surface_callback)))
        system_name = key[(length(quantity) + 2):end]
        data[key] = create_series_dict(values, statistics_times, system_name)
    end

    filename = joinpath(abspath(output_directory),
                        add_underscore_to_optional_prefix(prefix) * "statistics")
    write_time_series_files(filename * ".json", filename * ".csv", data)

    return nothing
end

function surface_meta_data(surface_callback, integrator)
    meta_data = create_meta_data_dict(surface_callback, integrator)
    meta_data["surface_reconstruction"] = reconstruction_configuration(surface_callback,
                                                                       integrator.p.semi)
    return meta_data
end

# Metadata file like the one of the `SolutionSavingCallback` (`meta_<prefix>.json`)
function write_surface_meta_data(surface_callback, integrator)
    (; output_directory, prefix) = surface_callback

    file = joinpath(output_directory,
                    "meta" * add_underscore_to_optional_postfix(prefix) * ".json")
    open(file, "w") do io
        JSON.json(io, surface_meta_data(surface_callback, integrator); pretty=2,
                  allownan=true)
    end

    return nothing
end

function reconstruction_configuration(surface_callback, semi)
    (; reconstruction, fluid_indices, boundary_indices, static_boundaries, formats,
     interpolated_quantities, interval, save_times) = surface_callback
    (; options, min_corner, max_corner, open_faces) = reconstruction
    names = system_names(semi.systems)
    face_names = ("-x", "+x", "-y", "+y", "-z", "+z")

    configuration = Dict{String, Any}("ndims" => ndims(reconstruction),
                                      "particle_spacing" => reconstruction.particle_spacing,
                                      "voxel_size" => reconstruction.voxel_size,
                                      "gaussian_sigma_voxels" => reconstruction.sigma_voxels,
                                      "grid_padding" => reconstruction.grid_padding,
                                      "boundary_clearance" => reconstruction.boundary_clearance,
                                      "isovalue" => options.isovalue,
                                      "volume_tolerance_percent" => options.volume_tolerance_percent,
                                      "volume_max_iterations" => options.volume_max_iterations,
                                      "isovalue_range" => [options.minimum_isovalue,
                                          options.maximum_isovalue],
                                      "warm_start" => options.warm_start,
                                      "sparse_component_fallback" => options.sparse_component_fallback,
                                      "keep_enclosed_fluid" => options.keep_enclosed_fluid,
                                      "parallelization_backend" => reconstruction.parallelization_backend ===
                                                                   nothing ?
                                                                   "inherited" :
                                                                   type2string(reconstruction.parallelization_backend),
                                      "fluid_systems" => names[fluid_indices],
                                      "boundary_systems" => names[boundary_indices],
                                      "n_static_boundaries" => length(static_boundaries),
                                      "formats" => [string(format) for format in formats],
                                      "interpolated_quantities" => string.(interpolated_quantities)
                                      )

    if min_corner === nothing
        configuration["domain"] = "adaptive"
    else
        configuration["domain"] = Dict("min_corner" => collect(min_corner),
                                       "max_corner" => collect(max_corner),
                                       "open_faces" => [face_names[face]
                                                        for face in eachindex(open_faces)
                                                        if open_faces[face]])
    end

    if !isempty(save_times)
        configuration["save_times"] = save_times
    elseif interval isa AbstractFloat
        configuration["dt"] = interval
    else
        configuration["interval"] = interval
    end

    return configuration
end

# With `interval`
function Base.show(io::IO,
                   cb::DiscreteCallback{<:SurfaceReconstructionCallback,
                                        <:SurfaceReconstructionCallback})
    @nospecialize cb # reduce precompilation time

    surface_reconstruction = cb.affect!
    print(io, "SurfaceReconstructionCallback(interval=", surface_reconstruction.interval,
          ")")
end

# With `dt`
function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SurfaceReconstructionCallback}})
    @nospecialize cb # reduce precompilation time

    surface_reconstruction = cb.affect!.affect!
    print(io, "SurfaceReconstructionCallback(dt=", surface_reconstruction.interval, ")")
end

# With `save_times`
function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:SurfaceReconstructionCallback})
    @nospecialize cb # reduce precompilation time

    surface_callback = cb.affect!
    print(io, "SurfaceReconstructionCallback(save_times=",
          surface_callback.save_times, ")")
end

# With `interval`
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:SurfaceReconstructionCallback,
                                        <:SurfaceReconstructionCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        surface_reconstruction = cb.affect!

        setup = [
            "interval" => surface_reconstruction.interval,
            "save initial surface" => surface_reconstruction.save_initial_surface ?
                                      "yes" : "no",
            "save final surface" => surface_reconstruction.save_final_surface ? "yes" :
                                    "no",
            "output directory" => abspath(surface_reconstruction.output_directory),
            "prefix" => surface_reconstruction.prefix,
            "formats" => join(String.(surface_reconstruction.formats), ", ")
        ]
        summary_box(io, "SurfaceReconstructionCallback", setup)
    end
end

# With `dt`
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SurfaceReconstructionCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        surface_reconstruction = cb.affect!.affect!

        setup = [
            "dt" => surface_reconstruction.interval,
            "save initial surface" => surface_reconstruction.save_initial_surface ?
                                      "yes" : "no",
            "save final surface" => surface_reconstruction.save_final_surface ? "yes" :
                                    "no",
            "output directory" => abspath(surface_reconstruction.output_directory),
            "prefix" => surface_reconstruction.prefix,
            "formats" => join(String.(surface_reconstruction.formats), ", ")
        ]
        summary_box(io, "SurfaceReconstructionCallback", setup)
    end
end

# With `save_times`
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SurfaceReconstructionCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        surface_callback = cb.affect!

        setup = [
            "save times" => surface_callback.save_times,
            "save initial surface" => surface_callback.save_initial_surface ?
                                      "yes" : "no",
            "save final surface" => surface_callback.save_final_surface ? "yes" :
                                    "no",
            "output directory" => abspath(surface_callback.output_directory),
            "prefix" => surface_callback.prefix,
            "formats" => join(String.(surface_callback.formats), ", ")
        ]
        summary_box(io, "SurfaceReconstructionCallback", setup)
    end
end
