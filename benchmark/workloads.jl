# Benchmark workloads for the surface reconstruction pipeline.
#
# All workloads are deterministic (no RNG) and scale on one size parameter.
# Timings use `time_ns`; allocations use `@allocated` on a warm workspace.
module Workloads

using TrixiParticles

const REPEATS = Ref(3)

set_repeats!(n) = (REPEATS[] = n)

# Note that `reduce(hcat, A)` needs a vector `A` to return a plain `Matrix`. For a 3D array
# of `SVector`s, it concatenates pairwise into ever larger `SMatrix`es, which takes minutes
# to compile for a few hundred points.
function lattice_block(n_side, spacing)
    points = reduce(hcat,
                    vec([SVector{3, Float64}(x, y, z) * spacing
                         for z in 0:(n_side - 1), y in 0:(n_side - 1),
                             x in 0:(n_side - 1)]))
    volumes = fill(spacing^3, size(points, 2))
    return points, volumes
end

function boundary_cube(lower, side)
    return reduce(hcat,
                  vec([SVector{3, Float64}(lower[1], lower[2], lower[3]) +
                       side * SVector{3, Float64}(i, j, k)
                       for i in (0.0, 1.0), j in (0.0, 1.0), k in (0.0, 1.0)]))
end

function measure(name, f; repeats=REPEATS[], evaluations=1, warmup=1)
    for _ in 1:warmup
        f()
    end
    # Short workloads are evaluated several times per sample to average out timer noise.
    # The number of calls is fixed, so results that depend on the warm start (e.g. the
    # sparse fallback, whose two correction passes have different targets) are reproducible.
    times = Float64[]
    results = nothing
    for _ in 1:repeats
        GC.gc(false)
        start_time = time_ns()
        for _ in 1:evaluations
            results = f()
        end
        push!(times, (time_ns() - start_time) / 1.0e9 / evaluations)
    end
    GC.gc(false)
    allocated = @allocated f()

    return (name=name, time_min=minimum(times), time_median=median(times),
            allocated_bytes=allocated, result=results)
end

median(values) = sort(values)[cld(length(values), 2)]

function workload_block_small()
    spacing = 0.1
    points, volumes = lattice_block(7, spacing)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           voxel_size=spacing / 2)
    measured = measure("block_small",
                       () -> reconstruct_surface!(reconstruction, points, volumes);
                       evaluations=50)
    return report(measured, reconstruction, volumes)
end

function workload_block_scaling(particle_count)
    # Cubic lattices with approximately the requested particle count
    n_side = round(Int, particle_count^(1 / 3))
    spacing = 0.05
    points, volumes = lattice_block(n_side, spacing)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           voxel_size=spacing / 2)
    measured = measure("block_$(size(points, 2))",
                       () -> reconstruct_surface!(reconstruction, points, volumes);
                       evaluations=5)
    return report(measured, reconstruction, volumes)
end

function workload_boundaries()
    spacing = 0.1
    points, volumes = lattice_block(7, spacing)
    cube = boundary_cube((0.2, 0.2, 0.2), 0.3)
    topology = TrixiParticles.lattice_surface_topology(cube)
    boundary = TrixiParticles.BoundaryMesh(cube, topology)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           voxel_size=spacing / 2)
    measured = measure("boundaries",
                       () -> reconstruct_surface!(reconstruction, points, volumes,
                                                  [boundary]);
                       evaluations=20)
    return report(measured, reconstruction, volumes)
end

function workload_sparse()
    spacing = 0.1
    points, volumes = lattice_block(7, spacing)
    # Isolated droplet far from the block, as in the sparse reference case
    points = hcat(points, reshape([0.9, 0.9, 0.9], 3, 1))
    volumes = vcat(volumes, spacing^3)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           voxel_size=spacing / 2,
                                           sparse_component_fallback=true)
    measured = measure("sparse",
                       () -> reconstruct_surface!(reconstruction, points, volumes);
                       evaluations=10)
    return report(measured, reconstruction, volumes)
end

function wave_tank_particles(spacing)
    # Synthetic production-scale frame: a 2 m x 4 m x 1 m tank filled up to a sinusoidal
    # free surface with a mean height of 0.18 m
    n_x, n_z = floor(Int, 2.0 / spacing), floor(Int, 1.0 / spacing)
    points = SVector{3, Float64}[]
    for iz in 0:(n_z - 1), ix in 0:(n_x - 1)
        x, z = (ix + 0.5) * spacing, (iz + 0.5) * spacing
        height = 0.18 + 0.05 * sin(2pi * x / 1.0)
        for iy in 0:floor(Int, height / spacing)
            push!(points, SVector(x, (iy + 0.5) * spacing, z))
        end
    end
    return Matrix(reduce(hcat, points)), fill(spacing^3, length(points))
end

function workload_wave_tank()
    # Particle spacing and tank size of the production wave tank (1.1e6 particles)
    spacing = 0.006885
    points, volumes = wave_tank_particles(spacing)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           tank_size=(2.0, 4.0, 1.0))
    # Restart the isovalue correction at the base isovalue in every repeat to time the
    # full correction of a new frame instead of the warm-started steady state
    measured = measure("wave_tank",
                       () -> reconstruct_surface!(reconstruction, points, volumes;
                                                  initial_isovalue=0.5))
    return report(measured, reconstruction, volumes)
end

function report(measured, reconstruction, volumes)
    mesh, stats = measured.result
    evaluations = stats["isovalue_correction_evaluations"]
    workspace = reconstruction.cache.workspace[]
    workspace_bytes = sizeof(workspace.field) +
                      sizeof(workspace.constraint) +
                      sizeof(workspace.temporary) +
                      sizeof(workspace.scratch)

    return Dict{String, Any}(
        "name" => measured.name,
        "time_min_seconds" => measured.time_min,
        "time_median_seconds" => measured.time_median,
        "allocated_bytes" => measured.allocated_bytes,
        "workspace_bytes" => workspace_bytes,
        "particles" => length(volumes),
        "grid_dimensions" => collect(stats["grid_dimensions"]),
        "n_correction_evaluations" => length(evaluations),
        "vertices" => length(mesh.vertices),
        "faces" => length(mesh.faces),
        "volume" => stats["volume"],
        "effective_isovalue" => stats["effective_isovalue"]
    )
end

const WORKLOADS = Dict{String, Function}(
    "block_small" => workload_block_small,
    "block_1e4" => () -> workload_block_scaling(10_000),
    "boundaries" => workload_boundaries,
    "sparse" => workload_sparse,
    "wave_tank" => workload_wave_tank
)

# Workloads run when `--workloads` is not given. `wave_tank` is opt-in because of its
# runtime and memory footprint.
const DEFAULT_WORKLOADS = ["block_small", "block_1e4", "boundaries", "sparse"]

end # module
