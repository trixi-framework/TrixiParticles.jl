#!/usr/bin/env julia
# Cross-thread-count determinism check for the reconstruction pipeline.
#
# Prints mesh checksums and selected statistics for the reference block and two wave tanks.
# The medium case has enough faces to exercise fixed-bucket parallel mesh reductions.
# Run twice with different thread counts and compare the output:
#
#   JULIA_NUM_THREADS=1 julia --project=benchmark/surface_reconstruction \
#       benchmark/surface_reconstruction/determinism.jl > det1.txt
#   JULIA_NUM_THREADS=4 julia --project=benchmark/surface_reconstruction \
#       benchmark/surface_reconstruction/determinism.jl > det4.txt
#   diff <(grep -v "^threads=" det1.txt) <(grep -v "^threads=" det4.txt) && echo DETERMINISTIC
#
# Grid writes are disjoint and reduction buckets have fixed bounds, so the results must
# agree bitwise across thread counts on the same platform.
using Pkg

Pkg.develop(path=normpath(joinpath(@__DIR__, "..", "..")))
Pkg.instantiate()

include(joinpath(@__DIR__, "workloads.jl"))

using .Workloads
using TrixiParticles
using SHA: SHA

function report(name, (mesh, stats))
    bytes = vcat(reinterpret(UInt8, mesh.vertices), reinterpret(UInt8, mesh.faces))
    println("$name: mesh_sha256=$(bytes2hex(SHA.sha256(bytes)))")
    println("$name: vertices=$(length(mesh.vertices)) faces=$(length(mesh.faces))")
    println("$name: effective_isovalue=$(repr(stats["effective_isovalue"]))")
    println("$name: volume=$(repr(stats["volume"]))")
end

function main()
    println("threads=$(Threads.nthreads())")

    # Numerical reference block
    spacing = 0.1
    points, volumes = Workloads.lattice_block(7, spacing)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           voxel_size=spacing / 2,
                                           tank_size=(1.0, 1.0, 1.0))
    report("block", reconstruct_surface!(reconstruction, points, volumes))

    # Coarse wave tank, large enough to split all threaded loops and with several
    # isovalue correction evaluations
    spacing = 0.03
    points, volumes = Workloads.wave_tank_particles(spacing)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           tank_size=(2.0, 4.0, 1.0))
    report("wave_tank", reconstruct_surface!(reconstruction, points, volumes))

    # Explicitly guard coverage above the parallel mesh-reduction threshold, even if
    # the other workloads' particle counts or grid dimensions change.
    spacing = 0.018
    points, volumes = Workloads.wave_tank_particles(spacing)
    reconstruction = SurfaceReconstruction(; particle_spacing=spacing,
                                           tank_size=(2.0, 4.0, 1.0))
    mesh, stats = reconstruct_surface!(reconstruction, points, volumes)
    length(mesh.faces) >= TrixiParticles.PARALLEL_REDUCTION_MIN_FACES ||
        error("wave-tank determinism case does not exercise parallel mesh reductions")
    report("wave_tank_mesh_reductions", (mesh, stats))

    return 0
end

exit(main())
