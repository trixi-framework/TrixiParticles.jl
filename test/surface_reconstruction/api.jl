@trixi_testset "SurfaceReconstruction" begin
    using LinearAlgebra: norm, cross

    # The same lattice is used for constructor, cache, and output tests. Changing its
    # extent can affect both the adaptive grid and which box faces clip the contour.
    particle_spacing = 0.1
    # `vec` makes `reduce(hcat, ...)` return a `Matrix` instead of a slow-to-compile `SMatrix`
    points = reduce(hcat,
                    vec([TrixiParticles.SVector{3, Float64}(x, y, z) * particle_spacing
                         for z in 0:6, y in 0:6, x in 0:6]))
    volumes = fill(particle_spacing^3, size(points, 2))

    @testset "workspace reuse and warm start" begin
        # A second frame with the same grid reuses buffers and remembers the selected
        # isovalue; the fresh/moving-grid cases below exercise cache invalidation.
        reconstruction = SurfaceReconstruction(; particle_spacing=particle_spacing)
        @test reconstruction.cache.workspace[] === nothing
        mesh_first, _ = reconstruct_surface!(reconstruction, points, volumes)
        @test reconstruction.cache.workspace[] !== nothing
        mesh_second, stats_second = reconstruct_surface!(reconstruction, points, volumes)
        @test reconstruction.cache.previous_isovalue[] == stats_second["effective_isovalue"]
        @test length(mesh_second.vertices) == length(mesh_first.vertices)
    end

    @testset "backend determinism" begin
        serial = SurfaceReconstruction(; particle_spacing=particle_spacing,
                                       voxel_size=particle_spacing / 2,
                                       tank_size=(1.0, 1.0, 1.0),
                                       parallelization_backend=TrixiParticles.SerialBackend())
        threaded = SurfaceReconstruction(; particle_spacing=particle_spacing,
                                         voxel_size=particle_spacing / 2,
                                         tank_size=(1.0, 1.0, 1.0))
        mesh_serial, stats_serial = reconstruct_surface!(serial, points, volumes)
        mesh_threaded, stats_threaded = reconstruct_surface!(threaded, points, volumes)
        # This small mesh keeps float statistics serial. The threaded grid passes must
        # then agree bitwise with the serial backend.
        @test mesh_serial.vertices == mesh_threaded.vertices
        @test mesh_serial.faces == mesh_threaded.faces
        @test stats_serial["effective_isovalue"] == stats_threaded["effective_isovalue"]
        @test stats_serial["volume"] == stats_threaded["volume"]
        # Repeated invocations are exactly reproducible as well
        mesh_repeat, _ = reconstruct_surface!(threaded, points, volumes)
        @test mesh_repeat.vertices == mesh_threaded.vertices
        @test mesh_repeat.faces == mesh_threaded.faces
    end

    @testset "reconstruction grid and domain" begin
        voxel_size = particle_spacing / 2
        padding = 4 * particle_spacing

        grid, domain = TrixiParticles.reconstruction_grid(points, voxel_size, padding)
        @test grid isa TrixiParticles.ReconstructionGrid
        @test grid.spacing == voxel_size
        @test all(grid.origin .≈ 0 - padding - voxel_size / 2)
        # Adaptive: closed box equal to the grid, i.e. inactive for the particles
        @test domain.open_faces == (false, false, false, false, false, false)
        @test domain.min_corner == grid.origin
        @test all(TrixiParticles.domain_distance(domain, points[1, i], points[2, i],
                                                 points[3, i]) > padding
                  for i in axes(points, 2))

        grid_tank,
        domain_tank = TrixiParticles.reconstruction_grid(points, voxel_size,
                                                         padding;
                                                         min_corner=(0.0, 0.0, 0.0),
                                                         max_corner=(1.0, 1.0, 1.0),
                                                         open_faces=(false, false,
                                                                     false, true,
                                                                     false, false))
        # Fixed tank: production convention (origin corner, open top)
        @test domain_tank.open_faces == (false, false, false, true, false, false)
        @test domain_tank.min_corner == zero(SVector{3, Float64})
        @test domain_tank.max_corner == SVector(1.0, 1.0, 1.0)
        @test all(grid_tank.origin .≈ -padding - voxel_size / 2)
        # Open top: no clipping above the tank, walls clip on the other faces
        @test TrixiParticles.domain_distance(domain_tank, 0.5, 5.0, 0.5) > 0
        @test TrixiParticles.domain_distance(domain_tank, 0.5, 0.5, 1.2) < 0
        @test TrixiParticles.domain_violation(domain_tank, SVector(0.5, 0.5, 0.5)) == 0
        @test TrixiParticles.domain_violation(domain_tank, SVector(-0.1, 0.5, 0.5)) ≈ 0.1
    end

    @testset "custom domain corners" begin
        # Explicit corners equivalent to `tank_size` reconstruct identically
        mesh_tank,
        stats_tank = reconstruct_surface(points, volumes;
                                         particle_spacing=particle_spacing,
                                         voxel_size=particle_spacing / 2,
                                         tank_size=(1.0, 1.0, 1.0))
        mesh_corners,
        stats_corners = reconstruct_surface(points, volumes;
                                            particle_spacing=particle_spacing,
                                            voxel_size=particle_spacing / 2,
                                            min_corner=(0.0, 0.0, 0.0),
                                            max_corner=(1.0, 1.0, 1.0),
                                            open_faces=(false, false,
                                                        false, true,
                                                        false, false))
        @test mesh_corners.vertices == mesh_tank.vertices
        @test mesh_corners.faces == mesh_tank.faces
        @test stats_corners["effective_isovalue"] == stats_tank["effective_isovalue"]

        # A lid below the free surface only clips when the +y face is closed. The
        # unclipped surface rises to ~0.72, so a lid at 0.65 discriminates the flag.
        lid = 0.65
        mesh_open,
        stats_open = reconstruct_surface(points, volumes;
                                         particle_spacing=particle_spacing,
                                         voxel_size=particle_spacing / 2,
                                         min_corner=(0.0, 0.0, 0.0),
                                         max_corner=(1.0, lid, 1.0),
                                         open_faces=(false, false, false,
                                                     true, false, false))
        mesh_closed,
        stats_closed = reconstruct_surface(points, volumes;
                                           particle_spacing=particle_spacing,
                                           voxel_size=particle_spacing / 2,
                                           min_corner=(0.0, 0.0, 0.0),
                                           max_corner=(1.0, lid, 1.0))
        @test maximum(vertex[2] for vertex in mesh_open.vertices) > lid + 0.05
        @test maximum(vertex[2] for vertex in mesh_closed.vertices) <= lid + 1.0e-6
        # Clipping keeps the mesh closed and the volume correction satisfied
        @test stats_closed["n_boundary_edges"] == 0
        @test stats_closed["n_nonmanifold_edges"] == 0
        @test stats_closed["closed_boundary_vertices_outside_implicit_domain"] == 0
        @test abs(100 * (stats_closed["volume"] - sum(volumes)) / sum(volumes)) <=
              0.1
        @test stats_closed["effective_isovalue"] < stats_open["effective_isovalue"]

        # With every face open, the domain is inert: on the same grid the result is
        # bitwise identical to the adaptive (unconstrained) reconstruction. Corners are
        # chosen so that the pinned grid origin coincides with the adaptive origin.
        mesh_adaptive,
        stats_adaptive = reconstruct_surface(points, volumes;
                                             particle_spacing=particle_spacing,
                                             voxel_size=particle_spacing / 2)
        grid_adaptive,
        _ = TrixiParticles.reconstruction_grid(points,
                                               particle_spacing / 2,
                                               4 * particle_spacing)
        pad = 4 * particle_spacing + particle_spacing / 4
        lower = grid_adaptive.origin .+ pad
        upper = lower .+ (grid_adaptive.dimensions .- 1) .* (particle_spacing / 2) .-
                2pad
        mesh_inert,
        stats_inert = reconstruct_surface(points, volumes;
                                          particle_spacing=particle_spacing,
                                          voxel_size=particle_spacing / 2,
                                          min_corner=Tuple(lower),
                                          max_corner=Tuple(upper),
                                          open_faces=ntuple(_ -> true, 6))
        @test mesh_inert.vertices == mesh_adaptive.vertices
        @test mesh_inert.faces == mesh_adaptive.faces
        @test stats_inert["effective_isovalue"] == stats_adaptive["effective_isovalue"]

        # Invalid combinations are rejected
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=particle_spacing,
                                                         tank_size=(1.0, 1.0, 1.0),
                                                         min_corner=(0.0, 0.0, 0.0),
                                                         max_corner=(1.0, 1.0, 1.0))
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=particle_spacing,
                                                         min_corner=(0.0, 0.0, 0.0))
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=particle_spacing,
                                                         min_corner=(1.0, 1.0, 1.0),
                                                         max_corner=(0.0, 0.0, 0.0))
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=particle_spacing,
                                                         min_corner=(0.0, 0.0, 0.0),
                                                         max_corner=(1.0, 1.0, 1.0),
                                                         open_faces=(true, true))
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=particle_spacing,
                                                         open_faces=(false, false,
                                                                     false, true,
                                                                     false, false))
    end

    @testset "workspace reuse requires an identical grid" begin
        reconstruction = SurfaceReconstruction(; particle_spacing=particle_spacing,
                                               voxel_size=particle_spacing / 2)
        reconstruct_surface!(reconstruction, points, volumes)
        workspace = reconstruction.cache.workspace[]
        reconstruct_surface!(reconstruction, points, volumes)
        @test reconstruction.cache.workspace[] === workspace
        # Same dimensions, shifted origin: the marching-cubes coordinates are baked into
        # the workspace, so it must be rebuilt.
        reconstruct_surface!(reconstruction, points .+ particle_spacing / 2, volumes)
        @test reconstruction.cache.workspace[] !== workspace
        @test size(reconstruction.cache.workspace[].field) == size(workspace.field)
    end

    @testset "reused workspace matches a fresh one bitwise" begin
        # A reused workspace is only zeroed where the previous frame wrote. Frames whose
        # particles move, splash, and drop must not see stale values of earlier frames.
        spacing = 0.05
        tank_size = (1.0, 1.0, 1.0)
        function lattice(n_x, n_y, n_z; offset=(0.0, 0.0, 0.0))
            return reduce(hcat,
                          vec([SVector{3, Float64}(offset[1] + (i - 0.5) * spacing,
                                                   offset[2] + (j - 0.5) * spacing,
                                                   offset[3] + (k - 0.5) * spacing)
                               for i in 1:n_x, j in 1:n_y, k in 1:n_z]))
        end
        splash = lattice(3, 3, 3; offset=(0.4, 0.8, 0.4))
        frames = [lattice(12, 12, 12), hcat(lattice(12, 12, 12), splash),
            lattice(12, 3, 12), lattice(12, 6, 12; offset=(0.35, 0.0, 0.2))]
        reconstruction = SurfaceReconstruction(; particle_spacing=spacing, tank_size)
        for frame_points in frames
            frame_volumes = fill(spacing^3, size(frame_points, 2))
            mesh,
            stats = reconstruct_surface!(reconstruction, frame_points, frame_volumes;
                                         initial_isovalue=0.5)
            fresh = SurfaceReconstruction(; particle_spacing=spacing, tank_size)
            fresh_mesh,
            fresh_stats = reconstruct_surface!(fresh, frame_points, frame_volumes;
                                               initial_isovalue=0.5)
            @test mesh.vertices == fresh_mesh.vertices
            @test mesh.faces == fresh_mesh.faces
            @test stats["effective_isovalue"] == fresh_stats["effective_isovalue"]
            @test stats["field_integral"] == fresh_stats["field_integral"]
            @test stats["field_max"] == fresh_stats["field_max"]
        end

        # The domain constraint is kept between frames and only restored where boundary
        # constraints of the previous frame modified it
        block = lattice(12, 12, 12)
        block_volumes = fill(spacing^3, size(block, 2))
        function cube_boundary(lower)
            cube = reduce(hcat,
                          vec([SVector{3, Float64}(lower) +
                               0.2 * SVector{3, Float64}(i, j, k)
                               for i in (0.0, 1.0), j in (0.0, 1.0), k in (0.0, 1.0)]))
            return BoundaryMesh(cube, lattice_surface_topology(cube))
        end
        boundary_frames = [[cube_boundary((0.1, 0.2, 0.2))],
            [cube_boundary((0.35, 0.2, 0.2)), cube_boundary((0.1, 0.45, 0.1))],
            BoundaryMesh[], [cube_boundary((0.3, 0.3, 0.3))]]
        for boundaries in boundary_frames
            mesh,
            stats = reconstruct_surface!(reconstruction, block, block_volumes, boundaries;
                                         initial_isovalue=0.5)
            fresh = SurfaceReconstruction(; particle_spacing=spacing, tank_size)
            fresh_mesh,
            fresh_stats = reconstruct_surface!(fresh, block, block_volumes, boundaries;
                                               initial_isovalue=0.5)
            @test mesh.vertices == fresh_mesh.vertices
            @test mesh.faces == fresh_mesh.faces
            @test stats["n_cavity_regions"] == fresh_stats["n_cavity_regions"]
        end
    end

    @testset "reconstruction caches the liquid analysis for writers" begin
        rec = SurfaceReconstruction(; particle_spacing, voxel_size=particle_spacing / 2)
        mesh, _ = reconstruct_surface!(rec, points, volumes; initial_isovalue=0.5)
        cached = rec.cache.last_mesh_analysis[]
        @test !isnothing(cached) && cached[1] === mesh
        @test cached[2].liquid_component_volumes ==
              TrixiParticles.mesh_liquid_analysis(mesh).liquid_component_volumes
        mktempdir() do dir
            plain = TrixiParticles.write_ply(mesh, joinpath(dir, "plain.ply"))
            fast = TrixiParticles.write_ply(mesh, joinpath(dir, "fast.ply");
                                            analysis=cached[2])
            @test read(plain) == read(fast)
        end
        # A sparse-fallback combine produces a new mesh object: nothing is cached
        rec_sparse = SurfaceReconstruction(; particle_spacing,
                                           voxel_size=particle_spacing / 2,
                                           sparse_component_fallback=true)
        points_sparse = hcat(points, reshape([0.9, 0.9, 0.9], 3, 1))
        volumes_sparse = vcat(volumes, particle_spacing^3)
        _,
        stats_sparse = reconstruct_surface!(rec_sparse, points_sparse, volumes_sparse;
                                            initial_isovalue=0.5)
        @test stats_sparse["n_sparse_fallback_components"] == 1
        @test isnothing(rec_sparse.cache.last_mesh_analysis[])
    end

    @testset "surface mesh interface" begin
        # SurfaceMesh is also the interchange type for concatenating independent
        # fluids and for VTK/PLY output; it must expose its vertex/index types.
        mesh, _ = reconstruct_surface(points, volumes;
                                      particle_spacing=particle_spacing)
        @test mesh isa SurfaceMesh{Float32, Int32}
        @test eltype(mesh) == Float32
        @test ndims(mesh) == 3
        combined = TrixiParticles.combine_surface_meshes([mesh, mesh])
        @test combined isa SurfaceMesh{Float32, Int32}
        @test length(combined.vertices) == 2 * length(mesh.vertices)
        @test length(combined.faces) == 2 * length(mesh.faces)
        @test_throws ArgumentError TrixiParticles.combine_surface_meshes(SurfaceMesh[])
    end

    @testset "vtk polydata output" begin
        mesh, _ = reconstruct_surface(points, volumes;
                                      particle_spacing=particle_spacing)
        output_directory = mktempdir()
        file = trixi2vtk(mesh; output_directory, filename="surf", iter=0, t=0.25)
        @test basename(file) == "surf_0"
        @test sort(readdir(output_directory)) == ["surf.pvd", "surf_0.vtp"]

        # Point data, winding, and normals are read back against an independent
        # nested-shell reference in the correctness suite.
        pvd = read(joinpath(output_directory, "surf.pvd"), String)
        @test occursin("surf_0.vtp", pvd) && occursin("0.25", pvd)

        # Overwrite mode reuses one `_current` file
        trixi2vtk(mesh; output_directory, filename="cur", overwrite=true)
        trixi2vtk(mesh; output_directory, filename="cur", overwrite=true)
        @test count(startswith("cur"), readdir(output_directory)) == 2

        # Empty meshes are skipped like empty systems
        empty_mesh = SurfaceMesh(SVector{3, Float32}[], SVector{3, Int32}[])
        @test trixi2vtk(empty_mesh; output_directory, filename="empty") === nothing
        @test !any(startswith("empty"), readdir(output_directory))
    end

    @testset "statistics" begin
        # Typed access and dictionary access are both public, and field names are
        # consumed by metadata/CSV output rather than just by these direct tests.
        mesh,
        stats = reconstruct_surface(points, volumes;
                                    particle_spacing=particle_spacing)
        @test stats isa SurfaceReconstructionStatistics
        @test stats.volume == stats["volume"]
        @test stats.effective_isovalue == stats["effective_isovalue"]
        @test stats.n_correction_evaluations ==
              length(stats["isovalue_correction_evaluations"])
        @test haskey(stats, "timings")
        @test !haskey(stats, "does-not-exist")
        details = statistics_dict(stats)
        @test details isa Dict{String, Any}
        # TrixiParticles naming: no unit suffixes, no domain-specific names
        for key in keys(details)
            @test !occursin(r"^water_|_m[234]?$|_seconds$|blade", key)
        end
        # Direct calls without a configured backend use the default CPU backend
        reconstruction = SurfaceReconstruction(; particle_spacing=particle_spacing)
        reconstruct_surface!(reconstruction, points, volumes)
        @test reconstruction.cache.workspace[].backend isa PolyesterBackend
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=0.1,
                                                         parallelization_backend=TrixiParticles.KernelAbstractions.CPU())
        @test Set(keys(stats)) == Set(keys(details))
        @test all(stats[key] == details[key]
                  for key in keys(details)
                  if !(details[key] isa AbstractFloat && isnan(details[key])))
        @test occursin("SurfaceReconstructionStatistics", repr(stats))
        @test occursin("effective isovalue", repr(MIME"text/plain"(), stats))
    end

    @testset "boundaries away from the grid" begin
        # Outside the grid along two axes; such boxes used to be traversed with negative
        # extents, writing out of bounds
        cube = reduce(hcat,
                      vec([TrixiParticles.SVector{3, Float64}(5.0, 5.0, 0.2) +
                           0.3 * TrixiParticles.SVector{3, Float64}(i, j, k)
                           for i in (0.0, 1.0), j in (0.0, 1.0), k in (0.0, 1.0)]))
        far_boundary = BoundaryMesh(cube, lattice_surface_topology(cube))
        mesh, stats = reconstruct_surface(points, volumes, [far_boundary];
                                          particle_spacing)
        mesh_free, _ = reconstruct_surface(points, volumes; particle_spacing)
        @test stats["boundary_sample_grid_points"] == 0
        @test mesh.vertices == mesh_free.vertices
        @test mesh.faces == mesh_free.faces
    end

    @testset "enclosed fluid" begin
        cube = reduce(hcat,
                      vec([TrixiParticles.SVector{3, Float64}(0.2, 0.2, 0.2) +
                           0.3 * TrixiParticles.SVector{3, Float64}(i, j, k)
                           for i in (0.0, 1.0), j in (0.0, 1.0), k in (0.0, 1.0)]))
        boundary = BoundaryMesh(cube, lattice_surface_topology(cube))
        enclosed = enclosed_particles(points, [boundary])
        retained = findall(iszero, enclosed)

        # Default: particles inside boundaries are excluded, bitwise identical to removing
        # them manually (the production workflow)
        mesh_auto,
        stats_auto = reconstruct_surface(points, volumes, [boundary];
                                         particle_spacing)
        mesh_manual,
        stats_manual = reconstruct_surface(points[:, retained],
                                           volumes[retained], [boundary];
                                           particle_spacing)
        @test stats_auto["n_enclosed_particles"] == 8
        @test stats_auto["enclosed_particle_volume"] ≈ 8 * particle_spacing^3
        @test stats_manual["n_enclosed_particles"] == 0
        @test mesh_auto.vertices == mesh_manual.vertices
        @test mesh_auto.faces == mesh_manual.faces
        @test stats_auto["volume"] == stats_manual["volume"]

        # `keep_enclosed_fluid=true` targets the full particle volume instead
        _,
        stats_keep = reconstruct_surface(points, volumes, [boundary];
                                         particle_spacing, keep_enclosed_fluid=true)
        @test stats_keep["n_enclosed_particles"] == 0
        @test stats_keep["particle_volume"] ≈ sum(volumes)
        @test stats_auto["particle_volume"] ≈ sum(volumes[retained])
    end

    @testset "triangle mesh bridge" begin
        mesh, _ = reconstruct_surface(points, volumes;
                                      particle_spacing=particle_spacing)

        # Reconstructed surfaces convert to preprocessing TriangleMesh geometry
        geometry = TrixiParticles.TriangleMesh(mesh)
        @test geometry isa TrixiParticles.TriangleMesh{3, Float64}
        @test length(geometry.vertices) == length(mesh.vertices)
        @test length(geometry.face_vertices_ids) == length(mesh.faces)

        # A TriangleMesh builds an boundary mesh with identical signed distances
        lattice = reduce(hcat,
                         vec([TrixiParticles.SVector{3, Float64}(0.2, 0.2, 0.2) +
                              0.3 * TrixiParticles.SVector{3, Float64}(i, j, k)
                              for i in (0.0, 1.0), j in (0.0, 1.0), k in (0.0, 1.0)]))
        lattice_topology = TrixiParticles.lattice_surface_topology(lattice)
        lattice_obstacle = TrixiParticles.BoundaryMesh(lattice, lattice_topology)
        face_vertices = [(SVector{3, Float64}(lattice[:, face[1]]),
                          SVector{3, Float64}(lattice[:, face[2]]),
                          SVector{3, Float64}(lattice[:, face[3]]))
                         for face in lattice_topology.faces]
        face_normals = map(face_vertices) do (a, b, c)
            n = cross(b - a, c - a)
            n / norm(n)
        end
        vertices = [SVector{3, Float64}(lattice[:, index])
                    for index in axes(lattice, 2)]
        triangle_geometry = TrixiParticles.TriangleMesh(face_vertices, face_normals,
                                                        vertices)
        geometry_obstacle = TrixiParticles.BoundaryMesh(triangle_geometry)
        @test geometry_obstacle.lower == lattice_obstacle.lower
        @test geometry_obstacle.upper == lattice_obstacle.upper
        # Same enclosed particles and matching signed distances on the fluid lattice
        @test TrixiParticles.enclosed_particles(points,
                                                [geometry_obstacle]) ==
              TrixiParticles.enclosed_particles(points, [lattice_obstacle])
        probe = SVector(0.35, 0.5, 0.9)
        @test TrixiParticles.signed_distance(geometry_obstacle.bvh, probe) ≈
              TrixiParticles.signed_distance(lattice_obstacle.bvh, probe) rtol=1.0e-12
    end

    @testset "invalid options" begin
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=-0.1)
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=0.1,
                                                         minimum_isovalue=0.5,
                                                         maximum_isovalue=0.2)
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=0.1,
                                                         isovalue=0.95)
        @test_throws ArgumentError SurfaceReconstruction(particle_spacing=0.1,
                                                         gaussian_sigma=0.05,
                                                         gaussian_sigma_voxels=2.0)
    end

    @testset "particles beyond the nominal grid" begin
        voxel_size = particle_spacing / 2

        # A splash above an open-top tank: the grid extends over the open face, and the
        # splash is reconstructed instead of being absorbed by the block surface.
        splash = reduce(hcat,
                        vec([SVector{3, Float64}(0.5 + x, 2.0 + y, 0.5 + z)
                             for z in 0:0.1:0.2, y in 0:0.1:0.2, x in 0:0.1:0.2]))
        points_splash = hcat(points, splash)
        volumes_splash = fill(particle_spacing^3, size(points_splash, 2))
        mesh,
        stats = reconstruct_surface(points_splash, volumes_splash;
                                    particle_spacing, voxel_size,
                                    tank_size=(1.0, 1.0, 1.0))
        @test stats["n_excluded_particles"] == 0
        @test stats["deposited_volume"] ≈
              stats["particle_volume"] rtol=1.0e-6
        @test stats["n_connected_regions"] == 2
        @test maximum(vertex[2] for vertex in mesh.vertices) > 2.0
        @test abs(100 * (stats["volume"] - sum(volumes_splash)) /
                  sum(volumes_splash)) <= 0.1

        # Particles that cannot be represented (below a closed floor beyond the padding,
        # or beyond the extension cap of an open face) are excluded consistently from the
        # deposit and the correction target, with a warning.
        for stray_point in ([0.3, -2.0, 0.3], [0.3, 100.0, 0.3])
            stray = reshape(stray_point, 3, 1)
            _,
            stats_stray = @test_logs (:warn, r"outside the reconstruction grid") match_mode=:any reconstruct_surface(hcat(points,
                                                                                                                          stray),
                                                                                                                     vcat(volumes,
                                                                                                                          particle_spacing^3);
                                                                                                                     particle_spacing,
                                                                                                                     voxel_size,
                                                                                                                     tank_size=(1.0,
                                                                                                                                1.0,
                                                                                                                                1.0))
            @test stats_stray["n_excluded_particles"] == 1
            @test stats_stray["excluded_particle_volume"] ≈ particle_spacing^3
        end
        # The extension cap bounds the grid to twice the nominal extent per axis
        _,
        stats_capped = reconstruct_surface(hcat(points, reshape([0.3, 100.0, 0.3], 3, 1)),
                                           vcat(volumes, particle_spacing^3);
                                           particle_spacing, voxel_size,
                                           tank_size=(1.0, 1.0, 1.0))
        _,
        stats_nominal = reconstruct_surface(points, volumes; particle_spacing,
                                            voxel_size, tank_size=(1.0, 1.0, 1.0))
        @test stats_capped["grid_dimensions"][2] <= 2 * stats_nominal["grid_dimensions"][2]
    end

    @testset "Float32 systems" begin
        # GPU simulations commonly store Float32 states; this CPU path checks that the
        # public system entry accepts that precision before testing device transfer.
        spacing_32 = 0.05f0
        tank_32 = RectangularTank(spacing_32, (0.25f0, 0.25f0, 0.25f0),
                                  (0.5f0, 0.6f0, 0.5f0), 1000.0f0; n_layers=3)
        fluid_32 = WeaklyCompressibleSPHSystem(tank_32.fluid;
                                               smoothing_kernel=WendlandC2Kernel{3}(),
                                               smoothing_length=1.5f0 * spacing_32,
                                               density_calculator=SummationDensity(),
                                               state_equation=StateEquationCole(;
                                                                                sound_speed=10.0f0,
                                                                                reference_density=1000.0f0,
                                                                                exponent=7))
        semi_32 = Semidiscretization(fluid_32)
        ode_32 = semidiscretize(semi_32, (0.0, 0.001))
        v_32, u_32 = ode_32.u0.x
        TrixiParticles.update_systems_and_nhs(v_32, u_32, semi_32, 0.0)
        mesh_32, stats_32 = reconstruct_surface(fluid_32, v_32, u_32, semi_32)
        @test eltype(mesh_32) == Float32
        @test stats_32.particle_volume > 0
    end
end
