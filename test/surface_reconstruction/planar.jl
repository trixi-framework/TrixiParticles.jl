@trixi_testset "planar surface reconstruction" begin
    using OrdinaryDiffEqLowStorageRK
    TP = TrixiParticles

    function lattice(n_x, n_y, spacing; offset=(0.0, 0.0))
        return reduce(hcat,
                      vec([SVector(offset[1] + (i - 0.5) * spacing,
                                   offset[2] + (j - 0.5) * spacing)
                           for i in 1:n_x, j in 1:n_y]))
    end
    function square(lower, upper)
        vertices = [SVector(lower, lower), SVector(upper, lower),
            SVector(upper, upper), SVector(lower, upper)]
        return SurfaceMesh(vertices,
                           TP.ContourSegment[TP.ContourSegment(1, 2),
                                             TP.ContourSegment(2, 3),
                                             TP.ContourSegment(3, 4),
                                             TP.ContourSegment(4, 1)])
    end
    # Independent shoelace integral over the directed edges returned/written by the API.
    area(vertices,
         edges) = sum(BigFloat(vertices[e[1]][1]) * BigFloat(vertices[e[2]][2]) -
                      BigFloat(vertices[e[2]][1]) * BigFloat(vertices[e[1]][2])
                      for e in edges) / 2

    @testset "bilinear CIC and separable Gaussian" begin
        origin, spacing = SVector(-0.2, 0.1), 0.05
        field = zeros(Float32, 21, 21)
        point = reshape([-0.037, 0.394], 2, 1)
        deposited = TP.deposit_volume_cic!(field, point, [0.0123], origin, spacing)
        moments = TP.scalar_field_moments(field, origin, spacing)
        @test count(!iszero, field) == 4
        @test deposited.deposited_volume ≈ 0.0123 rtol=4eps(Float32)
        @test moments.centroid ≈ point[:, 1] rtol=4eps(Float32)

        kernel = TP.gaussian_kernel(0.7)
        fill!(field, 0)
        field[11, 11] = 1
        TP.gaussian_filter!(field, similar(field), similar(field), kernel)
        expected = zeros(Float64, size(field))
        radius = length(kernel) ÷ 2
        for j in eachindex(kernel), i in eachindex(kernel)
            expected[11 + i - radius - 1,
                     11 + j - radius - 1] = Float64(kernel[i]) * kernel[j]
        end
        @test field ≈ expected rtol=4eps(Float32)
        @test sum(Float64, field) ≈ 1 rtol=4eps(Float32)
    end

    @testset "nested-loop area and invalid geometry" begin
        nested = TP.combine_surface_meshes([
                                               square(-2.0, 2.0),
                                               square(-1.0, 1.0),
                                               square(-0.3, 0.3)
                                           ])
        analysis = TP.contour_analysis(nested)
        geometry = TP.mesh_geometry_stats(nested; analysis)
        @test geometry.volume ≈ 16 - 4 + 0.36
        @test geometry.surface_area ≈ 16 + 8 + 2.4
        @test geometry.n_connected_regions == 2
        @test geometry.n_cavity_regions == 1
        @test ndims(SurfaceMesh{Float32, Int32}(square(0.0, 1.0).vertices,
                                                square(0.0, 1.0).faces)) == 2
        @test Float64(area(nested.vertices, analysis.oriented_faces)) ≈ geometry.volume
        oriented = SurfaceMesh(nested.vertices, analysis.oriented_faces)
        @test TP.contour_analysis(oriented).oriented_faces == oriented.faces
        mktempdir() do dir
            # Reusing the analysis of an oriented hole must preserve edge ordering in
            # serialized output, not just the geometric set of line segments.
            write_ply(oriented, joinpath(dir, "cached.ply"); analysis)
            write_ply(oriented, joinpath(dir, "fresh.ply"))
            @test read(joinpath(dir, "cached.ply")) == read(joinpath(dir, "fresh.ply"))
        end
        boundary = BoundaryMesh(nested)
        @test TP.signed_distance(boundary.bvh, SVector(1.5, 0.0)) ≈ -0.5
        @test TP.signed_distance(boundary.bvh, SVector(0.5, 0.0)) ≈ 0.2
        @test TP.signed_distance(boundary.bvh, SVector(0.0, 0.0)) ≈ -0.3
        @test TP.signed_distance(boundary.bvh, SVector(3.0, 3.0)) ≈ sqrt(2)
        normals, _ = TP.surface_vertex_normals(nested; analysis)
        @test all(dot(normals[i], nested.vertices[i]) > 0 for i in 1:4)
        @test all(dot(normals[i], nested.vertices[i]) < 0 for i in 5:8)
        @test_throws ArgumentError TP.contour_analysis(SurfaceMesh(nested.vertices,
                                                                   nested.faces[2:end]))
        crossing = SurfaceMesh([
                                   SVector(0.0, 0.0),
                                   SVector(1.0, 1.0),
                                   SVector(0.0, 1.0),
                                   SVector(1.0, 0.0)
                               ],
                               square(0.0, 1.0).faces)
        @test_throws ArgumentError TP.contour_analysis(crossing)
    end

    @testset "asymptotic decider and scale-independent saddle topology" begin
        # xy+mu > 0 inside a square has one liquid component for mu>0 and two for
        # mu<0. At this resolution the checkerboard cell needs an ambiguity decision.
        R, n = 1.5, 12
        coords = range(-2R, 2R; length=n)
        for mu in (-0.009, 0.009), amplitude in (1.0f-4, 1.0f0, 1000.0f0)
            field = amplitude .* Float32[min(x * y + mu, R - abs(x), R - abs(y))
                            for x in coords, y in coords]
            mesh = TP.marching_squares(field, SVector(-2R, -2R), step(coords))
            geometry = TP.mesh_geometry_stats(mesh)
            @test geometry.n_connected_regions == (mu > 0 ? 1 : 2)
            @test geometry.n_cavity_regions == 0
        end
        # All 16 cases, padded by negative samples, must assemble closed degree-two loops.
        for case in 0:15
            field = fill(-1.0f0, 6, 6)
            for (bit, index) in enumerate(((3, 3), (4, 3), (4, 4), (3, 4)))
                field[index...] = (case & (1 << (bit - 1))) == 0 ? -0.7f0 : 0.9f0
            end
            mesh = TP.marching_squares(field, SVector(0.0, 0.0), 0.1)
            if case == 0
                @test isempty(mesh.faces)
            else
                @test TP.mesh_geometry_stats(mesh).volume > 0
            end
        end
        # A sample exactly at the contour level needs one shared tie convention.
        # Rescaling the field must not turn that tie into a large geometric displacement.
        field = fill(-1.0f0, 6, 6)
        field[3, 3], field[3, 4] = 0, 1
        reference = TP.marching_squares(field, SVector(0.0, 0.0), 0.1)
        for scale in (1.0f-4, 1000.0f0)
            scaled = TP.marching_squares(scale .* field, SVector(0.0, 0.0), 0.1)
            @test scaled.faces == reference.faces
            @test scaled.vertices≈reference.vertices rtol=0 atol=1.0e-12
        end
    end

    @testset "area correction, dimensions and workspace reuse" begin
        spacing = 0.05
        points = lattice(12, 9, spacing)
        volumes = fill(spacing^2, size(points, 2))
        rec = SurfaceReconstruction(; particle_spacing=spacing, tank_size=(1.0, 1.0))
        mesh, stats = reconstruct_surface!(rec, points, volumes)
        @test ndims(rec) == ndims(mesh) == 2
        @test length(rec.open_faces) == 4 && rec.open_faces[4]
        @test stats["area"] ≈ sum(volumes) rtol=0.001
        @test stats["perimeter"] == stats.surface_area
        @test area(mesh.vertices, mesh.faces) ≈ stats.volume rtol=1.0e-12
        @test stats.grid_dimensions == collect(size(rec.cache.workspace[].field))
        workspace = rec.cache.workspace[]
        for shape in (lattice(12, 4, spacing), lattice(8, 10, spacing; offset=(0.4, 0.1)))
            areas = fill(spacing^2, size(shape, 2))
            reused,
            reused_stats = reconstruct_surface!(rec, shape, areas; initial_isovalue=0.5)
            fresh,
            fresh_stats = reconstruct_surface(shape, areas; particle_spacing=spacing,
                                              tank_size=(1.0, 1.0))
            @test rec.cache.workspace[] === workspace
            @test reused.vertices == fresh.vertices && reused.faces == fresh.faces
            @test reused_stats.volume == fresh_stats.volume
        end
        shifted,
        shifted_stats = reconstruct_surface(points .+ [1.7, -2.3], volumes;
                                            particle_spacing=spacing)
        original,
        original_stats = reconstruct_surface(points, volumes; particle_spacing=spacing)
        @test shifted_stats.volume ≈ original_stats.volume rtol=1.0e-7
        @test length(shifted.faces) == length(original.faces)
        @test all(isapprox(a, b + SVector(1.7, -2.3); atol=1.0e-7)
                  for (a, b) in zip(shifted.vertices, original.vertices))
        serial,
        serial_stats = reconstruct_surface(points, volumes; particle_spacing=spacing,
                                           parallelization_backend=SerialBackend())
        @test serial.vertices == original.vertices && serial.faces == original.faces
        @test serial_stats.volume == original_stats.volume
        @test_throws ArgumentError reconstruct_surface!(SurfaceReconstruction(;
                                                                              particle_spacing=spacing),
                                                        points, volumes)
        @test_throws ArgumentError SurfaceReconstruction(; particle_spacing=spacing,
                                                         ndims=1)
        @test_throws ArgumentError SurfaceReconstruction(; particle_spacing=spacing,
                                                         ndims=2, tank_size=(1.0, 1.0, 1.0))
    end

    @testset "polygon constraints, holes, open faces and sparse circles" begin
        spacing = 0.05
        points = lattice(18, 18, spacing)
        volumes = fill(spacing^2, size(points, 2))
        polygon = TP.Polygon([0.3 0.6 0.6 0.3; 0.3 0.3 0.6 0.6])
        boundary = BoundaryMesh(polygon)
        mesh,
        stats = reconstruct_surface(points, volumes, [boundary]; particle_spacing=spacing,
                                    tank_size=(1.0, 1.0), record_field_moments=true)
        @test stats.n_cavity_regions == 1
        @test stats["n_enclosed_particles"] == 36
        @test stats.volume ≈ sum(volumes) - 36spacing^2 rtol=0.001
        @test stats.n_vertices_inside_boundaries == 0
        @test length(stats["field_deposition_stats"]["field_moment_history"]) == 3
        reference = lattice(4, 4, spacing; offset=(0.3, 0.3))
        moving = BoundaryMesh(reference, lattice_surface_topology(reference))
        @test length(moving.faces) == 12
        @test TP.signed_distance(moving.bvh, SVector(0.4, 0.4)) < 0
        for shift in (0.0, 0.2)
            tracked = BoundaryMesh(reference .+ [shift, 0.0],
                                   lattice_surface_topology(reference))
            @test TP.signed_distance(tracked.bvh, SVector(0.4 + shift, 0.4)) < 0
        end
        # A splash above the open +y face must extend the grid instead of being lost.
        splash = lattice(5, 5, spacing; offset=(0.2, 1.2))
        all_points = hcat(points, splash)
        _,
        above = reconstruct_surface(all_points, fill(spacing^2, size(all_points, 2));
                                    particle_spacing=spacing, tank_size=(1.0, 1.0))
        @test above["n_excluded_particles"] == 0 && above.n_connected_regions == 2
        small = lattice(7, 7, spacing)
        sparse = hcat(small, [0.8, 0.8])
        droplet,
        sparse_stats = reconstruct_surface(sparse, fill(spacing^2, size(sparse, 2));
                                           particle_spacing=spacing,
                                           sparse_component_fallback=true)
        @test sparse_stats["n_sparse_fallback_components"] == 1
        @test sparse_stats.n_connected_regions == 2
        @test sparse_stats.volume ≈ size(sparse, 2) * spacing^2 rtol=0.001
        @test Float64(area(droplet.vertices, droplet.faces)) ≈ sparse_stats.volume rtol=1.0e-12
    end

    @testset "2D callback, interpolation, offline reconstruction and line output" begin
        spacing = 0.05
        shape = RectangularShape(spacing, (7, 6), (0.1, 0.1); density=1000.0)
        fluid = WeaklyCompressibleSPHSystem(shape; smoothing_kernel=WendlandC2Kernel{2}(),
                                            smoothing_length=1.5spacing,
                                            density_calculator=SummationDensity(),
                                            state_equation=StateEquationCole(;
                                                                             sound_speed=10.0,
                                                                             reference_density=1000.0,
                                                                             exponent=7))
        semi = Semidiscretization(fluid)
        rec = SurfaceReconstruction(; particle_spacing=spacing, ndims=2)
        mktempdir() do dir
            callback = SurfaceReconstructionCallback(rec, semi; interval=1,
                                                     output_directory=dir,
                                                     formats=(:vtp, :ply),
                                                     interpolated_quantities=(:density,))
            sol = solve(semidiscretize(semi, (0.0, 0.002)), RDPK3SpFSAL35(); dt=0.001,
                        adaptive=false, save_everystep=true, callback=CallbackSet(callback))
            affect = callback.affect!
            @test ndims(affect.latest_mesh) == 2
            @test length(affect.statistics_times) == 3
            @test haskey(affect.statistics_data, "area_fluid_1")
            @test haskey(affect.statistics_data, "perimeter_fluid_1")
            file = joinpath(dir, "surface_fluid_1_0.vtp")
            vtk = TP.ReadVTK.VTKFile(file)
            coordinates = TP.ReadVTK.get_points(vtk)
            lines = TP.ReadVTK.get_primitives(vtk, "Lines")
            @test all(iszero, coordinates[3, :])
            @test all(==(2), diff(vcat(0, lines.offsets)))
            normals = TP.ReadVTK.get_data(TP.ReadVTK.get_point_data(vtk)["Normals"])
            @test all(iszero, normals[3, :])
            @test all(isapprox(norm(normal), 1; atol=1.0e-12)
                      for normal in eachcol(normals))
            values = TP.ReadVTK.get_data(TP.ReadVTK.get_point_data(vtk)["density"])
            @test any(isfinite, values)
            offline, stats = reconstruct_surface(semi, sol; frame=1)
            @test ndims(offline) == 2 && stats.volume > 0
            initial, initial_stats = reconstruct_surface(shape)
            @test ndims(initial) == 2
            @test initial_stats.particle_volume ≈ sum(shape.mass ./ shape.density)
            # Decode PLY edge records independently: planar export must not invent
            # triangle faces or nonzero z coordinates.
            ply = write_ply(initial, joinpath(dir, "initial.ply"))
            open(ply) do io
                header = String[]
                while true
                    line = readline(io)
                    push!(header, line)
                    line == "end_header" && break
                end
                @test "element edge $(length(initial.faces))" in header
                vertices = read!(io, Matrix{Float32}(undef, 6, length(initial.vertices)))
                edges = read!(io, Matrix{Int32}(undef, 2, length(initial.faces))) .+ 1
                @test all(iszero, vertices[3, :]) && eof(io)
                @test Float64(area(SVector{2, Float64}.(eachcol(vertices[1:2, :])),
                                   TP.ContourSegment.(eachcol(edges)))) ≈
                      initial_stats.volume rtol=1.0e-6
            end
        end
    end

    @testset "tracked and static planar boundaries in a callback" begin
        spacing = 0.05
        kernel = WendlandC2Kernel{2}()
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=7)
        fluid_shape = RectangularShape(spacing, (10, 9), (0.1, 0.1); density=1000.0)
        solid_shape = RectangularShape(spacing, (4, 4), (0.25, 0.25); density=1000.0)
        fluid = WeaklyCompressibleSPHSystem(fluid_shape; smoothing_kernel=kernel,
                                            smoothing_length=1.5spacing, state_equation,
                                            density_calculator=SummationDensity())
        model = BoundaryModelDummyParticles(solid_shape.density, solid_shape.mass,
                                            AdamiPressureExtrapolation(), kernel,
                                            1.5spacing; state_equation)
        solid = WallBoundarySystem(solid_shape, model)
        semi = Semidiscretization(fluid, solid)
        rec = SurfaceReconstruction(; particle_spacing=spacing, ndims=2)
        mktempdir() do dir
            callback = SurfaceReconstructionCallback(rec, semi; boundaries=solid,
                                                     output_directory=dir,
                                                     write_boundaries=true,
                                                     save_final_surface=false)
            solve(semidiscretize(semi, (0.0, 0.001)), RDPK3SpFSAL35(); dt=0.001,
                  adaptive=false, callback=CallbackSet(callback))
            @test callback.affect!.latest_stats.n_cavity_regions == 1
            @test callback.affect!.boundary_topologies[2] isa BoundaryTopology{2}
            vtk = TP.ReadVTK.VTKFile(joinpath(dir, "surface_boundary_1_0.vtp"))
            @test length(TP.ReadVTK.get_primitives(vtk, "Lines").offsets) == 12
        end
        static = TP.Polygon([0.25 0.4 0.4 0.25; 0.25 0.25 0.4 0.4])
        callback = SurfaceReconstructionCallback(rec, semi; boundaries=static, formats=(),
                                                 output_directory=mktempdir(),
                                                 save_final_surface=false)
        solve(semidiscretize(semi, (0.0, 0.001)), RDPK3SpFSAL35(); dt=0.001,
              adaptive=false, callback=CallbackSet(callback))
        @test length(callback.affect!.static_boundaries) == 1
        @test callback.affect!.latest_stats.n_cavity_regions == 1
    end
end
