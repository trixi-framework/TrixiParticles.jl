@trixi_testset "SurfaceReconstructionCallback" begin
    using OrdinaryDiffEqLowStorageRK

    # Drive real short ODE solves rather than invoking `affect!` on a mock integrator:
    # neighborhood-search and density caches must describe each saved frame.
    function make_test_semi(; particle_spacing=0.05, tank_size=(0.5, 0.6, 0.5),
                            fluid_size=(0.25, 0.25, 0.25),
                            parallelization_backend=PolyesterBackend())
        smoothing_length = 1.5 * particle_spacing

        tank = RectangularTank(particle_spacing, fluid_size, tank_size, 1000.0;
                               n_layers=3, acceleration=(0.0, -9.81, 0.0))
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0, exponent=7)
        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid;
                                                   smoothing_kernel=WendlandC2Kernel{3}(),
                                                   smoothing_length,
                                                   density_calculator=SummationDensity(),
                                                   state_equation)
        boundary_system = WallBoundarySystem(tank.boundary,
                                             BoundaryModelDummyParticles(tank.boundary.density,
                                                                         tank.boundary.mass,
                                                                         AdamiPressureExtrapolation(),
                                                                         WendlandC2Kernel{3}(),
                                                                         smoothing_length;
                                                                         state_equation))
        semi = Semidiscretization(fluid_system, boundary_system;
                                  parallelization_backend)

        return semi, particle_spacing, tank_size
    end

    function run_surface_reconstruction_test(semi, callback; tspan=(0.0, 0.02),
                                             dt=0.001)
        ode = semidiscretize(semi, tspan)

        return solve(ode, RDPK3SpFSAL35(); dt, adaptive=false,
                     save_everystep=false, callback)
    end

    @testset "show" begin
        semi, _, _ = make_test_semi()
        callback = SurfaceReconstructionCallback(SurfaceReconstruction(particle_spacing=0.05),
                                                 semi, interval=10)
        @test repr(callback) == "SurfaceReconstructionCallback(interval=10)"

        callback_dt = SurfaceReconstructionCallback(SurfaceReconstruction(particle_spacing=0.05),
                                                    semi, dt=0.01)
        @test repr(callback_dt) == "SurfaceReconstructionCallback(dt=0.01)"

        callback_times = SurfaceReconstructionCallback(SurfaceReconstruction(particle_spacing=0.05),
                                                       semi, save_times=[0.01])
        @test repr(callback_times) == "SurfaceReconstructionCallback(save_times=[0.01])"

        @test_nowarn show(devnull, MIME"text/plain"(), callback)
        @test_nowarn show(devnull, MIME"text/plain"(), callback_dt)
    end

    @testset "construction" begin
        semi, _, _ = make_test_semi()
        reconstruction = SurfaceReconstruction(; particle_spacing=0.05,
                                               tank_size=(0.5, 0.6, 0.5))

        # Unknown system index and non-fluid system are rejected
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 fluid_systems=5)
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 fluid_systems=2)
        # Spacing mismatch warns
        @test_warn "does not match" SurfaceReconstructionCallback(SurfaceReconstruction(particle_spacing=0.06;
                                                                                        tank_size=(0.5,
                                                                                                   0.6,
                                                                                                   0.5)),
                                                                  semi)
        # Unknown output format is rejected
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 formats=(:obj,))
        # interval/dt/save_times are mutually exclusive
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 interval=1, dt=0.1)
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 dt=0.1,
                                                                 save_times=[0.01])
    end

    @testset "interval output" begin
        # Initial, tenth accepted step, and final events must each update the time
        # collection and the callback's latest mesh/statistics aliases.
        semi, _, _ = make_test_semi()
        output_directory = mktempdir()
        reconstruction = SurfaceReconstruction(; particle_spacing=0.05,
                                               tank_size=(0.5, 0.6, 0.5))
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=10,
                                                 output_directory,
                                                 save_initial_surface=true)

        sol = run_surface_reconstruction_test(semi, CallbackSet(callback))

        files = readdir(output_directory)
        vtp_files = filter(endswith(".vtp"), files)
        pvd_files = filter(endswith(".pvd"), files)
        @test length(vtp_files) == 3
        @test length(pvd_files) == 1
        @test all(startswith("surface_fluid_1_"), vtp_files)
        @test pvd_files == ["surface_fluid_1.pvd"]

        surface_callback = callback.affect!
        @test surface_callback.latest_mesh isa SurfaceMesh
        @test surface_callback.latest_mesh === surface_callback.latest_meshes[1]
        @test length(surface_callback.latest_meshes) == 1
        @test surface_callback.latest_statistics[1] === surface_callback.latest_stats
        @test surface_callback.reconstruction.cache.workspace[] !== nothing
    end

    @testset "dt output and fluid system selection" begin
        semi, _, _ = make_test_semi()
        output_directory = mktempdir()
        reconstruction = SurfaceReconstruction(; particle_spacing=0.05,
                                               tank_size=(0.5, 0.6, 0.5))
        callback = SurfaceReconstructionCallback(reconstruction, semi; dt=0.01,
                                                 output_directory,
                                                 fluid_systems=1,
                                                 formats=(:vtp, :ply),
                                                 save_initial_surface=true,
                                                 save_final_surface=true)

        sol = run_surface_reconstruction_test(semi, CallbackSet(callback))
        files = readdir(output_directory)
        # Initial surface, two periodic events, and the final surface, in both formats
        vtp_files = filter(endswith(".vtp"), files)
        ply_files = filter(endswith(".ply"), files)
        @test length(vtp_files) == length(ply_files) >= 3
        @test count(endswith(".pvd"), files) == 1

        surface_callback = callback.affect!.affect!
        @test surface_callback.fluid_indices == [1]
        @test surface_callback.latest_mesh isa SurfaceMesh
    end

    @testset "save_times output" begin
        semi, _, _ = make_test_semi()
        output_directory = mktempdir()
        reconstruction = SurfaceReconstruction(; particle_spacing=0.05,
                                               tank_size=(0.5, 0.6, 0.5))
        callback = SurfaceReconstructionCallback(reconstruction, semi;
                                                 save_times=[0.01],
                                                 output_directory,
                                                 save_initial_surface=false,
                                                 save_final_surface=true)

        sol = run_surface_reconstruction_test(semi, CallbackSet(callback))
        files = readdir(output_directory)
        # One event at t=0.01 plus the final surface at t=0.02
        @test count(endswith(".vtp"), files) == 2
        @test count(endswith(".pvd"), files) == 1
    end

    @testset "multi-fluid output" begin
        # Two overlapping fluids: physics is irrelevant here, only the per-system
        # output structure (files, collections, stored meshes) is tested.
        semi, particle_spacing, tank_size = make_test_semi()
        fluid_2 = WeaklyCompressibleSPHSystem(semi.systems[1].initial_condition;
                                              smoothing_kernel=WendlandC2Kernel{3}(),
                                              smoothing_length=1.5 * particle_spacing,
                                              density_calculator=SummationDensity(),
                                              state_equation=semi.systems[1].state_equation)
        semi_multi = Semidiscretization(semi.systems[1], fluid_2, semi.systems[2])

        output_directory = mktempdir()
        reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
        callback = SurfaceReconstructionCallback(reconstruction, semi_multi; interval=1,
                                                 output_directory,
                                                 save_initial_surface=true)

        sol = run_surface_reconstruction_test(semi_multi, CallbackSet(callback);
                                              tspan=(0.0, 0.002))
        files = readdir(output_directory)
        @test count(f -> startswith(f, "surface_fluid_1_") && endswith(f, ".vtp"),
                    files) == 3
        @test count(f -> startswith(f, "surface_fluid_2_") && endswith(f, ".vtp"),
                    files) == 3
        @test sort(filter(endswith(".pvd"), files)) ==
              ["surface_fluid_1.pvd", "surface_fluid_2.pvd"]

        surface_callback = callback.affect!
        @test length(surface_callback.latest_meshes) == 2
        @test all(mesh isa SurfaceMesh for mesh in surface_callback.latest_meshes)
        @test surface_callback.latest_mesh === surface_callback.latest_meshes[1]

        # Independent warm starts per fluid system, shared workspace buffers
        reconstruction_1, reconstruction_2 = surface_callback.reconstructions
        @test reconstruction_1 === surface_callback.reconstruction
        @test reconstruction_1.cache.workspace === reconstruction_2.cache.workspace
        @test reconstruction_1.cache.previous_isovalue !==
              reconstruction_2.cache.previous_isovalue
        @test reconstruction_1.cache.previous_isovalue[] ==
              surface_callback.latest_statistics[1]["effective_isovalue"]
        @test reconstruction_2.cache.previous_isovalue[] ==
              surface_callback.latest_statistics[2]["effective_isovalue"]
    end

    @testset "boundaries clip the surface" begin
        semi, particle_spacing, tank_size = make_test_semi()
        fluid_system, wall_system = semi.systems
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=7)

        # A wall cube overlapping the upper corner of the fluid block
        cube = RectangularShape(particle_spacing, (3, 3, 3), (0.15, 0.1, 0.15),
                                density=1000.0)
        cube_model = BoundaryModelDummyParticles(cube.density, cube.mass,
                                                 AdamiPressureExtrapolation(),
                                                 WendlandC2Kernel{3}(),
                                                 1.5 * particle_spacing; state_equation)
        cube_system = WallBoundarySystem(cube, cube_model)
        semi_cube = Semidiscretization(fluid_system, wall_system, cube_system)

        # Fluid systems cannot act as boundaries
        reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi_cube;
                                                                 boundaries=1)

        callback = SurfaceReconstructionCallback(reconstruction, semi_cube; boundaries=3,
                                                 interval=1, output_directory=mktempdir())
        sol = run_surface_reconstruction_test(semi_cube, CallbackSet(callback);
                                              tspan=(0.0, 0.002))
        surface_callback = callback.affect!
        @test surface_callback.boundary_indices == [3]
        @test haskey(surface_callback.boundary_topologies, 3)
        @test surface_callback.latest_stats["boundary_sample_grid_points"] > 0

        # The clipped surface stays outside the cube up to marching-cubes resolution,
        # while the same frame without the boundary penetrates it.
        cube_mesh = BoundaryMesh(cube.coordinates,
                                 lattice_surface_topology(cube.coordinates))
        penetration(mesh) = -minimum(TrixiParticles.signed_distance(cube_mesh.bvh,
                                                                    SVector{3, Float64}(vertex))
                                     for vertex in mesh.vertices)
        voxel_size = reconstruction.voxel_size
        @test penetration(surface_callback.latest_mesh) <= voxel_size
        mesh_free, _ = reconstruct_surface(semi_cube, sol; tank_size)
        @test penetration(mesh_free) > voxel_size
    end

    @testset "statistics and metadata" begin
        semi, particle_spacing, tank_size = make_test_semi()
        output_directory = mktempdir()
        reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=10,
                                                 output_directory)
        run_surface_reconstruction_test(semi, CallbackSet(callback))
        files = readdir(output_directory)
        @test "surface_statistics.csv" in files
        @test "surface_statistics.json" in files
        @test "meta_surface.json" in files

        # Time series in the `PostprocessCallback` format, one value per event
        data = JSON.parsefile(joinpath(output_directory, "surface_statistics.json"))
        @test haskey(data, "meta")
        volume = data["volume_fluid_1"]
        @test length(volume["values"]) == length(volume["time"]) == 3
        @test volume["system_name"] == "fluid_1"
        @test volume["values"][end] ≈ callback.affect!.latest_stats.volume
        @test all(abs.(data["relative_volume_error_fluid_1"]["values"]) .<= 1.0e-3)

        csv = DataFrame(CSV.File(joinpath(output_directory, "surface_statistics.csv")))
        @test size(csv, 1) == 3
        @test "time" in names(csv)
        @test all(quantity * "_fluid_1" in names(csv)
                  for quantity in TrixiParticles.SURFACE_STATISTICS_QUANTITIES)

        # Metadata like the `SolutionSavingCallback`, plus the configuration
        meta = JSON.parsefile(joinpath(output_directory, "meta_surface.json"))
        @test haskey(meta, "simulation_info")
        @test haskey(meta, "system_data")
        configuration = meta["surface_reconstruction"]
        @test configuration["voxel_size"] ≈ reconstruction.voxel_size
        @test configuration["fluid_systems"] == ["fluid_1"]
        @test configuration["interval"] == 10
        @test configuration["domain"]["open_faces"] == ["+y"]

        # Statistics can be disabled; metadata is always written
        output_directory = mktempdir()
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=10,
                                                 output_directory, write_statistics=false)
        run_surface_reconstruction_test(semi, CallbackSet(callback); tspan=(0.0, 0.002))
        files = readdir(output_directory)
        @test !any(startswith("surface_statistics"), files)
        @test "meta_surface.json" in files
    end

    @testset "interpolated quantities" begin
        semi, particle_spacing, tank_size = make_test_semi()
        reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 interpolated_quantities=(:vorticity,))
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 formats=(:ply,),
                                                                 interpolated_quantities=(:pressure,))

        output_directory = mktempdir()
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=10,
                                                 output_directory,
                                                 interpolated_quantities=(:velocity,
                                                                          :pressure,
                                                                          :density))
        sol = run_surface_reconstruction_test(semi, CallbackSet(callback))
        # The last event is the final state
        vtp_files = filter(file -> startswith(file, "surface_fluid_1_") &&
                                   endswith(file, ".vtp"), readdir(output_directory))
        vtp_file = last(sort(vtp_files;
                             by=file -> parse(Int, match(r"_(\d+)\.vtp$", file)[1])))
        vtk = TrixiParticles.ReadVTK.VTKFile(joinpath(output_directory, vtp_file))
        point_data = TrixiParticles.ReadVTK.get_point_data(vtk)
        @test issubset(["Normals", "velocity", "pressure", "density"],
                       collect(keys(point_data)))
        n_points = size(TrixiParticles.ReadVTK.get_points(vtk), 2)
        density = TrixiParticles.ReadVTK.get_data(point_data["density"])
        velocity = TrixiParticles.ReadVTK.get_data(point_data["velocity"])
        @test length(density) == n_points
        @test size(velocity) == (3, n_points)
        # Surface vertices lie within the kernel support of the fluid particles
        finite_density = filter(isfinite, density)
        @test length(finite_density) >= 0.9 * n_points
        # Shepard interpolation is a convex combination of fluid particle values, so the
        # interpolated densities lie within the particle density range of the frame.
        fluid_system = semi.systems[1]
        v_final, u_final = sol.u[end].x
        TrixiParticles.update_systems_and_nhs(v_final, u_final, semi, sol.t[end])
        particle_density = TrixiParticles.current_density(TrixiParticles.wrap_v(v_final,
                                                                                fluid_system,
                                                                                semi),
                                                          fluid_system)
        @test minimum(finite_density) >= minimum(particle_density) * (1 - 1.0e-12)
        @test maximum(finite_density) <= maximum(particle_density) * (1 + 1.0e-12)
    end

    @testset "callback caches belong to one solve" begin
        semi, particle_spacing, tank_size = make_test_semi()
        reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=1,
                                                 output_directory=mktempdir(),
                                                 interpolated_quantities=(:density,))
        run_surface_reconstruction_test(semi, CallbackSet(callback); tspan=(0.0, 0.002))
        affect = callback.affect!
        previous_handler = affect.cpu_nhs_handler[]
        @test previous_handler === semi.neighborhood_search_handler

        # A new solve has different particle counts and search state. Reusing the callback
        # must replace its handler, topology cache, and warm-start seed at initialization.
        next_semi, _, _ = make_test_semi(fluid_size=(0.3, 0.2, 0.25))
        affect.boundary_topologies[2] = BoundaryTopology(TrixiParticles.Face[])
        reconstruction.cache.previous_isovalue[] = NaN
        run_surface_reconstruction_test(next_semi, CallbackSet(callback);
                                        tspan=(0.0, 0.002))
        @test affect.cpu_nhs_handler[] === next_semi.neighborhood_search_handler
        @test affect.cpu_nhs_handler[] !== previous_handler
        @test isempty(affect.boundary_topologies)
        @test isfinite(affect.latest_stats.volume)
        @test isfinite(reconstruction.cache.previous_isovalue[])
        @test length(affect.statistics_times) == 3
    end

    @testset "boundary output and static boundaries" begin
        semi, particle_spacing, tank_size = make_test_semi()
        fluid_system, wall_system = semi.systems
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=7)
        cube = RectangularShape(particle_spacing, (3, 3, 3), (0.15, 0.1, 0.15),
                                density=1000.0)
        cube_model = BoundaryModelDummyParticles(cube.density, cube.mass,
                                                 AdamiPressureExtrapolation(),
                                                 WendlandC2Kernel{3}(),
                                                 1.5 * particle_spacing; state_equation)
        semi_cube = Semidiscretization(fluid_system, wall_system,
                                       WallBoundarySystem(cube, cube_model))

        # Tracked boundary surfaces are written like fluid surfaces
        output_directory = mktempdir()
        reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
        callback = SurfaceReconstructionCallback(reconstruction, semi_cube; boundaries=3,
                                                 interval=10, output_directory,
                                                 write_boundaries=true)
        run_surface_reconstruction_test(semi_cube, CallbackSet(callback);
                                        tspan=(0.0, 0.002))
        files = readdir(output_directory)
        @test "surface_boundary_2.pvd" in files
        boundary_files = filter(file -> startswith(file, "surface_boundary_2_") &&
                                        endswith(file, ".vtp"), files)
        @test !isempty(boundary_files)
        vtk = TrixiParticles.ReadVTK.VTKFile(joinpath(output_directory,
                                                      first(boundary_files)))
        # A 3×3×3 lattice has 26 surface particles; the interior one is not written
        @test size(TrixiParticles.ReadVTK.get_points(vtk), 2) == 26

        # Static geometries clip the surface like tracked boundaries
        cube_mesh = BoundaryMesh(cube.coordinates,
                                 lattice_surface_topology(cube.coordinates))
        geometry = TrixiParticles.TriangleMesh(TrixiParticles.boundary_surface_mesh(cube_mesh))
        penetration(mesh) = -minimum(TrixiParticles.signed_distance(cube_mesh.bvh,
                                                                    SVector{3, Float64}(vertex))
                                     for vertex in mesh.vertices)
        for static_boundary in (geometry, cube_mesh)
            reconstruction = SurfaceReconstruction(; particle_spacing, tank_size)
            callback = SurfaceReconstructionCallback(reconstruction, semi;
                                                     boundaries=static_boundary,
                                                     interval=10,
                                                     output_directory=mktempdir())
            run_surface_reconstruction_test(semi, CallbackSet(callback);
                                            tspan=(0.0, 0.002))
            @test length(callback.affect!.static_boundaries) == 1
            @test penetration(callback.affect!.latest_mesh) <= reconstruction.voxel_size
        end

        # Unsupported boundary specifications are rejected
        @test_throws ArgumentError SurfaceReconstructionCallback(reconstruction, semi;
                                                                 boundaries="cube")
    end

    @testset "threading backend follows the simulation" begin
        # Default: inherit the semidiscretization's CPU backend
        semi, _, _ = make_test_semi(parallelization_backend=SerialBackend())
        reconstruction = SurfaceReconstruction(; particle_spacing=0.05,
                                               tank_size=(0.5, 0.6, 0.5))
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=10,
                                                 output_directory=mktempdir())
        run_surface_reconstruction_test(semi, CallbackSet(callback); tspan=(0.0, 0.002))
        @test reconstruction.cache.workspace[].backend isa SerialBackend

        # An explicitly configured backend takes precedence
        semi, _, _ = make_test_semi(parallelization_backend=SerialBackend())
        reconstruction = SurfaceReconstruction(; particle_spacing=0.05,
                                               tank_size=(0.5, 0.6, 0.5),
                                               parallelization_backend=ThreadsStaticBackend())
        callback = SurfaceReconstructionCallback(reconstruction, semi; interval=10,
                                                 output_directory=mktempdir())
        run_surface_reconstruction_test(semi, CallbackSet(callback); tspan=(0.0, 0.002))
        @test reconstruction.cache.workspace[].backend isa ThreadsStaticBackend

        # GPU and KernelAbstractions backends run on CPU copies with the default
        @test TrixiParticles.cpu_threading_backend(TrixiParticles.KernelAbstractions.CPU()) isa
              PolyesterBackend
        @test TrixiParticles.cpu_threading_backend(SerialBackend()) isa SerialBackend
    end

    @testset "configuration and fluid dimensions must match" begin
        tank_2d = RectangularTank(0.05, (0.25, 0.25), (0.5, 0.6), 1000.0;
                                  n_layers=3, acceleration=(0.0, -9.81))
        state_equation_2d = StateEquationCole(; sound_speed=10.0,
                                              reference_density=1000.0, exponent=7)
        fluid_2d = WeaklyCompressibleSPHSystem(tank_2d.fluid;
                                               smoothing_kernel=WendlandC2Kernel{2}(),
                                               smoothing_length=1.5 * 0.05,
                                               density_calculator=SummationDensity(),
                                               state_equation=state_equation_2d)
        boundary_2d = WallBoundarySystem(tank_2d.boundary,
                                         BoundaryModelDummyParticles(tank_2d.boundary.density,
                                                                     tank_2d.boundary.mass,
                                                                     AdamiPressureExtrapolation(),
                                                                     WendlandC2Kernel{2}(),
                                                                     1.5 * 0.05;
                                                                     state_equation=state_equation_2d))
        semi_2d = Semidiscretization(fluid_2d, boundary_2d)
        @test_throws ArgumentError SurfaceReconstructionCallback(SurfaceReconstruction(particle_spacing=0.05),
                                                                 semi_2d)
    end

    @testset "offline entry points" begin
        semi, particle_spacing, tank_size = make_test_semi()
        fluid_system = semi.systems[1]
        ode = semidiscretize(semi, (0.0, 0.001))
        v_ode, u_ode = ode.u0.x
        TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

        # Volumes from live state: particle count, all finite and positive
        volumes = TrixiParticles.particle_volumes(fluid_system,
                                                  TrixiParticles.wrap_v(v_ode,
                                                                        fluid_system,
                                                                        semi))
        @test length(volumes) == TrixiParticles.nparticles(fluid_system)
        @test all(isfinite, volumes) && all(>(0), volumes)
        # `SummationDensity` underestimates near-wall densities at t=0, so the deposited
        # volume legitimately exceeds the ideal lattice volume (bounded sanity check).
        @test sum(volumes) >= TrixiParticles.nparticles(fluid_system) * particle_spacing^3
        @test sum(volumes) <=
              1.25 *
              TrixiParticles.nparticles(fluid_system) * particle_spacing^3

        # System-based one-shot reconstruction with derived spacing
        _,
        stats = reconstruct_surface(fluid_system, v_ode, u_ode, semi;
                                    tank_size=tank_size)
        @test abs(100 * (stats["volume"] - sum(volumes)) / sum(volumes)) <= 0.5

        # Initial-condition entry
        tank = RectangularTank(particle_spacing, (0.25, 0.25, 0.25), tank_size, 1000.0;
                               n_layers=3, acceleration=(0.0, -9.81, 0.0))
        _, stats_ic = reconstruct_surface(tank.fluid; tank_size=tank_size)
        @test stats_ic.particle_volume ≈ sum(tank.fluid.mass ./ tank.fluid.density)

        # Semidiscretization + solution entry
        sol = solve(semidiscretize(semi, (0.0, 0.02)), RDPK3SpFSAL35(); dt=0.001,
                    adaptive=false, save_everystep=true)
        _,
        stats_sol = reconstruct_surface(semi, sol; frame=lastindex(sol.u),
                                        tank_size=tank_size)
        # Same state through the system entry gives the same deposited volume
        v_final, u_final = sol.u[end].x
        TrixiParticles.update_systems_and_nhs(v_final, u_final, semi, sol.t[end])
        volumes_final = TrixiParticles.particle_volumes(semi.systems[1],
                                                        TrixiParticles.wrap_v(v_final,
                                                                              semi.systems[1],
                                                                              semi))
        @test stats_sol["particle_volume"] ≈ sum(volumes_final) rtol=1.0e-12

        # Middle frames must refresh the density cache to the frame state; without the
        # refresh, `SummationDensity` caches from the last RHS evaluation (the final
        # state) would corrupt the deposited volumes.
        _, stats_mid = reconstruct_surface(semi, sol; frame=5, tank_size=tank_size)
        @test stats_mid["particle_volume"] !=
              stats_sol["particle_volume"]

        # A solution of a different semidiscretization is rejected
        semi_other, _, _ = make_test_semi(fluid_size=(0.2, 0.25, 0.25))
        @test_throws ArgumentError reconstruct_surface(semi_other, sol; tank_size)
    end
end
