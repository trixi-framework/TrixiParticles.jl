@trixi_testset "planar reconstruction GPU callback" begin
    using OrdinaryDiffEqLowStorageRK
    TP = TrixiParticles
    spacing = 0.05f0
    shape = RectangularShape(spacing, (7, 6), (0.1f0, 0.1f0);
                             density=1000.0f0, coordinates_eltype=Float32)
    fluid = WeaklyCompressibleSPHSystem(shape; smoothing_kernel=WendlandC2Kernel{2}(),
                                        smoothing_length=1.5f0 * spacing,
                                        density_calculator=SummationDensity(),
                                        state_equation=StateEquationCole(;
                                                                         sound_speed=10.0f0,
                                                                         reference_density=1000.0f0,
                                                                         exponent=7))
    cells = FullGridCellList(; min_corner=(-0.5f0, -0.5f0), max_corner=(1.5f0, 1.5f0))
    semi = Semidiscretization(fluid; parallelization_backend=Main.parallelization_backend,
                              neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                            cell_list=cells))
    rec = SurfaceReconstruction(; particle_spacing=spacing, ndims=2)
    mktempdir() do dir
        callback = SurfaceReconstructionCallback(rec, semi; interval=1,
                                                 output_directory=dir,
                                                 interpolated_quantities=(:density,
                                                                          :velocity))
        ode = semidiscretize(semi, (0.0f0, 0.002f0))
        sol = solve(ode, RDPK3SpFSAL35(); dt=0.001f0, adaptive=false, save_everystep=true,
                    callback=CallbackSet(callback))
        @test sol.retcode == ReturnCode.Success
        @test TP.KernelAbstractions.get_backend(sol.u[end].x[1]) ==
              Main.parallelization_backend
        @test eltype(sol.u[end].x[1]) == Float32
        @test ndims(callback.affect!.latest_mesh) == 2
        @test callback.affect!.cpu_nhs_handler[] !== nothing
        @test rec.cache.workspace[].backend isa PolyesterBackend
        offline, stats = reconstruct_surface(semi, sol)
        @test ndims(offline) == 2
        @test stats["area"] ≈ stats["particle_area"] rtol=0.001
        vtk = TP.ReadVTK.VTKFile(joinpath(dir, "surface_fluid_1_0.vtp"))
        @test all(iszero, TP.ReadVTK.get_points(vtk)[3, :])
        @test !isempty(TP.ReadVTK.get_primitives(vtk, "Lines").connectivity)
        density = TP.ReadVTK.get_data(TP.ReadVTK.get_point_data(vtk)["density"])
        @test any(isfinite, density)
    end
end
