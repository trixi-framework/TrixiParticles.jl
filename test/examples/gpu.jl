const TRIXIPARTICLES_TEST_ = lowercase(get(ENV, "TRIXIPARTICLES_TEST", "all"))

if TRIXIPARTICLES_TEST_ == "cuda"
    using CUDA
    CUDA.versioninfo()
    parallelization_backend = CUDABackend()
    supports_double_precision = true
elseif TRIXIPARTICLES_TEST_ == "amdgpu"
    using AMDGPU
    AMDGPU.versioninfo()
    parallelization_backend = ROCBackend()
    supports_double_precision = true
elseif TRIXIPARTICLES_TEST_ == "metal"
    using Metal
    Metal.versioninfo()
    parallelization_backend = MetalBackend()
    supports_double_precision = false
elseif TRIXIPARTICLES_TEST_ == "oneapi"
    using oneAPI
    oneAPI.versioninfo()
    parallelization_backend = oneAPIBackend()
    # The runners are using an iGPU, which does not support double precision
    supports_double_precision = false
else
    error("Unknown GPU backend: $TRIXIPARTICLES_TEST_")
end

@testset verbose=true "Examples $TRIXIPARTICLES_TEST_" begin
    @testset verbose=true "Fluid" begin
        @trixi_testset "fluid/dam_break_2d_gpu.jl Float64" begin
            if Main.supports_double_precision
                @trixi_test_nowarn trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "dam_break_2d_gpu.jl"),
                                                 tspan=(0.0, 0.1),
                                                 parallelization_backend=Main.parallelization_backend) [
                    r"┌ Info: The desired tank length in y-direction .*\n",
                    r"└ New tank length in y-direction.*\n"
                ]
                @test semi.neighborhood_searches[1][1].cell_list isa FullGridCellList
                @test sol.retcode == ReturnCode.Success
                backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
                @test backend == Main.parallelization_backend
            else
                error = "Metal does not support Float64 values, try using Float32 instead"
                @test_throws error trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "dam_break_2d_gpu.jl"),
                                                 tspan=(0.0, 0.1),
                                                 parallelization_backend=Main.parallelization_backend)
            end
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(),
                                                   "fluid", "dam_break_2d.jl");
                                          sol=nothing, ode=nothing)

            dam_break_tests = Dict(
                "default" => (),
                "DensityDiffusionMolteniColagrossi" => (density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1f0),),
                "DensityDiffusionFerrari" => (density_diffusion=DensityDiffusionFerrari(),)
            )

            for (test_description, kwargs) in dam_break_tests
                @testset "$test_description" begin
                    println("═"^100)
                    println("$test_description")

                    @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                                     joinpath(examples_dir(),
                                                                              "fluid",
                                                                              "dam_break_2d_gpu.jl");
                                                                     tspan=(0.0f0, 0.1f0),
                                                                     parallelization_backend=Main.parallelization_backend,
                                                                     kwargs...) [
                        r"┌ Info: The desired tank length in y-direction .*\n",
                        r"└ New tank length in y-direction.*\n"
                    ]
                    @test semi.neighborhood_searches[1][1].cell_list isa FullGridCellList
                    @test sol.retcode == ReturnCode.Success
                    backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
                    @test backend == Main.parallelization_backend
                end
            end
        end

        @trixi_testset "fluid/dam_break_2d_gpu.jl Float32 BoundaryModelMonaghanKajtar" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(),
                                                   "fluid", "dam_break_2d.jl");
                                          boundary_layers=1, spacing_ratio=3,
                                          sol=nothing, ode=nothing)

            boundary_model = BoundaryModelMonaghanKajtar(0.5f0,
                                                         spacing_ratio,
                                                         boundary_particle_spacing,
                                                         tank.boundary.mass)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "dam_break_2d_gpu.jl");
                                                             tspan=(0.0f0, 0.1f0),
                                                             boundary_layers=1,
                                                             spacing_ratio=3,
                                                             boundary_model=boundary_model,
                                                             parallelization_backend=Main.parallelization_backend) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n"
            ]
            @test semi.neighborhood_searches[1][1].cell_list isa FullGridCellList
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/dam_break_3d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "dam_break_3d.jl"),
                                          fluid_particle_spacing=0.1,
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2)
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                               neighborhood_search=GridNeighborhoodSearch{3}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            # Note that this simulation only takes 42 time steps on the CPU.
            # TODO This takes 43 time steps on Metal.
            # Maybe related to https://github.com/JuliaGPU/Metal.jl/issues/549
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "dam_break_3d.jl"),
                                          tspan=(0.0f0, 0.1f0),
                                          fluid_particle_spacing=0.1,
                                          semi=semi_fullgrid,
                                          maxiters=43)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend

            @testset "`SymplecticPositionVerlet`" begin
                stepsize_callback = StepsizeCallback(cfl=0.65)
                callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

                sol = solve(ode, SymplecticPositionVerlet(),
                            dt=1, # This is overwritten by the stepsize callback
                            save_everystep=false, callback=callbacks)
                @test sol.retcode == ReturnCode.Success
                @test maximum(maximum.(abs, sol.u[end].x)) < 2^15
                backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
                @test backend == Main.parallelization_backend
            end
        end

        # Short tests to make sure that different models and kernels work on GPUs
        @trixi_testset "fluid/hydrostatic_water_column_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "hydrostatic_water_column_2d.jl"),
                                          sol=nothing, ode=nothing)

            hydrostatic_water_column_tests = Dict(
                "WCSPH default" => (),
                "WCSPH with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1.0f-4),),
                "WCSPH with SummationDensity" => (fluid_density_calculator=SummationDensity(),
                                                  clip_negative_pressure=true),
                "WCSPH with ViscosityAdami" => (
                                                # from 0.02*10.0*1.2*0.05/8
                                                viscosity_fluid=ViscosityAdami(nu=0.0015f0),),
                "WCSPH with ViscosityMorris" => (
                                                 # from 0.02*10.0*1.2*0.05/8
                                                 viscosity_fluid=ViscosityMorris(nu=0.0015f0),),
                "WCSPH with ViscosityAdami and SummationDensity" => (
                                                                     # from 0.02*10.0*1.2*0.05/8
                                                                     viscosity_fluid=ViscosityAdami(nu=0.0015f0),
                                                                     fluid_density_calculator=SummationDensity(),
                                                                     maxiters=38, # 38 time steps on CPU
                                                                     clip_negative_pressure=true),
                # Broken due to https://github.com/JuliaGPU/CUDA.jl/issues/2681
                # and https://github.com/JuliaGPU/Metal.jl/issues/550.
                # "WCSPH with SchoenbergQuarticSplineKernel" => (smoothing_length=1.1,
                #                                                smoothing_kernel=SchoenbergQuarticSplineKernel{2}()),
                "WCSPH with SchoenbergQuinticSplineKernel" => (smoothing_length=1.1,
                                                               smoothing_kernel=SchoenbergQuinticSplineKernel{2}()),
                "WCSPH with WendlandC2Kernel" => (smoothing_length=1.5,
                                                  smoothing_kernel=WendlandC2Kernel{2}()),
                "WCSPH with WendlandC4Kernel" => (smoothing_length=1.75,
                                                  smoothing_kernel=WendlandC4Kernel{2}()),
                "WCSPH with WendlandC6Kernel" => (smoothing_length=2.0,
                                                  smoothing_kernel=WendlandC6Kernel{2}()),
                "EDAC with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1.0f-4),
                                                    fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                             smoothing_kernel,
                                                                                             smoothing_length,
                                                                                             sound_speed,
                                                                                             viscosity=viscosity_fluid,
                                                                                             density_calculator=ContinuityDensity(),
                                                                                             acceleration=(0.0,
                                                                                                           -gravity))),
                "EDAC with SummationDensity" => (fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                          smoothing_kernel,
                                                                                          smoothing_length,
                                                                                          sound_speed,
                                                                                          viscosity=viscosity_fluid,
                                                                                          density_calculator=SummationDensity(),
                                                                                          acceleration=(0.0,
                                                                                                        -gravity)),)
            )

            for (test_description, kwargs) in hydrostatic_water_column_tests
                @testset "$test_description" begin
                    println("═"^100)
                    println("$test_description")

                    # Create systems with the given keyword arguments
                    trixi_include_changeprecision(Float32, @__MODULE__,
                                                  joinpath(examples_dir(), "fluid",
                                                           "hydrostatic_water_column_2d.jl");
                                                  sol=nothing, ode=nothing,
                                                  kwargs...)

                    # Neighborhood search with `FullGridCellList` for GPU compatibility
                    min_corner = minimum(tank.boundary.coordinates, dims=2)
                    max_corner = maximum(tank.boundary.coordinates, dims=2)
                    cell_list = FullGridCellList(; min_corner, max_corner,
                                                 max_points_per_cell=500)
                    semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                                       neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                                     cell_list),
                                                       parallelization_backend=Main.parallelization_backend)

                    # Run the simulation
                    @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                                     joinpath(examples_dir(),
                                                                              "fluid",
                                                                              "hydrostatic_water_column_2d.jl");
                                                                     semi=semi_fullgrid,
                                                                     tspan=(0.0f0, 0.1f0),
                                                                     kwargs...)

                    @test sol.retcode == ReturnCode.Success
                    backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
                    @test backend == Main.parallelization_backend
                end
            end
        end

        # Test periodic neighborhood search
        @trixi_testset "fluid/periodic_channel_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "periodic_channel_2d.jl"),
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            search_radius = TrixiParticles.compact_support(smoothing_kernel,
                                                           smoothing_length)
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2) .+ 2 * search_radius
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                               neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "periodic_channel_2d.jl"),
                                                             tspan=(0.0f0, 0.1f0),
                                                             semi=semi_fullgrid)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        # Test open boundaries and steady-state callback
        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelLastiwka (WCSPH)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             wcsph=true,
                                                             parallelization_backend=Main.parallelization_backend)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelLastiwka (EDAC)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             parallelization_backend=Main.parallelization_backend)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelTafuni (EDAC)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             open_boundary_model=BoundaryModelTafuni(),
                                                             boundary_type_in=BidirectionalFlow(),
                                                             boundary_type_out=BidirectionalFlow(),
                                                             reference_density_in=nothing,
                                                             reference_pressure_in=nothing,
                                                             reference_density_out=nothing,
                                                             reference_velocity_out=nothing,
                                                             parallelization_backend=Main.parallelization_backend)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelTafuni (WCSPH)" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             tspan=(0.0f0, 0.5f0),
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "pipe_flow_2d.jl"),
                                                             wcsph=true, sound_speed=20.0f0,
                                                             pressure=0.0f0,
                                                             open_boundary_model=BoundaryModelTafuni(;
                                                                                                     mirror_method=ZerothOrderMirroring()),
                                                             boundary_type_in=BidirectionalFlow(),
                                                             boundary_type_out=BidirectionalFlow(),
                                                             reference_density_in=nothing,
                                                             reference_pressure_in=nothing,
                                                             reference_density_out=nothing,
                                                             reference_pressure_out=nothing,
                                                             reference_velocity_out=nothing,
                                                             parallelization_backend=Main.parallelization_backend)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end
    end

    @testset verbose=true "Solid" begin
        # TODO after https://github.com/trixi-framework/PointNeighbors.jl/pull/10
        # is merged, there should be no need to use the `FullGridCellList`.
        @trixi_testset "solid/oscillating_beam_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "solid",
                                                   "oscillating_beam_2d.jl"),
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            min_corner = minimum(solid.coordinates, dims=2)
            max_corner = maximum(solid.coordinates, dims=2)
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(solid_system,
                                               neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "solid",
                                                                      "oscillating_beam_2d.jl"),
                                                             tspan=(0.0f0, 0.1f0),
                                                             semi=semi_fullgrid)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end
    end

    @testset verbose=true "FSI" begin
        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            # Import variables into scope
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fsi",
                                                   "dam_break_gate_2d.jl"),
                                          sol=nothing, ode=nothing)

            # Neighborhood search with `FullGridCellList` for GPU compatibility
            min_corner = minimum(tank.boundary.coordinates, dims=2)
            max_corner = maximum(tank.boundary.coordinates, dims=2)
            max_corner[2] = gate_height + movement_function(0.1)[2]
            # We need a very high `max_points_per_cell` because the plate resolution
            # is much finer than the fluid resolution.
            cell_list = FullGridCellList(; min_corner, max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system_tank,
                                               boundary_system_gate, solid_system,
                                               neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                             cell_list),
                                               parallelization_backend=Main.parallelization_backend)

            @trixi_test_nowarn trixi_include_changeprecision(Float32, @__MODULE__,
                                                             joinpath(examples_dir(), "fsi",
                                                                      "dam_break_gate_2d.jl"),
                                                             tspan=(0.0f0, 0.4f0),
                                                             semi=semi_fullgrid,
                                                             # Needs <1500 steps on the CPU
                                                             maxiters=1500)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end
    end

    @testset verbose=true "Postprocessing $TRIXIPARTICLES_TEST_" begin
        @testset verbose=true "Interpolation" begin
            # Run the dam break example to get a solution
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "dam_break_2d_gpu.jl");
                                          fluid_particle_spacing=0.05f0,
                                          tspan=(0.0f0, 0.01f0),
                                          parallelization_backend=Main.parallelization_backend)

            semi_new = sol.prob.p

            @testset verbose=true "Line" begin
                # Interpolation parameters
                n_interpolation_points = 10
                start_point = Float32[0.5, 0.0]
                end_point = Float32[0.5, 0.5]

                result = interpolate_line(start_point, end_point, n_interpolation_points,
                                          semi_new, semi_new.systems[1], sol;
                                          cut_off_bnd=false)

                @test isapprox(Array(result.computed_density),
                               Float32[500.33255, 893.09766, 997.7032, 1001.14355, 1001.234,
                                       1001.0098, 1000.4352, 999.7572, 999.1139, 989.6319])

                @test isapprox(Array(result.density),
                               Float32[1002.3152, 1002.19653, 1001.99915, 1001.7685,
                                       1001.5382,
                                       1001.3093, 1001.0836, 1000.8649, 1000.635,
                                       1000.4053])

                @test isapprox(Array(result.pressure),
                               Float32[5450.902, 5171.2856, 4706.551, 4163.9185, 3621.5042,
                                       3082.6948, 2551.5725, 2036.1208, 1494.8608,
                                       954.14355])
            end

            @testset verbose=true "Plane" begin
                interpolation_start = Float32[0.0, 0.0]
                interpolation_end = Float32[1.0, 1.0]
                resolution = 0.4f0

                result = interpolate_plane_2d(interpolation_start, interpolation_end,
                                              resolution, semi_new, semi_new.systems[1],
                                              sol;
                                              cut_off_bnd=false)

                @test isapprox(Array(result.computed_density),
                               Float32[250.18625, 500.34482, 499.77225, 254.3632, 499.58026,
                                       999.1413, 998.6351, 503.0122])

                @test isapprox(Array(result.density),
                               Float32[1002.34467, 1002.3365, 1001.5021, 999.7109,
                                       1000.84863,
                                       1000.8373, 1000.3423, 1000.20734])

                @test isapprox(Array(result.pressure),
                               Float32[5520.0513, 5501.1846, 3536.2256, -680.5194,
                                       1997.7814,
                                       1971.0717, 805.8584, 488.4068])
            end
        end
    end
end
