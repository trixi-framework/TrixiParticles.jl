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
                @test_nowarn_mod trixi_include(@__MODULE__,
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

                    @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
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

            @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
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
                                                                                             cell_list))

            # Note that this simulation only takes 36 time steps on the CPU.
            # Due to https://github.com/JuliaGPU/Metal.jl/issues/549, it doesn't work on Metal.
            trixi_include_changeprecision(Float32, @__MODULE__,
                                          joinpath(examples_dir(), "fluid",
                                                   "dam_break_3d.jl"),
                                          tspan=(0.0f0, 0.1f0),
                                          fluid_particle_spacing=0.1,
                                          semi=semi_fullgrid,
                                          parallelization_backend=Main.parallelization_backend,
                                          maxiters=36)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
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
                                                viscosity=ViscosityAdami(nu=0.0015f0),),
                "WCSPH with ViscosityMorris" => (
                                                 # from 0.02*10.0*1.2*0.05/8
                                                 viscosity=ViscosityMorris(nu=0.0015f0),),
                "WCSPH with ViscosityAdami and SummationDensity" => (
                                                                     # from 0.02*10.0*1.2*0.05/8
                                                                     viscosity=ViscosityAdami(nu=0.0015f0),
                                                                     fluid_density_calculator=SummationDensity(),
                                                                     maxiters=38, # 38 time steps on CPU
                                                                     clip_negative_pressure=true),
                # Broken due to https://github.com/JuliaGPU/CUDA.jl/issues/2681
                # and https://github.com/JuliaGPU/Metal.jl/issues/550.
                # "WCSPH with SchoenbergQuarticSplineKernel" => (smoothing_length=1.1,
                #                                                smoothing_kernel=SchoenbergQuarticSplineKernel{2}()),
                "WCSPH with SchoenbergQuinticSplineKernel" => (smoothing_length=1.1,
                                                               smoothing_kernel=SchoenbergQuinticSplineKernel{2}()),
                "WCSPH with WendlandC2Kernel" => (smoothing_length=3.0,
                                                  smoothing_kernel=WendlandC2Kernel{2}()),
                "WCSPH with WendlandC4Kernel" => (smoothing_length=3.5,
                                                  smoothing_kernel=WendlandC4Kernel{2}()),
                "WCSPH with WendlandC6Kernel" => (smoothing_length=4.0,
                                                  smoothing_kernel=WendlandC6Kernel{2}()),
                "EDAC with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1.0f-4),
                                                    fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                             smoothing_kernel,
                                                                                             smoothing_length,
                                                                                             sound_speed,
                                                                                             viscosity=viscosity,
                                                                                             density_calculator=ContinuityDensity(),
                                                                                             acceleration=(0.0,
                                                                                                           -gravity))),
                "EDAC with SummationDensity" => (fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                          smoothing_kernel,
                                                                                          smoothing_length,
                                                                                          sound_speed,
                                                                                          viscosity=viscosity,
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
                                                                                                     cell_list))

                    # Run the simulation
                    @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
                                                                   joinpath(examples_dir(),
                                                                            "fluid",
                                                                            "hydrostatic_water_column_2d.jl");
                                                                   semi=semi_fullgrid,
                                                                   parallelization_backend=Main.parallelization_backend,
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
                                                                                             cell_list))

            @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
                                                           joinpath(examples_dir(), "fluid",
                                                                    "periodic_channel_2d.jl"),
                                                           tspan=(0.0f0, 0.1f0),
                                                           semi=semi_fullgrid,
                                                           parallelization_backend=Main.parallelization_backend)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end

        # Test open boundaries and steady-state callback
        @testset "fluid/pipe_flow_2d.jl - steady state reached (`dt`)" begin
            # TODO This currently doesn't work on GPUs due to
            # https://github.com/trixi-framework/PointNeighbors.jl/issues/20.

            # # Import variables into scope
            # trixi_include_changeprecision(Float32, @__MODULE__,
            #                               joinpath(examples_dir(), "fluid",
            #                                        "pipe_flow_2d.jl"),
            #                               sol=nothing, ode=nothing)

            # # Neighborhood search with `FullGridCellList` for GPU compatibility
            # min_corner = minimum(pipe.boundary.coordinates, dims=2) .- 8 * particle_spacing
            # max_corner = maximum(pipe.boundary.coordinates, dims=2) .+ 8 * particle_spacing
            # cell_list = FullGridCellList(; min_corner, max_corner)
            # semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
            #                                    neighborhood_search=GridNeighborhoodSearch{2}(;
            #                                                                                  cell_list))

            # steady_state_reached = SteadyStateReachedCallback(; dt=0.002, interval_size=10)

            # @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
            #                                                joinpath(examples_dir(), "fluid",
            #                                                         "pipe_flow_2d.jl"),
            #                                                extra_callback=steady_state_reached,
            #                                                tspan=(0.0f0, 1.5f0),
            #                                                semi=semi_fullgrid,
            #                                                parallelization_backend=Main.parallelization_backend)

            # TODO This currently doesn't work on GPUs due to
            # https://github.com/trixi-framework/PointNeighbors.jl/issues/20.

            # Make sure that the simulation is terminated after a reasonable amount of time
            @test_skip 0.1 < sol.t[end] < 1.0
            @test_skip sol.retcode == ReturnCode.Terminated
            @test_skip TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1]) == Main.parallelization_backend
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
                                                                                             cell_list))

            @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
                                                           joinpath(examples_dir(), "solid",
                                                                    "oscillating_beam_2d.jl"),
                                                           tspan=(0.0f0, 0.1f0),
                                                           semi=semi_fullgrid,
                                                           parallelization_backend=Main.parallelization_backend)
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
                                                                                             cell_list))

            @test_nowarn_mod trixi_include_changeprecision(Float32, @__MODULE__,
                                                           joinpath(examples_dir(), "fsi",
                                                                    "dam_break_gate_2d.jl"),
                                                           tspan=(0.0f0, 0.4f0),
                                                           semi=semi_fullgrid,
                                                           # Needs <1500 steps on the CPU
                                                           maxiters=1500,
                                                           parallelization_backend=Main.parallelization_backend)
            @test sol.retcode == ReturnCode.Success
            backend = TrixiParticles.KernelAbstractions.get_backend(sol.u[end].x[1])
            @test backend == Main.parallelization_backend
        end
    end
end
