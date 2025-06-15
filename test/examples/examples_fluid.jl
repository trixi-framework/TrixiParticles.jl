@testset verbose=true "Fluid" begin
    @trixi_testset "fluid/hydrostatic_water_column_2d.jl" begin
        # Import variables into scope
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "fluid",
                               "hydrostatic_water_column_2d.jl"),
                      sol=nothing, ode=nothing)

        # Neighborhood search for `FullGridCellList` test below
        min_corner = minimum(tank.boundary.coordinates, dims=2)
        max_corner = maximum(tank.boundary.coordinates, dims=2)
        cell_list = FullGridCellList(; min_corner, max_corner)
        semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                           neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                         cell_list))

        hydrostatic_water_column_tests = Dict(
            "WCSPH default" => (),
            "with Threads.@threads :static" => (parallelization_backend=ThreadsStaticBackend(),),
            "with Threads.@threads :dynamic" => (parallelization_backend=ThreadsDynamicBackend(),),
            "with SerialBackend" => (parallelization_backend=SerialBackend(),),
            "WCSPH with FullGridCellList" => (semi=semi_fullgrid,),
            "WCSPH with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1e-4),),
            "WCSPH with SummationDensity" => (fluid_density_calculator=SummationDensity(),
                                              clip_negative_pressure=true),
            "WCSPH with ViscosityAdami" => (
                                            # from 0.02*10.0*1.2*0.05/8
                                            viscosity_fluid=ViscosityAdami(nu=0.0015),),
            "WCSPH with ViscosityMorris" => (
                                             # from 0.02*10.0*1.2*0.05/8
                                             viscosity_fluid=ViscosityMorris(nu=0.0015),),
            "WCSPH with ViscosityAdami and SummationDensity" => (
                                                                 # from 0.02*10.0*1.2*0.05/8
                                                                 viscosity_fluid=ViscosityAdami(nu=0.0015),
                                                                 fluid_density_calculator=SummationDensity(),
                                                                 clip_negative_pressure=true),
            "WCSPH with ViscosityMorris and SummationDensity" => (
                                                                  # from 0.02*10.0*1.2*0.05/8
                                                                  viscosity_fluid=ViscosityMorris(nu=0.0015),
                                                                  fluid_density_calculator=SummationDensity(),
                                                                  clip_negative_pressure=true),
            "WCSPH with smoothing_length=1.3" => (smoothing_length=1.3,),
            "WCSPH with SchoenbergQuarticSplineKernel" => (smoothing_length=1.1,
                                                           smoothing_kernel=SchoenbergQuarticSplineKernel{2}()),
            "WCSPH with SchoenbergQuinticSplineKernel" => (smoothing_length=1.1,
                                                           smoothing_kernel=SchoenbergQuinticSplineKernel{2}()),
            "WCSPH with WendlandC2Kernel" => (smoothing_length=1.5,
                                              smoothing_kernel=WendlandC2Kernel{2}()),
            "WCSPH with WendlandC4Kernel" => (smoothing_length=1.75,
                                              smoothing_kernel=WendlandC4Kernel{2}()),
            "WCSPH with WendlandC6Kernel" => (smoothing_length=2.0,
                                              smoothing_kernel=WendlandC6Kernel{2}()),
            "EDAC with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1e-4),
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
                                                                                                    -gravity)),),
            "EDAC with ViscosityAdami" => (fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                    smoothing_kernel,
                                                                                    smoothing_length,
                                                                                    sound_speed,
                                                                                    viscosity=ViscosityAdami(nu=0.0015),
                                                                                    density_calculator=ContinuityDensity(),
                                                                                    acceleration=(0.0,
                                                                                                  -gravity)),),
            "EDAC with ViscosityMorris" => (fluid_system=EntropicallyDampedSPHSystem(tank.fluid,
                                                                                     smoothing_kernel,
                                                                                     smoothing_length,
                                                                                     sound_speed,
                                                                                     viscosity=ViscosityMorris(nu=0.0015),
                                                                                     density_calculator=ContinuityDensity(),
                                                                                     acceleration=(0.0,
                                                                                                   -gravity)),)
        )

        for (test_description, kwargs) in hydrostatic_water_column_tests
            @testset "$test_description" begin
                println("═"^100)
                println("$test_description")

                @trixi_test_nowarn trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "hydrostatic_water_column_2d.jl");
                                                 kwargs...)

                @test sol.retcode == ReturnCode.Success
                @test count_rhs_allocations(sol, semi) == 0
            end
        end
    end

    @trixi_testset "fluid/oscillating_drop_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "oscillating_drop_2d.jl"))
        @test sol.retcode == ReturnCode.Success
        # This error varies between serial and multithreaded runs
        @test isapprox(error_A, 0, atol=2e-4)
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/hydrostatic_water_column_3d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "hydrostatic_water_column_3d.jl"),
                                         tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/hydrostatic_water_column_3d.jl with SummationDensity" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "hydrostatic_water_column_3d.jl"),
                                         tspan=(0.0, 0.1),
                                         fluid_density_calculator=SummationDensity(),
                                         clip_negative_pressure=true)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/accelerated_tank_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "accelerated_tank_2d.jl"))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/dam_break_2d.jl" begin
        # Import variables into scope
        trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                      boundary_layers=1, spacing_ratio=3, sol=nothing, semi=nothing,
                      ode=nothing)

        dam_break_tests = Dict(
            "default" => (),
            "with SummationDensity" => (fluid_density_calculator=SummationDensity(),
                                        clip_negative_pressure=true),
            "with DensityDiffusionMolteniColagrossi" => (density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),),
            "no density diffusion" => (density_diffusion=nothing,),
            "with KernelAbstractions" => (parallelization_backend=TrixiParticles.KernelAbstractions.CPU(),),
            "with BoundaryModelMonaghanKajtar" => (boundary_model=BoundaryModelMonaghanKajtar(gravity,
                                                                                              spacing_ratio,
                                                                                              boundary_particle_spacing,
                                                                                              tank.boundary.mass),
                                                   boundary_layers=1, spacing_ratio=3),
            "with SurfaceTensionAkinci" => (surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.025),
                                            fluid_particle_spacing=0.5 *
                                                                   fluid_particle_spacing,
                                            smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                            smoothing_length=0.5 *
                                                             fluid_particle_spacing,
                                            correction=AkinciFreeSurfaceCorrection(fluid_density),
                                            density_diffusion=nothing,
                                            adhesion_coefficient=0.05,
                                            sound_speed=100.0,
                                            reference_particle_spacing=fluid_particle_spacing)
        )

        for (test_description, kwargs) in dam_break_tests
            @testset "$test_description" begin
                println("═"^100)
                println("$test_description")

                @trixi_test_nowarn trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "dam_break_2d.jl");
                                                 tspan=(0, 0.1), kwargs...) [
                    r"┌ Info: The desired tank length in y-direction .*\n",
                    r"└ New tank length in y-direction.*\n"
                ]

                @test sol.retcode == ReturnCode.Success
                @test count_rhs_allocations(sol, semi) == 0
            end
        end

        @testset "Float32" begin
            @trixi_test_nowarn trixi_include_changeprecision(Float32,
                                                             @__MODULE__,
                                                             joinpath(examples_dir(),
                                                                      "fluid",
                                                                      "dam_break_2d.jl"),
                                                             tspan=(0, 0.1)) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
            @test eltype(sol) == Float32
        end
    end

    @trixi_testset "fluid/dam_break_2d_gpu.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "dam_break_2d_gpu.jl"),
                                         tspan=(0.0, 0.1)) [
            r"┌ Info: The desired tank length in y-direction .*\n",
            r"└ New tank length in y-direction.*\n"
        ]
        @test semi.neighborhood_searches[1][1].cell_list isa FullGridCellList
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/dam_break_oil_film_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "dam_break_oil_film_2d.jl"),
                                         tspan=(0.0, 0.05)) [
            r"┌ Info: The desired tank length in y-direction .*\n",
            r"└ New tank length in y-direction.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/dam_break_2phase_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "dam_break_2phase_2d.jl"),
                                         tspan=(0.0, 0.05)) [
            r"┌ Info: The desired tank length in y-direction .*\n",
            r"└ New tank length in y-direction.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/dam_break_3d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "dam_break_3d.jl"),
                                         tspan=(0.0, 0.1), fluid_particle_spacing=0.1)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/falling_water_column_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "falling_water_column_2d.jl"),
                                         tspan=(0.0, 0.4))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/periodic_channel_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "periodic_channel_2d.jl"),
                                         tspan=(0.0, 0.4))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/periodic_channel_2d.jl with PST" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "periodic_channel_2d.jl"),
                                         tspan=(0.0, 0.2),
                                         extra_callback=ParticleShiftingCallback())
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/periodic_channel_2d.jl with PST and TIC" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "periodic_channel_2d.jl"),
                                         tspan=(0.0, 0.2),
                                         extra_callback=ParticleShiftingCallback(),
                                         pressure_acceleration=tensile_instability_control)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelLastiwka (WCSPH)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_2d.jl"),
                                         wcsph=true)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelLastiwka (EDAC)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_2d.jl"))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelTafuni (EDAC)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_2d.jl"),
                                         open_boundary_model=BoundaryModelTafuni(),
                                         boundary_type_in=BidirectionalFlow(),
                                         boundary_type_out=BidirectionalFlow(),
                                         reference_density_in=nothing,
                                         reference_pressure_in=nothing,
                                         reference_density_out=nothing,
                                         reference_velocity_out=nothing)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/pipe_flow_2d.jl - BoundaryModelTafuni (WCSPH)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_2d.jl"),
                                         wcsph=true, sound_speed=20.0, pressure=0.0,
                                         open_boundary_model=BoundaryModelTafuni(),
                                         boundary_type_in=BidirectionalFlow(),
                                         boundary_type_out=BidirectionalFlow(),
                                         reference_density_in=nothing,
                                         reference_pressure_in=nothing,
                                         reference_density_out=nothing,
                                         reference_pressure_out=nothing,
                                         reference_velocity_out=nothing)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/pipe_flow_2d.jl - steady state reached (`dt`)" begin
        steady_state_reached = SteadyStateReachedCallback(; dt=0.002, interval_size=10,
                                                          reltol=1e-3)

        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_2d.jl"),
                                         extra_callback=steady_state_reached,
                                         tspan=(0.0, 1.5), viscosity_boundary=nothing)

        # Make sure that the simulation is terminated after a reasonable amount of time
        @test 0.1 < sol.t[end] < 1.0
        @test sol.retcode == ReturnCode.Terminated
    end

    @trixi_testset "fluid/pipe_flow_2d.jl - steady state reached (`interval`)" begin
        steady_state_reached = SteadyStateReachedCallback(; interval=1, interval_size=10,
                                                          reltol=1e-3)
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_2d.jl"),
                                         extra_callback=steady_state_reached, dtmax=2e-3,
                                         tspan=(0.0, 1.5), viscosity_boundary=nothing)

        # Make sure that the simulation is terminated after a reasonable amount of time
        @test 0.1 < sol.t[end] < 1.0
        @test sol.retcode == ReturnCode.Terminated
    end

    @trixi_testset "fluid/pipe_flow_3d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "pipe_flow_3d.jl"))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/lid_driven_cavity_2d.jl (EDAC)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "lid_driven_cavity_2d.jl"),
                                         tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/lid_driven_cavity_2d.jl (WCSPH)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "lid_driven_cavity_2d.jl"),
                                         tspan=(0.0, 0.1), wcsph=true)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/taylor_green_vortex_2d.jl (EDAC)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "taylor_green_vortex_2d.jl"),
                                         tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/taylor_green_vortex_2d.jl (WCSPH)" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "taylor_green_vortex_2d.jl"),
                                         tspan=(0.0, 0.1), wcsph=true)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/sphere_surface_tension_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "sphere_surface_tension_2d.jl"))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/periodic_array_of_cylinders_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "periodic_array_of_cylinders_2d.jl"),
                                         tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/sphere_surface_tension_3d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "sphere_surface_tension_3d.jl"))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "fluid/falling_water_spheres_2d.jl" begin
        surface_tension_models = Dict(
            "SurfaceTensionAkinci" => SurfaceTensionAkinci(surface_tension_coefficient=0.05),
            "SurfaceTensionMorris" => SurfaceTensionMorris(surface_tension_coefficient=0.05),
            "SurfaceTensionMomentumMorris" => SurfaceTensionMomentumMorris(surface_tension_coefficient=0.05),
            "SurfaceTensionNone" => nothing  # For cases without surface tension
        )

        for (model_name, surface_tension) in surface_tension_models
            @testset "$model_name" begin
                println("═"^100)
                println("Running falling_water_spheres_2d.jl with $model_name")

                # Prepare keyword arguments
                kwargs = model_name == "SurfaceTensionNone" ?
                         (surface_tension=nothing,) :
                         (surface_tension=surface_tension,)

                # Execute the example script with the current surface tension model
                @trixi_test_nowarn trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "falling_water_spheres_2d.jl");
                                                 tspan=(0, 0.1), kwargs...)

                # Assert that the simulation ran successfully
                @test sol.retcode == ReturnCode.Success

                # Optionally, verify no unexpected RHS allocations
                @test count_rhs_allocations(sol, semi) == 0
            end
        end
    end

    @trixi_testset "fluid/falling_water_spheres_3d.jl" begin
        surface_tension_models = Dict(
            "SurfaceTensionAkinci" => SurfaceTensionAkinci(surface_tension_coefficient=0.05),
            "SurfaceTensionMorris" => SurfaceTensionMorris(surface_tension_coefficient=0.05),
            "SurfaceTensionMomentumMorris" => SurfaceTensionMomentumMorris(surface_tension_coefficient=0.05),
            "SurfaceTensionNone" => nothing  # For cases without surface tension
        )

        for (model_name, surface_tension) in surface_tension_models
            @testset "$model_name" begin
                println("═"^100)
                println("Running falling_water_spheres_3d.jl with $model_name")

                # Prepare keyword arguments
                kwargs = model_name == "SurfaceTensionNone" ?
                         (surface_tension=nothing,) :
                         (surface_tension=surface_tension,)

                # Execute the example script with the current surface tension model
                @trixi_test_nowarn trixi_include(@__MODULE__,
                                                 joinpath(examples_dir(), "fluid",
                                                          "falling_water_spheres_3d.jl");
                                                 tspan=(0, 0.05),
                                                 fluid_particle_spacing=0.01,
                                                 kwargs...) [
                    # Optional: Add regex patterns to ignore specific warnings or logs
                    r"┌ Info: The desired tank length in x-direction .*\n",
                    r"└ New tank length in x-direction.*\n",
                    r"┌ Info: The desired tank length in y-direction .*\n",
                    r"└ New tank length in y-direction.*\n",
                    r"┌ Info: The desired tank length in z-direction .*\n",
                    r"└ New tank length in z-direction.*\n"
                ]

                # Assert that the simulation ran successfully
                @test sol.retcode == ReturnCode.Success

                # Optionally, verify no unexpected RHS allocations
                @test count_rhs_allocations(sol, semi) == 0
            end
        end
    end

    @trixi_testset "fluid/sphere_surface_tension_wall_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fluid",
                                                  "sphere_surface_tension_wall_2d.jl"))
    end

    @trixi_testset "fluid/moving_wall_2d.jl" begin
        @trixi_test_nowarn trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                         joinpath(examples_dir(), "fluid",
                                                  "moving_wall_2d.jl"))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    include("dam_break_2d_corrections.jl")

    @testset "`SymplecticPositionVerlet`" begin
        @testset "2D unstable" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fluid",
                                                      "dam_break_2d.jl"),
                                             tspan=(0, 0.1), sol=nothing) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n"]

            sol = solve(ode, SymplecticPositionVerlet(),
                        dt=1.0, # This is overwritten by the stepsize callback
                        save_everystep=false, callback=callbacks)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0

            # Unstable with this CFL
            @test maximum(abs, sol.u[end]) > 2^15
        end

        @testset "2D stable" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fluid",
                                                      "dam_break_2d.jl"),
                                             tspan=(0, 0.1), sol=nothing,
                                             cfl=0.25) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n"]

            sol = solve(ode, SymplecticPositionVerlet(),
                        dt=1.0, # This is overwritten by the stepsize callback
                        save_everystep=false, callback=callbacks)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0

            # Stable with this CFL
            @test maximum(abs, sol.u[end]) < 2^15
        end

        @testset "3D" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(examples_dir(), "fluid",
                                                      "dam_break_3d.jl"),
                                             fluid_particle_spacing=0.1,
                                             tspan=(0, 0.1), sol=nothing)
            stepsize_callback = StepsizeCallback(cfl=0.65)
            callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

            sol = solve(ode, SymplecticPositionVerlet(),
                        dt=1.0, # This is overwritten by the stepsize callback
                        save_everystep=false, callback=callbacks)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
            @test maximum(abs, sol.u[end]) < 2^15
        end
    end
end
