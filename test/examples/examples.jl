# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "Examples" begin
    @testset verbose=true "Fluid" begin
        @trixi_testset "fluid/hydrostatic_water_column_2d.jl" begin
            # Import variables into scope
            trixi_include(@__MODULE__,
                          joinpath(examples_dir(), "fluid",
                                   "hydrostatic_water_column_2d.jl"),
                          fluid_system=nothing, sol=nothing, semi=nothing, ode=nothing)

            # Neighborhood search for `FullGridCellList` test below
            search_radius = TrixiParticles.compact_support(smoothing_kernel,
                                                           smoothing_length)
            min_corner = minimum(tank.boundary.coordinates, dims=2) .- search_radius
            max_corner = maximum(tank.boundary.coordinates, dims=2) .+ search_radius
            cell_list = TrixiParticles.PointNeighbors.FullGridCellList(; min_corner,
                                                                       max_corner)
            semi_fullgrid = Semidiscretization(fluid_system, boundary_system,
                                               neighborhood_search=GridNeighborhoodSearch{2}(;
                                                                                             cell_list))

            hydrostatic_water_column_tests = Dict(
                "WCSPH default" => (),
                "WCSPH with FullGridCellList" => (semi=semi_fullgrid,),
                "WCSPH with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1e-4),),
                "WCSPH with SummationDensity" => (fluid_density_calculator=SummationDensity(),
                                                  clip_negative_pressure=true),
                "WCSPH with ViscosityAdami" => (
                                                # from 0.02*10.0*1.2*0.05/8
                                                viscosity=ViscosityAdami(nu=0.0015),),
                "WCSPH with ViscosityMorris" => (
                                                 # from 0.02*10.0*1.2*0.05/8
                                                 viscosity=ViscosityMorris(nu=0.0015),),
                "WCSPH with ViscosityAdami and SummationDensity" => (
                                                                     # from 0.02*10.0*1.2*0.05/8
                                                                     viscosity=ViscosityAdami(nu=0.0015),
                                                                     fluid_density_calculator=SummationDensity(),
                                                                     clip_negative_pressure=true),
                "WCSPH with ViscosityMorris and SummationDensity" => (
                                                                      # from 0.02*10.0*1.2*0.05/8
                                                                      viscosity=ViscosityMorris(nu=0.0015),
                                                                      fluid_density_calculator=SummationDensity(),
                                                                      clip_negative_pressure=true),
                "WCSPH with smoothing_length=1.3" => (smoothing_length=1.3,),
                "WCSPH with SchoenbergQuarticSplineKernel" => (smoothing_length=1.1,
                                                               smoothing_kernel=SchoenbergQuarticSplineKernel{2}()),
                "WCSPH with SchoenbergQuinticSplineKernel" => (smoothing_length=1.1,
                                                               smoothing_kernel=SchoenbergQuinticSplineKernel{2}()),
                "WCSPH with WendlandC2Kernel" => (smoothing_length=3.0,
                                                  smoothing_kernel=WendlandC2Kernel{2}()),
                "WCSPH with WendlandC4Kernel" => (smoothing_length=3.5,
                                                  smoothing_kernel=WendlandC4Kernel{2}()),
                "WCSPH with WendlandC6Kernel" => (smoothing_length=4.0,
                                                  smoothing_kernel=WendlandC6Kernel{2}()),
                "EDAC with source term damping" => (source_terms=SourceTermDamping(damping_coefficient=1e-4),
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

                    @test_nowarn_mod trixi_include(@__MODULE__,
                                                   joinpath(examples_dir(), "fluid",
                                                            "hydrostatic_water_column_2d.jl");
                                                   kwargs...)

                    @test sol.retcode == ReturnCode.Success
                    @test count_rhs_allocations(sol, semi) == 0
                end
            end
        end

        @trixi_testset "fluid/oscillating_drop_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "oscillating_drop_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            # This error varies between serial and multithreaded runs
            @test isapprox(error_A, 0.0, atol=1.73e-4)
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/hydrostatic_water_column_3d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "hydrostatic_water_column_3d.jl"),
                                           tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/hydrostatic_water_column_3d.jl with SummationDensity" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "hydrostatic_water_column_3d.jl"),
                                           tspan=(0.0, 0.1),
                                           fluid_density_calculator=SummationDensity(),
                                           clip_negative_pressure=true)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/accelerated_tank_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__, tspan=(0.0, 0.5),
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
                "with KernelAbstractions" => (data_type=Array,),
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
                                                sound_speed=100.0)
            )

            for (test_description, kwargs) in dam_break_tests
                @testset "$test_description" begin
                    println("═"^100)
                    println("$test_description")

                    @test_nowarn_mod trixi_include(@__MODULE__,
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
        end

        @trixi_testset "fluid/dam_break_oil_film_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "dam_break_oil_film_2d.jl"),
                                           tspan=(0.0, 0.05)) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/dam_break_3d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "dam_break_3d.jl"),
                                           tspan=(0.0, 0.1), fluid_particle_spacing=0.1)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/falling_water_column_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "falling_water_column_2d.jl"),
                                           tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/periodic_channel_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "periodic_channel_2d.jl"),
                                           tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/pipe_flow_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                           joinpath(examples_dir(), "fluid",
                                                    "pipe_flow_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/lid_driven_cavity_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "lid_driven_cavity_2d.jl"),
                                           tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/taylor_green_vortex_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "taylor_green_vortex_2d.jl"),
                                           tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/sphere_surface_tension_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "sphere_surface_tension_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/periodic_array_of_cylinders_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "periodic_array_of_cylinders_2d.jl"),
                                           tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/sphere_surface_tension_3d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "sphere_surface_tension_3d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/falling_water_spheres_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "falling_water_spheres_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/falling_water_spheres_3d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "falling_water_spheres_3d.jl")) [
                r"┌ Info: The desired tank length in x-direction .*\n",
                r"└ New tank length in x-direction.*\n",
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n",
                r"┌ Info: The desired tank length in z-direction .*\n",
                r"└ New tank length in z-direction.*\n"
            ]
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/sphere_surface_tension_wall_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "sphere_surface_tension_wall_2d.jl"))
        end

        @trixi_testset "fluid/moving_wall_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__, tspan=(0.0, 0.5),
                                           joinpath(examples_dir(), "fluid",
                                                    "moving_wall_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        include("dam_break_2d_corrections.jl")
    end

    @testset verbose=true "Solid" begin
        @trixi_testset "solid/oscillating_beam_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "solid",
                                                    "oscillating_beam_2d.jl"),
                                           tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end
    end

    @testset verbose=true "FSI" begin
        @trixi_testset "fsi/falling_water_column_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fsi",
                                                    "falling_water_column_2d.jl"),
                                           tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/dam_break_plate_2d.jl" begin
            # Use rounded dimensions to avoid warnings
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fsi",
                                                    "dam_break_plate_2d.jl"),
                                           initial_fluid_size=(0.15, 0.29),
                                           tspan=(0.0, 0.4),
                                           dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fsi",
                                                    "dam_break_gate_2d.jl"),
                                           tspan=(0.0, 0.4),
                                           dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fsi/falling_spheres_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fsi",
                                                    "falling_spheres_2d.jl"),
                                           tspan=(0.0, 1.0))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end
    end

    @testset verbose=true "N-Body" begin
        @trixi_testset "n_body/n_body_solar_system.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "n_body",
                                                    "n_body_solar_system.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "n_body/n_body_benchmark_trixi.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "n_body",
                                                    "n_body_benchmark_trixi.jl")) [
                r"WARNING: Method definition interact!.*\n"
            ]
        end

        @trixi_testset "n_body/n_body_benchmark_reference.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "n_body",
                                                    "n_body_benchmark_reference.jl"))
        end

        @trixi_testset "n_body/n_body_benchmark_reference_faster.jl" begin
            @test_nowarn_mod trixi_include(joinpath(examples_dir(), "n_body",
                                                    "n_body_benchmark_reference_faster.jl"))
        end
    end

    @testset verbose=true "Postprocessing" begin
        @trixi_testset "postprocessing/interpolation_plane.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "postprocessing",
                                                    "interpolation_plane.jl"),
                                           tspan=(0.0, 0.01)) [
                r"WARNING: importing deprecated binding Makie.*\n",
                r"WARNING: using deprecated binding Colors.*\n",
                r"WARNING: using deprecated binding PlotUtils.*\n",
                r"WARNING: Makie.* is deprecated.*\n",
                r"  likely near none:1\n",
                r", use .* instead.\n"
            ]
            @test sol.retcode == ReturnCode.Success
        end
        @trixi_testset "postprocessing/interpolation_point_line.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "postprocessing",
                                                    "interpolation_point_line.jl"))
            @test sol.retcode == ReturnCode.Success
        end
        @trixi_testset "postprocessing/postprocessing.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(),
                                                    "postprocessing",
                                                    "postprocessing.jl"))
            @test sol.retcode == ReturnCode.Success
        end
    end
end

@testset verbose=true "DEM" begin
    @trixi_testset "dem/rectangular_tank_2d.jl" begin
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "dem",
                                                "rectangular_tank_2d.jl"),
                                       tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end
end
