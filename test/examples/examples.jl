# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "Examples" begin
    @testset verbose=true "Fluid" begin
        @trixi_testset "fluid/oscillating_drop_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "oscillating_drop_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            # This error varies between serial and multithreaded runs
            @test isapprox(error_A, 0.0001717690010767381, atol=5e-7)
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/hydrostatic_water_column_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "hydrostatic_water_column_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/hydrostatic_water_column_2d.jl with source term damping" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "hydrostatic_water_column_2d.jl"),
                                           source_terms=SourceTermDamping(;
                                                                          damping_coefficient=1e-4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/hydrostatic_water_column_2d.jl with SummationDensity" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "hydrostatic_water_column_2d.jl"),
                                           fluid_density_calculator=SummationDensity(),
                                           clip_negative_pressure=true)
            @test sol.retcode == ReturnCode.Success
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

        @trixi_testset "fluid/hydrostatic_water_column_edac_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "hydrostatic_water_column_edac_2d.jl"))
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
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "dam_break_2d.jl"), tspan=(0.0, 0.1)) [
                r"┌ Info: The desired tank length in y-direction .*\n",
                r"└ New tank length in y-direction.*\n",
            ]
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/dam_break_2d_surface_tension.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "dam_break_2d_surface_tension.jl"),
                                       relaxation_tspan=(0.0, 0.1),
                                       simulation_tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
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

        @trixi_testset "fluid/dam_break_2d_surface_tension.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "dam_break_2d_surface_tension.jl"),
                                           tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/deformation_sphere_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "deformation_sphere_2d.jl"))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/deformation_sphere_3d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "deformation_sphere_3d.jl"),
                                           tspan=(0.0, 1.0))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/falling_water_spheres_2d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "falling_water_spheres_2d.jl"),
                                           tspan=(0.0, 0.5))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/falling_water_spheres_3d.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "falling_water_spheres_3d.jl"),
                                           tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "fluid/dam_break_2d_surface_tension.jl" begin
            @test_nowarn_mod trixi_include(@__MODULE__,
                                           joinpath(examples_dir(), "fluid",
                                                    "dam_break_2d_surface_tension.jl"),
                                           tspan=(0.0, 0.4))
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

    @trixi_testset "fluid/deformation_sphere_2d.jl" begin
        @test_nowarn trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "fluid",
                                            "deformation_sphere_2d.jl"),
                                   tspan=(0.0, 3.0))
        @test sol.retcode == ReturnCode.Success
    end

    @trixi_testset "fluid/deformation_sphere_3d.jl" begin
        @test_nowarn trixi_include(@__MODULE__,
                                   joinpath(examples_dir(), "fluid",
                                            "deformation_sphere_3d.jl"),
                                   tspan=(0.0, 20.0))
        @test sol.retcode == ReturnCode.Success
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
                                                    "n_body_benchmark_trixi.jl"))
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
                r", use .* instead.\n",
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
