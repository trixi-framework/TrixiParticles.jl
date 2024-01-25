# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "Examples" begin
    @testset verbose=true "Fluid" begin
        @trixi_testset "fluid/rectangular_tank_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "rectangular_tank_2d.jl"), tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/rectangular_tank_2d.jl with SummationDensity" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "rectangular_tank_2d.jl"), tspan=(0.0, 0.1),
                                       fluid_density_calculator=SummationDensity(),
                                       clip_negative_pressure=true)
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/rectangular_tank_edac_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "rectangular_tank_edac_2d.jl"),
                                       tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/dam_break_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                                       tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/dam_break_2d_corrections.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "dam_break_2d_corrections.jl"),
                                       tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/dam_break_2d_surface_tension.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid", "dam_break_2d_surface_tension.jl"),
                                       relaxation_tspan=(0.0, 0.1),
                                       simulation_tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/dam_break_3d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid", "dam_break_3d.jl"),
                                       tspan=(0.0, 0.1), fluid_particle_spacing=0.1)
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/falling_water_column_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "falling_water_column_2d.jl"),
                                       tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/periodic_channel_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "periodic_channel_2d.jl"),
                                       tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
        end
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
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "solid",
                                                "oscillating_beam_2d.jl"), tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end
    end

    @testset verbose=true "FSI" begin
        @trixi_testset "fsi/falling_water_column_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "falling_water_column_2d.jl"),
                                       tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fsi/dam_break_2d.jl" begin
            # Use rounded dimensions to avoid warnings
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi", "dam_break_2d.jl"),
                                       initial_fluid_size=(0.15, 0.29),
                                       tspan=(0.0, 0.4),
                                       dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "dam_break_gate_2d.jl"),
                                       tspan=(0.0, 0.4),
                                       dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fsi/falling_spheres_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "falling_spheres_2d.jl"),
                                       tspan=(0.0, 1.0))
            @test sol.retcode == ReturnCode.Success
        end
    end

    @testset verbose=true "N-Body" begin
        @trixi_testset "n_body/n_body_solar_system.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "n_body",
                                                "n_body_solar_system.jl"))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "n_body/n_body_benchmark_trixi.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "n_body",
                                                "n_body_benchmark_trixi.jl"))
        end

        @trixi_testset "n_body/n_body_benchmark_reference.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "n_body",
                                                "n_body_benchmark_reference.jl"))
        end

        @trixi_testset "n_body/n_body_benchmark_reference_faster.jl" begin
            @test_nowarn trixi_include(joinpath(examples_dir(), "n_body",
                                                "n_body_benchmark_reference_faster.jl"))
        end
    end
end
