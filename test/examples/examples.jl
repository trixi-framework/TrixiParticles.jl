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
                                       relaxation_tspan=(0.0, 0.1),
                                       simulation_tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/dam_break_2d_corrections.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "dam_break_2d_corrections.jl"),
                                       relaxation_tspan=(0.0, 0.1),
                                       simulation_tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fluid/dam_break_3d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid", "dam_break_3d.jl"),
                                       tspan=(0.0, 0.1))
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
                                       water_width=0.15,
                                       water_height=0.29,
                                       tank_width=0.58,
                                       tspan_relaxing=(0.0, 2.0),
                                       tspan=(0.0, 0.4),
                                       dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
        end

        @trixi_testset "fsi/dam_break_gate_2d.jl" begin
            @test_nowarn trixi_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "dam_break_gate_2d.jl"),
                                       tspan_relaxing=(0.0, 2.0),
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
