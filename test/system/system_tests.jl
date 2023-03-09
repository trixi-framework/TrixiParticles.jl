# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "System Tests" begin
    @testset verbose=true "Fluid" begin
        @pixie_testset "fluid/rectangular_tank_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "rectangular_tank_2d.jl"), tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fluid/dam_break_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                                       tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fluid/dam_break_3d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid", "dam_break_3d.jl"),
                                       tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fluid/falling_water_column_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "falling_water_column_2d.jl"),
                                       tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fluid/deformation_sphere_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "deformation_sphere_2d.jl"),
                                       tspan=(0.0, 3.0))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fluid/deformation_sphere_3d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "deformation_sphere_3d.jl"),
                                       tspan=(0.0, 20.0))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fluid/rectangular_tank_surface_tension_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fluid",
                                                "rectangular_tank_surface_tension_2d.jl"),
                                       tspan=(0.0, 0.5))
            @test sol.retcode == ReturnCode.Success
        end
    end

    @testset verbose=true "Solid" begin
        @pixie_testset "solid/oscillating_beam_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "solid",
                                                "oscillating_beam_2d.jl"), tspan=(0.0, 0.1))
            @test sol.retcode == ReturnCode.Success
        end
    end

    @testset verbose=true "FSI" begin
        @pixie_testset "fsi/falling_water_column_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "falling_water_column_2d.jl"),
                                       tspan=(0.0, 0.4))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fsi/dam_break_2d.jl" begin
            # Use rounded dimensions to avoid warnings
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi", "dam_break_2d.jl"),
                                       water_width=0.15,
                                       water_height=0.29,
                                       tank_width=0.58,
                                       tspan_relaxing=(0.0, 2.0),
                                       tspan=(0.0, 0.4),
                                       dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fsi/dam_break_gate_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "dam_break_gate_2d.jl"),
                                       tspan_relaxing=(0.0, 2.0),
                                       tspan=(0.0, 0.4),
                                       dtmax=1e-3)
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fsi/bending_beam_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "bending_beam_2d.jl"),
                                       n_particles_y=5)
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "fsi/falling_spheres_2d.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "fsi",
                                                "falling_spheres_2d.jl"),
                                       tspan=(0.0, 1.0))
            @test sol.retcode == ReturnCode.Success
        end
    end

    @testset verbose=true "N-Body" begin
        @pixie_testset "n_body/n_body_solar_system.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "n_body",
                                                "n_body_solar_system.jl"))
            @test sol.retcode == ReturnCode.Success
        end

        @pixie_testset "n_body/n_body_benchmark_pixie.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "n_body",
                                                "n_body_benchmark_pixie.jl"))
        end

        @pixie_testset "n_body/n_body_benchmark_reference.jl" begin
            @test_nowarn pixie_include(@__MODULE__,
                                       joinpath(examples_dir(), "n_body",
                                                "n_body_benchmark_reference.jl"))
        end

        @pixie_testset "n_body/n_body_benchmark_reference_faster.jl" begin
            @test_nowarn pixie_include(joinpath(examples_dir(), "n_body",
                                                "n_body_benchmark_reference_faster.jl"))
        end
    end
end
