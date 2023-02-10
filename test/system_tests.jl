# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset "System Tests" begin
    @testset "Fluid" begin
        @test_nowarn pixie_include(joinpath(examples_dir(), "fluid",
                                            "rectangular_tank_2d.jl"), tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success

        @test_nowarn pixie_include(joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
                                   tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success

        @test_nowarn pixie_include(joinpath(examples_dir(), "fluid", "dam_break_3d.jl"),
                                   tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success

        @test_nowarn pixie_include(joinpath(examples_dir(), "fluid",
                                            "falling_water_column_2d.jl"), tspan=(0.0, 0.4))
        @test sol.retcode == ReturnCode.Success
    end

    @testset "Solid" begin
        @test_nowarn pixie_include(joinpath(examples_dir(), "solid",
                                            "oscillating_beam_2d.jl"), tspan=(0.0, 0.1))
        @test sol.retcode == ReturnCode.Success
    end

    @testset "FSI" begin
        @test_nowarn pixie_include(joinpath(examples_dir(), "fsi",
                                            "falling_water_column_2d.jl"), tspan=(0.0, 0.4))
        @test sol.retcode == ReturnCode.Success

        # Use rounded dimensions to avoid warnings
        @test_nowarn pixie_include(joinpath(examples_dir(), "fsi", "dam_break_2d.jl"),
                                   water_width=0.15,
                                   water_height=0.29,
                                   container_width=0.58,
                                   tspan_relaxing=(0.0, 2.0),
                                   tspan=(0.0, 0.4),
                                   dtmax=1e-3)
        @test sol.retcode == ReturnCode.Success

        @test_nowarn pixie_include(joinpath(examples_dir(), "fsi", "dam_break_gate_2d.jl"),
                                   tspan_relaxing=(0.0, 2.0),
                                   tspan=(0.0, 0.4),
                                   dtmax=1e-3)
        @test sol.retcode == ReturnCode.Success

        @test_nowarn pixie_include(joinpath(examples_dir(), "fsi", "bending_beam_2d.jl"),
                                   n_particles_y=5)
        @test sol.retcode == ReturnCode.Success
    end
end
