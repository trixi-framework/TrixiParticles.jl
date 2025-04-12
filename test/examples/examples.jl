# Smoke tests, i.e., tests to verify that examples are running without crashing,
# but without checking the correctness of the solution.
@testset verbose=true "Examples" begin
    # include("examples_fluid.jl")

    # @testset verbose=true "Solid" begin
    #     @trixi_testset "solid/oscillating_beam_2d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "solid",
    #                                                 "oscillating_beam_2d.jl"),
    #                                        tspan=(0.0, 0.1))
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end
    # end

    # @testset verbose=true "FSI" begin
    #     @trixi_testset "fsi/falling_water_column_2d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "fsi",
    #                                                 "falling_water_column_2d.jl"),
    #                                        tspan=(0.0, 0.4))
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end

    #     @trixi_testset "fsi/dam_break_plate_2d.jl" begin
    #         # Use rounded dimensions to avoid warnings
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "fsi",
    #                                                 "dam_break_plate_2d.jl"),
    #                                        initial_fluid_size=(0.15, 0.29),
    #                                        tspan=(0.0, 0.4),
    #                                        dtmax=1e-3)
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end

    #     @trixi_testset "fsi/dam_break_gate_2d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "fsi",
    #                                                 "dam_break_gate_2d.jl"),
    #                                        tspan=(0.0, 0.4),
    #                                        dtmax=1e-3)
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end

    #     @trixi_testset "fsi/falling_spheres_2d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "fsi",
    #                                                 "falling_spheres_2d.jl"),
    #                                        tspan=(0.0, 1.0))
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end
    # end

    # @testset verbose=true "N-Body" begin
    #     @trixi_testset "n_body/n_body_solar_system.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "n_body",
    #                                                 "n_body_solar_system.jl"))
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end

    #     @trixi_testset "n_body/n_body_benchmark_trixi.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "n_body",
    #                                                 "n_body_benchmark_trixi.jl")) [
    #             r"WARNING: Method definition interact!.*\n"
    #         ]
    #     end

    #     @trixi_testset "n_body/n_body_benchmark_reference.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "n_body",
    #                                                 "n_body_benchmark_reference.jl"))
    #     end

    #     @trixi_testset "n_body/n_body_benchmark_reference_faster.jl" begin
    #         @test_nowarn_mod trixi_include(joinpath(examples_dir(), "n_body",
    #                                                 "n_body_benchmark_reference_faster.jl"))
    #     end
    # end

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

    # @testset verbose=true "Preprocessing" begin
    #     @trixi_testset "preprocessing/packing_2d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "preprocessing",
    #                                                 "packing_2d.jl"))
    #         @test sol.retcode == ReturnCode.Terminated
    #     end
    #     @trixi_testset "preprocessing/packing_3d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "preprocessing",
    #                                                 "packing_3d.jl"))
    #     end
    # end

    # @testset verbose=true "DEM" begin
    #     @trixi_testset "dem/rectangular_tank_2d.jl" begin
    #         @test_nowarn_mod trixi_include(@__MODULE__,
    #                                        joinpath(examples_dir(), "dem",
    #                                                 "rectangular_tank_2d.jl"),
    #                                        tspan=(0.0, 0.1))
    #         @test sol.retcode == ReturnCode.Success
    #         @test count_rhs_allocations(sol, semi) == 0
    #     end
    # end
end
