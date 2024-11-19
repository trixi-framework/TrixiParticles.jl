@testset verbose=true "Validation" begin
    @trixi_testset "general" begin
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(validation_dir(), "general",
                                                "investigate_relaxation.jl"),
                                       tspan=(0.0, 1.0))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
        # Verify number of plots
        @test plot1.n == 4
    end

    @trixi_testset "oscillating_beam_2d" begin
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(validation_dir(), "oscillating_beam_2d",
                                                "validation_oscillating_beam_2d.jl"),
                                       tspan=(0.0, 1.0))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
        @test isapprox(error_deflection_x, 0, atol=eps())
        @test isapprox(error_deflection_y, 0, atol=eps())

        # Ignore method redefinitions from duplicate `include("../validation_util.jl")`
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(validation_dir(), "oscillating_beam_2d",
                                                "plot_oscillating_beam_results.jl")) [
            r"WARNING: Method definition linear_interpolation.*\n",
            r"WARNING: Method definition interpolated_mse.*\n",
            r"WARNING: Method definition extract_number_from_filename.*\n",
            r"WARNING: Method definition extract_resolution_from_filename.*\n",
            r"WARNING: importing deprecated binding Makie.*\n",
            r"WARNING: Makie.* is deprecated.*\n",
            r"  likely near none:1\n",
            r", use .* instead.\n"
        ]
        # Verify number of plots
        @test length(ax1.scene.plots) >= 6
    end

    @trixi_testset "dam_break_2d" begin
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(validation_dir(), "dam_break_2d",
                                                "validation_dam_break_2d.jl")) [
            r"┌ Info: The desired tank length in y-direction.*\n",
            r"└ New tank length in y-direction is set to.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0

        if VERSION == v"1.10"
            @test isapprox(error_edac_P1, 0, atol=eps())
            @test isapprox(error_edac_P2, 0, atol=eps())
            @test isapprox(error_wcsph_P1, 0, atol=eps())
            @test isapprox(error_wcsph_P2, 0, atol=eps())
        else
            # 1.9 causes a large difference in the solution
            # TODO 1.11 requires a performance hotfix which will likely change these results again
            @test isapprox(error_edac_P1, 0, atol=4e-9)
            @test isapprox(error_edac_P2, 0, atol=3e-11)
            @test isapprox(error_wcsph_P1, 0, atol=26.3)
            @test isapprox(error_wcsph_P2, 0, atol=8.2e-3)
        end

        # Ignore method redefinitions from duplicate `include("../validation_util.jl")`
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(validation_dir(), "dam_break_2d",
                                                "plot_dam_break_results.jl")) [
            r"WARNING: Method definition linear_interpolation.*\n",
            r"WARNING: Method definition interpolated_mse.*\n",
            r"WARNING: Method definition extract_number_from_filename.*\n",
            r"WARNING: Method definition extract_resolution_from_filename.*\n"
        ]
        # Verify number of plots
        @test length(axs_edac[1].scene.plots) >= 2
    end
end
