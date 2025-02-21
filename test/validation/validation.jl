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

        if Sys.ARCH === :aarch64
            # MacOS ARM produces slightly different pressure values than x86.
            # Note that pressure values are in the order of 1e5.
            @test isapprox(error_edac_P1, 0, atol=8e-10)
            @test isapprox(error_edac_P2, 0, atol=7e-12)
            @test isapprox(error_wcsph_P1, 0, atol=18.0)
            @test isapprox(error_wcsph_P2, 0, atol=0.004)
        elseif VERSION == v"1.10"
            # Reference values are computed with 1.10
            @test isapprox(error_edac_P1, 0, atol=eps())
            @test isapprox(error_edac_P2, 0, atol=eps())
            @test isapprox(error_wcsph_P1, 0, atol=eps())
            @test isapprox(error_wcsph_P2, 0, atol=eps())
        else
            # 1.11 produces slightly different pressure values than 1.10.
            # Note that pressure values are in the order of 1e5.
            @test isapprox(error_edac_P1, 0, atol=eps())
            @test isapprox(error_edac_P2, 0, atol=eps())
            @test isapprox(error_wcsph_P1, 0, atol=0.07)
            @test isapprox(error_wcsph_P2, 0, atol=8e-6)
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

    @trixi_testset "hydrostatic_water_column_2d" begin
        @test_nowarn_mod trixi_include(@__MODULE__,
                                       joinpath(validation_dir(),
                                                "hydrostatic_water_column_2d",
                                                "validation.jl")) [
            r"┌ Info: The desired tank length in y-direction.*\n",
            r"└ New tank length in y-direction is set to.*\n"
        ]

        @test isapprox(errors["edac"][2], 0, atol=0.028)
        @test isapprox(errors["wcsph"][2], 0, atol=0.043)
    end
end
