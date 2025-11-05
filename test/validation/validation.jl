@testset verbose=true "Validation" begin
    @trixi_testset "general" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "general",
                                                  "investigate_relaxation.jl"),
                                         tspan=(0.0, 1.0))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
        # Verify number of plots
        @test plot1.n == 4
    end

    @trixi_testset "oscillating_beam_2d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "oscillating_beam_2d",
                                                  "validation_oscillating_beam_2d.jl"),
                                         tspan=(0.0, 1.0))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
        @test isapprox(error_deflection_x, 0, atol=eps())
        @test isapprox(error_deflection_y, 0, atol=eps())

        # Ignore method redefinitions from duplicate `include("../validation_util.jl")`
        @trixi_test_nowarn trixi_include(@__MODULE__,
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
        # Use `SerialUpdate()` to obtain consistent results when using multiple
        # threads and a shorter tspan to speed up CI tests.
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "dam_break_2d",
                                                  "validation_dam_break_2d.jl"),
                                         update_strategy=SerialUpdate(),
                                         tspan=(0.0, 4 / sqrt(9.81 / 0.6))) [
            r"┌ Info: The desired tank length in y-direction.*\n",
            r"└ New tank length in y-direction is set to.*\n",
            r"WARNING: Method definition max_x_coord.*\n",
            r"WARNING: Method definition interpolated_pressure.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0

        if Sys.ARCH === :aarch64
            # MacOS ARM produces slightly different pressure values than x86.
            # Note that pressure values are in the order of 1e5.
            @test isapprox(error_edac_P1, 0, atol=eps(1e5))
            @test isapprox(error_edac_P2, 0, atol=eps(1e5))
            @test isapprox(error_wcsph_P1, 0, atol=eps(1e5))
            @test isapprox(error_wcsph_P2, 0, atol=eps(1e5))
        elseif VERSION < v"1.11"
            # 1.10 produces slightly different pressure values than 1.11
            @test isapprox(error_edac_P1, 0, atol=eps(1e5))
            @test isapprox(error_edac_P2, 0, atol=eps(1e5))
            @test isapprox(error_wcsph_P1, 0, atol=eps(1e5))
            @test isapprox(error_wcsph_P2, 0, atol=eps(1e5))
        else
            # Reference values are computed with 1.11 on x86
            @test isapprox(error_wcsph_P1, 0, atol=eps())
            @test isapprox(error_wcsph_P2, 0, atol=eps())
            # Why are these errors not zero?
            @test isapprox(error_edac_P1, 0, atol=eps(1e5))
            @test isapprox(error_edac_P2, 0, atol=eps(1e5))
        end

        # Ignore method redefinitions from duplicate `include("../validation_util.jl")`
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "dam_break_2d",
                                                  "plot_pressure_sensors.jl")) [
            r"WARNING: Method definition linear_interpolation.*\n",
            r"WARNING: Method definition interpolated_mse.*\n",
            r"WARNING: Method definition extract_number_from_filename.*\n"
        ]
        # Verify number of plots
        @test length(axs_edac[1].scene.plots) >= 2
        @test length(axs_wcsph[1].scene.plots) >= 2

        # Ignore method redefinitions from duplicate `include("../validation_util.jl")`
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "dam_break_2d",
                                                  "plot_surge_front.jl")) [
            r"WARNING: Method definition linear_interpolation.*\n",
            r"WARNING: Method definition interpolated_mse.*\n",
            r"WARNING: Method definition extract_number_from_filename.*\n"
        ]
        # Verify number of plots
        @test length(axs_edac[1].scene.plots) >= 2
        @test length(axs_wcsph[1].scene.plots) >= 2
    end

    @trixi_testset "TGV_2D" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(),
                                                  "taylor_green_vortex_2d",
                                                  "validation_taylor_green_vortex_2d.jl"),
                                         tspan=(0.0, 0.01)) [
            r"WARNING: Method definition pressure_function.*\n",
            r"WARNING: Method definition initial_pressure_function.*\n",
            r"WARNING: Method definition velocity_function.*\n",
            r"WARNING: Method definition initial_velocity_function.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "LDC_2D" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(),
                                                  "lid_driven_cavity_2d",
                                                  "validation_lid_driven_cavity_2d.jl"),
                                         tspan=(0.0, 0.02), dt=0.01,
                                         SENSOR_CAPTURE_TIME=0.01) [
            r"WARNING: Method definition lid_movement_function.*\n",
            r"WARNING: Method definition is_moving.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol, semi) == 0
    end

    @trixi_testset "vortex_street_2d" begin
        @trixi_testset "Validation" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(validation_dir(),
                                                      "vortex_street_2d",
                                                      "validation_vortex_street_2d.jl"),
                                             output_directory=joinpath(validation_dir(),
                                                                       "vortex_street_2d",
                                                                       "out_test"),
                                             tspan=(0.0, 0.001))
            @test sol.retcode == ReturnCode.Success
            @test count_rhs_allocations(sol, semi) == 0
        end

        @trixi_testset "Plot" begin
            @trixi_test_nowarn trixi_include(@__MODULE__,
                                             joinpath(validation_dir(),
                                                      "vortex_street_2d",
                                                      "plot_vortex_street_2d.jl")) [
                # exact info outputs
                r"┌ Info: Strouhal number\n└\s+round\(strouhal_number, digits = 3\) = 0\.228\n",
                r"┌ Info: Fraction of the dominant frequency band in the total spectrum\n└\s+round\(integral_peak / integral_total, digits = 3\) = 0\.39\n",
                r"┌ Info: C_L_max for the periodic shedding\n└\s+round\(maximum\(f_lift_cut\), digits = 3\) = 0\.7\n",
                r"┌ Info: C_D_max for the periodic shedding\n└\s+round\(maximum\(f_drag\[start_index:end\]\), digits = 3\) = 1\.631\n"
            ]
        end
    end
end
