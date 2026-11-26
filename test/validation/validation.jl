@testset verbose=true "Validation" begin
    @trixi_testset "poiseuille_carreau_2d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "poiseuille_carreau_2d",
                                                  "validation_poiseuille_carreau_2d.jl"),
                                         nu0=40.0, reynolds_number=0.05,
                                         ny=8, t_end_factor=0.0002,
                                         relative_l2_error_bounds=Dict(1.0 => 0.06,
                                                                       0.5 => 0.06),
                                         n_values=(1.0, 0.5), output_root=mktempdir()) [
            r"WARNING: Method definition linear_interpolation_clamped.*\n",
            r"WARNING: Method definition carreau_yasuda_kinematic_viscosity.*\n",
            r"WARNING: Method definition solve_shear_rate_from_stress.*\n",
            r"WARNING: Method definition analytical_ux_profile.*\n",
            r"WARNING: Method definition velocity_profile_errors.*\n",
            r"WARNING: Method definition newtonian_ux.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0
        @test all(isfinite, values(final_relative_l2_errors))
        @test all(error <= relative_l2_error_bounds[n]
                  for (n, error) in final_relative_l2_errors)
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "poiseuille_carreau_2d",
                                                  "plot_carreau_comparison.jl"),
                                         output_directory=output_root) [
            r"WARNING: Method definition linear_interpolation_clamped.*\n",
            r"WARNING: Method definition carreau_yasuda_kinematic_viscosity.*\n",
            r"WARNING: Method definition solve_shear_rate_from_stress.*\n",
            r"WARNING: Method definition analytical_ux_profile.*\n",
            r"WARNING: Method definition velocity_profile_errors.*\n",
            r"WARNING: Method definition newtonian_ux.*\n",
            r"GKS: cannot open display - headless operation mode active\n"
        ]
        @test profile_plot.n == 4
        @test error_plot.n == 2
        @test !any(endswith(file, ".png") for (_, _, files) in walkdir(output_root)
                   for file in files)
    end

    @trixi_testset "general" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "general",
                                                  "investigate_relaxation.jl"),
                                         tspan=(0.0, 1.0))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0
        # Verify number of plots
        @test plot1.n == 4
    end

    @trixi_testset "oscillating_beam_2d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "oscillating_beam_2d",
                                                  "validation_oscillating_beam_2d.jl"),
                                         tspan=(0.0, 1.0)) [
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        if VERSION < v"1.12"
            # Older Julia versions produce allocations because `get_neighborhood_search`
            # is not type-stable with TLSPH.
            @test count_rhs_allocations(sol) < 200
        else
            @test count_rhs_allocations(sol) == 0
        end
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
        @test count_rhs_allocations(sol) == 0

        # Note that pressure values are in the order of 1e5
        @test isapprox(error_wcsph_P1, 0, atol=eps(1e5))
        @test isapprox(error_wcsph_P2, 0, atol=eps(1e5))
        @test isapprox(error_edac_P1, 0, atol=eps(1e5))
        @test isapprox(error_edac_P2, 0, atol=eps(1e5))

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

    @trixi_testset "hydrostatic_water_column_2d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(),
                                                  "hydrostatic_water_column_2d",
                                                  "validation.jl"), tspan=(0.0, 0.35),
                                         n_particles_plate_y=3) [
            r"┌ Info: The desired tank length in y-direction.*\n",
            r"└ New tank length in y-direction is set to.*\n",
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]

        # We compare the relative error to the analytical solution
        @test isapprox(errors[:edac][2], 0.0, atol=0.033)
        @test isapprox(errors[:wcsph][2], 0.0, atol=0.045)
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
        @test count_rhs_allocations(sol) == 0
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
        @test count_rhs_allocations(sol) == 0
    end

    @trixi_testset "vortex_street_2d" begin
        # Use a very coarse resolution and a short `tspan` to keep the runtime low.
        # At this resolution, the force coefficients are too noisy to extract a meaningful
        # shedding frequency, but the velocity sensor in the wake of the cylinder still
        # yields a clean oscillation, so we only validate the Strouhal number from `v_y`.
        # Use `SerialUpdate()` to obtain consistent results when using multiple threads.
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "vortex_street_2d",
                                                  "validation_vortex_street_2d.jl"),
                                         resolution_factor=0.2, tspan=(0.0, 5.0),
                                         update_strategy=SerialUpdate()) [
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0

        # The vortex shedding is fully developed after about `t = 1.5`.
        # Starting the analysis at `t = 2` leaves about six shedding cycles.
        # Allow the `@info` output of the computed quantities.
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "vortex_street_2d",
                                                  "plot_vortex_street_reference_2d.jl"),
                                         directory="out",
                                         file_prefix="validation_result_vortex_street",
                                         resolution_factor=0.2, t_start=2.0) [
            r"┌ Info: .*\n",
            r"└ .*\n"
        ]

        # For this coarse resolution and short analysis time, even small differences
        # in the floating point results between platforms can shift the shedding frequency.
        # Use a large tolerance to avoid cross-platform CI failures.
        @test isapprox(strouhal_number, 0.2, atol=0.04)

        # `argmax` always returns a frequency, so also check that the spectrum has a clear
        # peak. The current value is about 0.55.
        @test band_fraction_v_y > 0.5

        # Check the drag and lift computation with the `ThrustCalculator`.
        # The output interval is `dt = 0.02`, so `tspan = (0.0, 5.0)` yields
        # 5.0 / 0.02 + 1 = 251 samples.
        @test length(drag_coefficients) == length(times) == 251
        @test length(lift_coefficients) == length(times)
        @test all(isfinite, drag_coefficients)
        @test all(isfinite, lift_coefficients)

        # The current mean drag coefficient is about 1.14. The value at the coarse
        # resolution is sensitive to small floating point differences, so use loose bounds.
        @test 0.5 < drag_coefficient_mean < 1.5

        # The lift oscillates around zero due to the symmetry of the setup.
        # The current mean lift coefficient is about -0.02.
        @test abs(mean(lift_coefficients_cut)) < 0.05
    end

    @trixi_testset "vortex_street_2d (reference data)" begin
        # Analyze the committed reference data with the default parameters.
        # No simulation is involved, so the results are deterministic and we can use
        # tight tolerances.
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "vortex_street_2d",
                                                  "plot_vortex_street_reference_2d.jl")) [
            r"┌ Info: .*\n",
            r"└ .*\n"
        ]

        @test isapprox(strouhal_number, 0.21398002853066556, rtol=1e-8)
        @test isapprox(drag_coefficient_mean, 1.456938198711208, rtol=1e-8)
        @test isapprox(lift_amplitude, 0.6721642375613147, rtol=1e-8)
        @test isapprox(band_fraction_v_y, 0.42087449922368775, rtol=1e-8)
    end
end
