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

    @trixi_testset "poiseuille_flow_2d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "poiseuille_flow_2d",
                                                  "validation_poiseuille_flow_2d.jl"),
                                         tspan=(0.0, 0.04))
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0

        # Results are not bitwise reproducible with a different number of threads.
        # Note that velocity values are in the order of 1e-3.
        @test isapprox(error_v_x, 0, atol=1e-16)

        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "poiseuille_flow_2d",
                                                  "plot_poiseuille_flow_2d.jl")) [
            r"GKS: cannot open display - headless operation mode active\n"
        ]
        # Verify number of plots
        @test length(p_rmsep.series_list) == 2
        @test length(p.series_list) == 10
    end

    @trixi_testset "poiseuille_flow_3d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "poiseuille_flow_3d",
                                                  "validation_poiseuille_flow_3d.jl"),
                                         particle_spacing_factor=10,
                                         tspan=(0.0, 0.02)) [
            r"┌ Info: .*edge 2 length.*\n",
            r"└ New edge 2 length.*\n",
            r"┌ Warning: .*boundary face.*\n",
            r"└ @ TrixiParticles .*boundary_zones\.jl:\d+\n"
        ]
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0

        # Results are not bitwise reproducible with a different number of threads.
        # Note that velocity values are in the order of 1e-3.
        @test isapprox(error_v_x, 0, atol=1e-16)

        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(), "poiseuille_flow_3d",
                                                  "plot_poiseuille_flow_3d.jl"),
                                         particle_spacing_factor=10) [
            r"┌ Info: .*edge 2 length.*\n",
            r"└ New edge 2 length.*\n",
            r"┌ Warning: .*boundary face.*\n",
            r"└ @ TrixiParticles .*boundary_zones\.jl:\d+\n",
            r"GKS: cannot open display - headless operation mode active\n"
        ]
        # Verify number of plots
        @test length(p_rmsep.series_list) == 2
        @test length(p.series_list) == 12
    end
end
