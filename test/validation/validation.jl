@testset verbose=true "Validation" begin
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

    @trixi_testset "Akinci cube-to-sphere 3D" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(),
                                                  "akinci_cube_to_sphere_3d",
                                                  "validation.jl");
                                         particles_per_dimension=3,
                                         tspan=(0.0, 2.5e-5),
                                         analysis_interval=2.5e-5,
                                         iisph_min_iterations=1,
                                         iisph_max_iterations=2,
                                         resolution_levels=(2, 3),
                                         write_results=false,
                                         print_results=false)

        @test size(shootout_results, 1) == 28
        @test Set(shootout_results.model) ==
              Set(["CohesionForceAkinci", "SurfaceTensionAkinci"])
        @test Set(shootout_results.sph_method) == Set(["wcsph", "edac", "iisph"])
        @test Set(shootout_results.pressure_formulation) ==
              Set(["density_matched", "inter_particle_averaged", "implicit"])
        @test Set(shootout_results.correction) ==
              Set(["none", "AkinciFreeSurfaceCorrection"])
        @test all(==("Success"), shootout_results.retcode)
        @test all(==(27), shootout_results.particle_count)
        @test all(==(1.0), shootout_results.smoothing_length_factor)
        @test all(isapprox(cfl, 0.2; rtol=2eps()) for cfl in shootout_results.acoustic_cfl)
        @test all(isfinite, shootout_results.initial_support_radius_cv)
        @test all(isfinite, shootout_results.final_support_radius_cv)
        @test all(isfinite, shootout_results.initial_particle_spacing_cv)
        @test all(isfinite, shootout_results.final_particle_spacing_cv)
        @test all(isfinite, shootout_results.late_time_mean_asphericity)
        @test all(isfinite, shootout_results.late_time_mean_particle_spacing_cv)
        @test all(isfinite, shootout_results.final_radial_cv)
        @test all(isfinite, shootout_results.mean_radius_ratio)
        @test all(>(0), shootout_results.kinetic_energy)
        @test all(<(1.0e-12), shootout_results.center_of_mass_drift)
        @test all(<(1.0e-12), shootout_results.momentum_norm)
        @test size(shootout_shapes, 1) == 28 * 27
        @test all(isfinite, shootout_shapes.x)
        @test all(isfinite, shootout_shapes.y)
        @test all(isfinite, shootout_shapes.z)
        @test size(shootout_time_series, 1) == 2 * 28
        @test Set(shootout_time_series.time) == Set([0.0, 2.5e-5])
        @test all(isfinite, shootout_time_series.asphericity)
        @test all(isfinite, shootout_time_series.particle_spacing_cv)
        @test size(resolution_results, 1) == 2 * 5
        @test Set(resolution_results.particles_per_dimension) == Set([2, 3])
        @test Set(resolution_results.particle_count) == Set([8, 27])
        @test all(==(1.0), resolution_results.smoothing_length_factor)
        @test length(Set(zip(resolution_results.sph_method,
                             resolution_results.density_calculator))) == 5
        @test all(==("Success"), resolution_results.retcode)
        @test all(isapprox(cfl, 0.2; rtol=2eps())
                  for cfl in resolution_results.acoustic_cfl)
        @test all(isfinite, resolution_results.final_support_radius_cv)
        @test all(isfinite, resolution_results.final_particle_spacing_cv)
        @test all(isfinite, resolution_results.late_time_mean_asphericity)
        @test all(isfinite, resolution_results.late_time_mean_particle_spacing_cv)

        cohesion_results = shootout_results[shootout_results.model .== "CohesionForceAkinci",
                                            :]
        @test any(!isapprox(cohesion, akinci; rtol=1.0e-6)
                  for (cohesion, akinci) in
                      zip(cohesion_results.kinetic_energy,
                          shootout_results[shootout_results.model .== "SurfaceTensionAkinci",
                                           :kinetic_energy]))
        @test count_rhs_allocations(sol) == 0
    end
end
