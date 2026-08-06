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

    @trixi_testset "surface tension" begin
        include(joinpath(validation_dir(), "surface_tension_common.jl"))

        laplace_2d = SurfaceTensionValidation.young_laplace_operator_fit(2, 100)
        laplace_3d = SurfaceTensionValidation.young_laplace_operator_fit(3, 905)
        rayleigh_coarse = SurfaceTensionValidation.rayleigh_mode2_stiffness(200;
                                                                            stretch=1.04)
        rayleigh_medium = SurfaceTensionValidation.rayleigh_mode2_stiffness(400;
                                                                            stretch=1.04)

        @test laplace_2d.relative_error < 0.06
        @test laplace_3d.relative_error < 0.02
        @test laplace_2d.total_force < 1.0e-12
        @test laplace_3d.total_force < 1.0e-12
        @test rayleigh_medium.frequency_error < 0.05
        @test rayleigh_medium.frequency_error < rayleigh_coarse.frequency_error

        scorecard = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                      "contact_angle_scorecard.csv"), DataFrame)
        @test size(scorecard, 1) == 2
        @test all(scorecard.static_eligible)
        @test !any(scorecard.eligible)
        @test only(scorecard[scorecard.mechanism .== "geometric", :response_passes]) == 1
        @test only(scorecard[scorecard.mechanism .== "contact_line_force",
                             :response_passes]) == 2

        normal_diagnostics = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                               "contact_angle_normal_components.csv"),
                                      DataFrame)
        fluid_middle = normal_diagnostics[(normal_diagnostics.variant .== "fluid_only") .& (normal_diagnostics.requested_particles .== 1500) .& (normal_diagnostics.target .== normal_diagnostics.initial_angle),
                                          :]
        @test maximum(fluid_middle.mean_error) > 5
        baseline_middle = normal_diagnostics[(normal_diagnostics.variant .== "baseline_total") .& (normal_diagnostics.requested_particles .== 1500) .& (normal_diagnostics.target .== normal_diagnostics.initial_angle),
                                             :]
        @test count(baseline_middle.corrected_cross_error_4x .<= 0.2) == 3

        line_normalization = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                               "contact_line_normalization.csv"),
                                      DataFrame)
        line_middle = line_normalization[line_normalization.cells_per_h .== 4, :]
        @test size(line_normalization, 1) == 150
        @test all(line_middle.coarea_middle_pass)
        @test !all(line_middle.coarea_endpoint_decreasing)
        @test !any(line_middle[line_middle.angle .!= 90,
                               :divergence_middle_pass])
        wendland_middle = line_middle[line_middle.kernel .== "wendland_c2", :]
        @test count(wendland_middle.wedge_middle_pass) == 2
        @test count(wendland_middle.gated_middle_pass) == 2

        cap_transfer = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                         "contact_line_cap_transfer.csv"), DataFrame)
        compatible_middle = cap_transfer[(cap_transfer.variant .== "compatible_indicator") .& (cap_transfer.requested_particles .== 1500),
                                         :]
        @test size(cap_transfer, 1) == 90
        @test all(compatible_middle.middle_pass)
        @test !any(compatible_middle.endpoint_decreasing)

        wetted_area = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                        "wetted_area_measure.csv"), DataFrame)
        wetted_middle = wetted_area[wetted_area.requested_particles .== 1500, :]
        @test count(wetted_middle.middle_pass) == 4
        @test count(wetted_middle.endpoint_decreasing) == 4

        recovery_comparison = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                "contact_angle_recovery_comparison.csv"),
                                       DataFrame)
        @test !any(recovery_comparison.eligible)

        measure_protocol = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                             "contact_measure_protocol.csv"), DataFrame)
        @test size(measure_protocol, 1) == 30
        @test all(measure_protocol.protocol_pass)
        protocol_middle = measure_protocol[(measure_protocol.series .== "production_resolution") .& (measure_protocol.requested_particles .== 1500),
                                           :]
        @test all(protocol_middle.middle_pass .& protocol_middle.endpoint_pass)

        corrected_wetted_area = CSV.read(joinpath(validation_dir(),
                                                  "surface_tension_3d",
                                                  "wetted_area_corrected.csv"), DataFrame)
        corrected_middle = corrected_wetted_area[corrected_wetted_area.requested_particles .== 1500,
                                                 :]
        @test all(corrected_middle.middle_pass .& corrected_middle.endpoint_pass)
        @test maximum(corrected_middle.corrected_area_error) < 0.06
        @test only(corrected_middle[corrected_middle.target .== 150.0,
                                    :corrected_area_error]) < 0.02

        extended_recovery = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                              "contact_angle_recovery_extended.csv"),
                                     DataFrame)
        @test size(extended_recovery, 1) == 60
        compatible_geometry_middle = extended_recovery[(extended_recovery.variant .== "compatible_geometry_wall") .& (extended_recovery.requested_particles .== 1500),
                                                       :]
        young_middle = extended_recovery[(extended_recovery.variant .== "young_color_boundary") .& (extended_recovery.requested_particles .== 1500),
                                         :]
        @test count(compatible_geometry_middle.middle_pass) == 5
        @test count(compatible_geometry_middle.endpoint_pass) == 2
        @test count(young_middle.middle_pass) == 5
        @test count(young_middle.endpoint_pass) == 5
        @test count(young_middle.angle_middle_pass) == 2
        @test count(young_middle.angle_endpoint_pass) == 1

        extended_signs = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                           "contact_angle_force_sign_extended.csv"),
                                  DataFrame)
        corrected_signs = extended_signs[extended_signs.variant .== "corrected_wetted_area",
                                         :]
        geometry_signs = extended_signs[extended_signs.variant .== "compatible_geometry_wall",
                                        :]
        young_signs = extended_signs[extended_signs.variant .== "young_color_boundary", :]
        @test count(corrected_signs.sign_pass) == 4
        @test all(skipmissing(corrected_signs.wall_zero_at_90))
        @test count(geometry_signs.sign_pass) == 4
        @test !any(geometry_signs.measure_eligible)
        @test count(young_signs.sign_pass) == 3
        @test !any(young_signs.static_eligible)

        extended_comparison = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                "contact_angle_recovery_extended_comparison.csv"),
                                       DataFrame)
        @test count(extended_comparison.eligible_for_dynamics) == 1
        @test only(extended_comparison[extended_comparison.method .== "R7-W corrected wetted-area energy",
                                       :eligible_for_dynamics])

        r4_static = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                      "contact_angle_static_r4_wetted_area.csv"), DataFrame)
        @test size(r4_static, 1) == 9
        @test all(r4_static.stage_pass)
        @test count(r4_static[r4_static.kind .== "force_sign", :total_sign_pass]) == 4
        @test maximum(r4_static[r4_static.kind .== "energy_gradient",
                                :gradient_relative_error]) <= 1.0e-5
        @test all(r4_static[r4_static.target .== 90, :zero_at_90])

        r4_initial = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                       "contact_angle_perturbation_r4_wetted_area.csv"),
                              DataFrame)
        r4_initial_candidates = r4_initial[r4_initial.mechanism .== "r4_wetted_area", :]
        @test count(r4_initial_candidates.response_pass) == 2
        r4_classified = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                          "contact_angle_perturbation_r4_wetted_area_classified.csv"),
                                 DataFrame)
        r4_classified_candidates = r4_classified[r4_classified.mechanism .== "r4_wetted_area",
                                                 :]
        @test count(r4_classified_candidates.formulation_response_pass) == 3
        @test count(r4_classified_candidates.effective_acceleration_toward_target) == 4
        @test all(r4_classified_candidates[r4_classified_candidates.target .== 90,
                                           :control_equivalent])

        r4_extended = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                        "contact_angle_perturbation_r4_wetted_area_extended.csv"),
                               DataFrame)
        r4_extended_candidates = r4_extended[r4_extended.mechanism .== "r4_wetted_area", :]
        @test count(r4_extended_candidates.formulation_response_pass) == 4
        @test all(r4_extended_candidates.reaction_pass)

        r4_threshold = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                         "contact_angle_threshold_r4_wetted_area.csv"),
                                DataFrame)
        r4_timestep = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                        "contact_angle_timestep_r4_wetted_area.csv"),
                               DataFrame)
        r4_cost = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                    "contact_angle_cost_r4_wetted_area.csv"), DataFrame)
        r4_active_cost = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                           "contact_angle_cost_r4_wetted_area_active.csv"),
                                  DataFrame)
        @test size(r4_threshold, 1) == 5 && all(r4_threshold.pass)
        @test size(r4_timestep, 1) == 2 && all(r4_timestep.pass)
        @test size(r4_cost, 1) == 6
        @test size(r4_active_cost, 1) == 6
        @test all(r4_cost[r4_cost.mechanism .== "r4_wetted_area",
                          :contact_cache_bytes] .> 0)
        @test all(r4_active_cost.target .== 60)

        r4_selected = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                        "contact_angle_selected_matrix_r4_wetted_area.csv"),
                               DataFrame)
        r4_sensitivity = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                           "contact_angle_sensitivity_r4_wetted_area.csv"),
                                  DataFrame)
        @test size(r4_selected, 1) == 15 && all(r4_selected.pass)
        @test maximum(abs, r4_selected.final_error) <= 5
        @test maximum(r4_selected.max_total_momentum_residual) <= 1.0e-12
        @test size(r4_sensitivity, 1) == 4 && all(r4_sensitivity.pass)
        @test only(unique(r4_sensitivity.angle_span)) <= 1

        production_static = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                              "contact_angle_static_r4_wetted_area_production.csv"),
                                     DataFrame)
        @test size(production_static, 1) == 9 && all(production_static.stage_pass)
        @test maximum(production_static[production_static.kind .== "energy_gradient",
                                        :gradient_relative_error]) <= 1.0e-5
        production_initial = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                               "contact_angle_perturbation_r4_wetted_area_production.csv"),
                                      DataFrame)
        production_initial_candidates = production_initial[production_initial.mechanism .== "wetted_area_production",
                                                           :]
        @test count(production_initial_candidates.formulation_response_pass) == 3
        production_extended = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                "contact_angle_perturbation_r4_wetted_area_extended_production.csv"),
                                       DataFrame)
        production_extended_candidates = production_extended[production_extended.mechanism .== "wetted_area_production",
                                                             :]
        @test size(production_extended_candidates, 1) == 4
        @test all(production_extended_candidates.formulation_response_pass)

        production_threshold = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                 "contact_angle_threshold_r4_wetted_area_production.csv"),
                                        DataFrame)
        production_timestep = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                "contact_angle_timestep_r4_wetted_area_production.csv"),
                                       DataFrame)
        @test size(production_threshold, 1) == 5 && all(production_threshold.pass)
        @test size(production_timestep, 1) == 2 && all(production_timestep.pass)
        production_cost = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                            "contact_angle_cost_r4_wetted_area_production.csv"),
                                   DataFrame)
        production_active_cost = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                   "contact_angle_cost_r4_wetted_area_active_production.csv"),
                                          DataFrame)
        @test size(production_cost, 1) == 6
        @test size(production_active_cost, 1) == 6
        production_control_median = median(production_active_cost[production_active_cost.mechanism .== "none",
                                                                  :solver_runtime])
        production_active_median = median(production_active_cost[production_active_cost.mechanism .== "wetted_area_production",
                                                                 :solver_runtime])
        @test production_active_median / production_control_median <= 1.2

        production_selected = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                "contact_angle_selected_matrix_r4_wetted_area_production.csv"),
                                       DataFrame)
        production_sensitivity = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                                   "contact_angle_sensitivity_r4_wetted_area_production.csv"),
                                          DataFrame)
        @test size(production_selected, 1) == 15 && all(production_selected.pass)
        @test maximum(production_selected.max_total_momentum_residual) <= 1.0e-12
        @test maximum(production_selected.force_consistency_residual) <= 1.0e-12
        @test size(production_sensitivity, 1) == 4 && all(production_sensitivity.pass)
        @test only(unique(production_sensitivity.angle_span)) <= 1

        tensile_stability = CSV.read(joinpath(validation_dir(), "surface_tension_2d",
                                              "rayleigh_tensile_stability.csv"), DataFrame)
        @test count(tensile_stability.admissible) == 8
        @test count(tensile_stability.accepted) == 1
        shifted_rayleigh = only(eachrow(tensile_stability[tensile_stability.variant .== "particle_shifting_tangential",
                                                          :]))
        @test shifted_rayleigh.periods_completed == 1.48
        @test shifted_rayleigh.minimum_pair_ratio >= 0.5
        @test shifted_rayleigh.density_min < 980
        @test shifted_rayleigh.status == "timestep_collapse"
        shifted_rayleigh_sun2017 = only(eachrow(tensile_stability[tensile_stability.variant .== "particle_shifting_sun2017_tangential",
                                                                  :]))
        @test shifted_rayleigh_sun2017.periods_completed == 0.4
        @test shifted_rayleigh_sun2017.minimum_pair_ratio >= 0.5
        @test shifted_rayleigh_sun2017.density_min < 980
        @test shifted_rayleigh_sun2017.status == "timestep_collapse"
        selected_tic = only(eachrow(tensile_stability[tensile_stability.variant .== "interface_tic_025_sun2017_tangential",
                                                      :]))
        @test selected_tic.admissible && selected_tic.accepted
        @test selected_tic.tic_strength == 0.25
        @test !selected_tic.clip_negative_pressure
        @test selected_tic.periods_completed >= selected_tic.requested_periods
        @test selected_tic.frequency_error <= 0.05
        @test selected_tic.minimum_pair_ratio >= 0.5
        @test 980 <= selected_tic.density_min <= selected_tic.density_max <= 1020
        @test selected_tic.status == "final_time"
        tic_controls = tensile_stability[in.(tensile_stability.tic_strength,
                                             Ref([0.1, 0.5, 1.0])), :]
        @test size(tic_controls, 1) == 3
        @test !any(tic_controls.accepted)

        ghost_signs = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                        "contact_angle_force_sign_ghost_geometric.csv"),
                               DataFrame)
        wall_energy_signs = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                              "contact_angle_force_sign_wall_energy.csv"),
                                     DataFrame)
        wetted_area_signs = CSV.read(joinpath(validation_dir(), "surface_tension_3d",
                                              "contact_angle_force_sign_wetted_area.csv"),
                                     DataFrame)
        @test count(ghost_signs.sign_pass) == 2
        @test count(wall_energy_signs[wall_energy_signs.variant .== "wall_energy_1x",
                                      :sign_pass]) == 3
        @test count(wall_energy_signs[wall_energy_signs.variant .== "wall_energy_2x",
                                      :sign_pass]) == 4
        @test count(wetted_area_signs.sign_pass) == 4
        @test all(skipmissing(wetted_area_signs.wall_zero_at_90))
        @test !any(wetted_area_signs.measure_eligible)
    end
end
