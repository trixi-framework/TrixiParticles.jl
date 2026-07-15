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

    @trixi_testset "oscillating_drop_2d" begin
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(),
                                                  "oscillating_drop_2d",
                                                  "validation_oscillating_drop_2d.jl"),
                                         n_periods=1)
        @test sol.retcode == ReturnCode.Success
        @test count_rhs_allocations(sol) == 0

        filename = "validation_result_oscillating_drop_2d_dx_0p0500.json"
        result_file = joinpath("out", filename)
        reference_file = joinpath(validation_dir(), "oscillating_drop_2d", filename)

        json_result = JSON.parsefile(result_file)
        json_reference = JSON.parsefile(reference_file)

        time = json_result["kinetic_energy_fluid_1"]["time"]
        kinetic = json_result["kinetic_energy_fluid_1"]["values"]
        potential = json_result["potential_energy_fluid_1"]["values"]
        compressible = json_result["compressible_energy_fluid_1"]["values"]
        q_delta = json_result["q_delta_fluid_1"]["values"]

        kinetic_reference = json_reference["kinetic_energy_fluid_1"]["values"]
        potential_reference = json_reference["potential_energy_fluid_1"]["values"]
        compressible_reference = json_reference["compressible_energy_fluid_1"]["values"]
        q_delta_reference = json_reference["q_delta_fluid_1"]["values"]

        # The last time of the simulation is at exactly 1 period, which is not included
        # in the reference data, so we compare only up to the second last time step,
        # where the times match exactly.
        length_ = length(time) - 1

        @test isapprox(kinetic[1:length_], kinetic_reference[1:length_], atol=1e-8)
        @test isapprox(potential[1:length_], potential_reference[1:length_], atol=1e-8)
        @test isapprox(compressible[1:length_], compressible_reference[1:length_],
                       atol=1e-8)
        @test isapprox(q_delta[1:length_], q_delta_reference[1:length_], atol=1e-8)
    end
end
