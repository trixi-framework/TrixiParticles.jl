@trixi_testset "Hydrostatic correction coupling" begin
    using Base64: base64decode
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "validation_corrections.jl");
                                     resolutions=(3,), tspan=(0.0, 0.002), sample_dt=0.001,
                                     correction_cases=(:none, :shepard, :mixed, :combined,
                                                       :continuity_reinit),
                                     reinitialization_dt=0.0005, save_solution=false,
                                     output_root=mktempdir()) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    @test length(summaries) == 11
    @test sol.retcode == ReturnCode.Success
    @test all(summary -> summary.max_density_error < 0.04, summaries)
    @test all(summary -> summary.max_pressure_error < 0.1, summaries)
    @test all(summary -> summary.interface_residual < 2e-12, summaries)
    @test all(summary -> summary.min_interface_force_scale > 1e3, summaries)
    @test all(summary -> isfinite(summary.final_deflection), summaries)
    @test all(summary -> ismissing(summary.deflection_error), summaries)
    @test all(summary -> summary.eos_residual < 1e-12,
              filter(summary -> summary.method == :wcsph, summaries))
    @test all(summary -> ismissing(summary.eos_residual),
              filter(summary -> summary.method == :edac, summaries))
    @test length(order_comparisons) == 2
    @test all(result -> maximum((result.density_difference, result.pressure_difference,
                                 result.deflection_difference)) < 1e-6, order_comparisons)

    for method in methods
        raw = histories[(method, :none, 3, false)]
        shepard = histories[(method, :shepard, 3, false)]
        raw_errors = correction_series(raw, "hydrostatic_errors")["values"]
        shepard_errors = correction_series(shepard, "hydrostatic_errors")["values"]
        # Pressure must start in the prescribed hydrostatic state for both schemes.
        @test first(raw_errors)[2] < 1e-4
        @test first(shepard_errors)[2] < 1e-4
        if method == :edac
            @test first(shepard_errors)[1] < first(raw_errors)[1] - 0.003
        end
        raw_summary = only(filter(summary -> summary.method == method &&
                                             summary.correction == :none,
                                  summaries))
        mixed_summary = only(filter(summary -> summary.method == method &&
                                               summary.correction == :mixed,
                                    summaries))
        @test mixed_summary.initial_affine_gradient_error < 1e-12
        @test mixed_summary.initial_affine_gradient_error <
              raw_summary.initial_affine_gradient_error / 100
        @test mixed_summary.initial_momentum_error_interior <
              raw_summary.initial_momentum_error_interior

        forward = histories[(method, :combined, 3, false)]
        reversed = histories[(method, :combined, 3, true)]
        for name in ("hydrostatic_errors", "plate_deflection")
            a = correction_series(forward, name)
            b = correction_series(reversed, name)
            @test a["time"] == b["time"]
            @test reduce(hcat, a["values"])≈reduce(hcat, b["values"]) rtol=1e-6 atol=1e-10
        end

        forces = correction_series(forward, "interface_force_balance")["values"]
        # A nonzero part of the interface reaction is carried by clamped particles.
        @test maximum(force -> abs(force[6]), forces) > 1.0

        reinitialized = only(filter(summary -> summary.method == method &&
                                               summary.correction ==
                                               (method == :wcsph ? :shepard :
                                                :continuity_reinit),
                                    summaries))
        @test reinitialized.last_reinitialization >= 0.001
        @test reinitialized.last_reinitialization ≈ last(tspan)
        @test reinitialized.reinitialization_count == 4
        @test reinitialized.reinitialization_volume_residual < 1e-12
        @test reinitialized.frozen_volume_difference > 1e-9
        @test reinitialized.evolved_density_change > 1e-9
        if method == :wcsph
            @test reinitialized.post_reinitialization_eos_residual < 1e-12
        else
            @test reinitialized.edac_reinitialization_pressure_change == 0
        end
        @test 0 < reinitialized.final_density_error < 0.001
    end

    saved_summaries = CSV.read(summary_file, DataFrame)
    @test saved_summaries.interface_residual == getproperty.(summaries, :interface_residual)
    @test isfile(joinpath(output_root, "parameters.json"))
    @test JSON.parsefile(joinpath(output_root, "parameters.json"))["validation_version"] ==
          2
    # Old campaign directories must not be included when absent from the new summary manifest.
    stale_directory = joinpath(output_root, "wcsph_continuity_reinit_3")
    mkpath(stale_directory)
    cp(joinpath(output_root, "wcsph_combined_3",
                "validation_run_hydrostatic_corrections.json"),
       joinpath(stale_directory, "validation_run_hydrostatic_corrections.json"))
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "validation_structure.jl");
                                     resolutions=(3, 5), output_directory=mktempdir()) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    structure_control_file = csv_file
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "plot_correction_results.jl");
                                     output_directory=output_root, operator_file=nothing,
                                     structure_file=structure_control_file) [
        r"GKS: cannot open display - headless operation mode active\n"
    ]
    @test report.coupled_runs == length(summaries)
    @test report.structural_rows == 2
    @test occursin("no fluid or time integration", report.html)
    @test Set(keys(report.figures)) == Set(["coupled"])
    @test length(report.figures["coupled"].subplots) == 4
    @test CorrectionValidationPlots.Plots.xlims(report.figures["coupled"].subplots[3]) ==
          tspan
    @test all(index -> CorrectionValidationPlots.Plots.xlims(report.figures["coupled"].subplots[index]) ==
                       (0.5, length(summaries) + 0.5), (1, 2, 4))
    @test count(!ismissing, report.order_differences) == 4
    @test maximum(skipmissing(report.order_differences)) < 1e-6
    @test occursin("analytical plate deflection", report.html)
    @test length(collect(eachmatch(r"<img ", report.html))) == 1
    @test read(report_file, String) == report.html
    encoded = match(r"download='hydrostatic_correction_histories.json' href='data:application/json;base64,([^']+)'",
                    report.html).captures[1]
    embedded = JSON.parse(String(base64decode(encoded)))
    @test length(embedded) == length(histories)
    expected = Dict("$(key[1])_$(key[2])_$(key[3])" *
                    (key[4] ? "_reversed" : "") => history
                    for (key, history) in histories)
    @test Dict(run["case"] => run["history"] for run in embedded) == expected
    encoded = match(r"download='hydrostatic_structure.csv' href='data:text/csv;base64,([^']+)'",
                    report.html).captures[1]
    @test isequal(CSV.read(IOBuffer(base64decode(encoded)), DataFrame),
                  CSV.read(structure_control_file, DataFrame))
    encoded = match(r"download='hydrostatic_structure_parameters.json' href='data:application/json;base64,([^']+)'",
                    report.html).captures[1]
    @test JSON.parse(String(base64decode(encoded)))["resolutions"] == [3, 5]
    old_directory = mktempdir()
    open(joinpath(old_directory, "parameters.json"), "w") do io
        JSON.print(io, (; validation_version=1))
    end
    @test_throws r"outdated hydrostatic reference data" CorrectionValidationPlots.build_report(;
                                                                                               hydrostatic_directories=(old_directory,))
    @test !any(endswith(".png"), readdir(output_root))
end

@trixi_testset "Summation-density coupled correction matrix" begin
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "validation_summation_density.jl");
                                     resolutions=(3,), tspan=(0.0, 0.002), sample_dt=0.001,
                                     reinitialization_dt=0.0005, output_root=mktempdir()) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    @test length(summaries) == 12
    @test sol.retcode == ReturnCode.Success
    @test all(row -> row.max_density_error < 0.04, summaries)
    @test all(row -> row.max_pressure_error < (row.method == :wcsph ? 1.5 : 0.1), summaries)
    @test all(row -> row.interface_residual < 2e-12, summaries)
    @test all(row -> row.min_interface_force_scale > 1e3, summaries)
    @test all(row -> row.eos_residual < 1e-12,
              filter(row -> row.method == :wcsph, summaries))
    @test all(row -> ismissing(row.eos_residual),
              filter(row -> row.method == :edac, summaries))
    reference = CSV.read(joinpath(validation_dir(), "hydrostatic_water_column_2d",
                                  "validation_reference_summation_predictor.csv"),
                         DataFrame)
    for row in summaries
        if row.method == :wcsph && row.correction != :continuity_reinit
            # Reference-initialized WCSPH is checked against the analytical state;
            # the CSV preserves measurements from the earlier startup formulation.
            @test row.initial_density_error < 1e-9
            @test row.initial_pressure_error < 1e-7
        else
            expected = only(eachrow(filter(candidate -> candidate.method ==
                                                        string(row.method) &&
                                                        candidate.correction ==
                                                        string(row.correction) &&
                                                        candidate.resolution ==
                                                        row.resolution &&
                                                        candidate.reverse_order ==
                                                        row.reverse_order, reference)))
            @test row.initial_density_error≈expected.initial_density_error rtol=1e-6 atol=1e-12
            @test row.initial_pressure_error≈expected.initial_pressure_error rtol=1e-6 atol=1e-12
        end
    end
    settings = JSON.parsefile(joinpath(output_root, "parameters.json"))
    for method in methods
        for correction in (:none, :shepard, :mixed, :combined)
            @test settings["cases"]["$(method)_$(correction)_3"]["density_calculator"] ==
                  "SummationDensity"
        end
        raw = only(filter(row -> row.method == method && row.correction == :none,
                          summaries))
        shepard = only(filter(row -> row.method == method && row.correction == :shepard,
                              summaries))
        if method == :edac
            @test shepard.initial_density_error < raw.initial_density_error - 0.003
        else
            @test raw.initial_density_error < 1e-9
            @test shepard.initial_density_error < 1e-9
        end
        reinit = only(filter(row -> row.method == method &&
                                    row.correction == :continuity_reinit, summaries))
        @test reinit.reinitialization_count == 4
        @test reinit.reinitialization_volume_residual < 1e-12
        @test reinit.frozen_volume_difference > 1e-9
        @test reinit.evolved_density_change > 1e-9
        if method == :wcsph
            @test reinit.post_reinitialization_eos_residual < 1e-12
        else
            @test reinit.edac_reinitialization_pressure_change == 0
            pressure = correction_series(histories[(method, :continuity_reinit, 3, false)],
                                         "hydrostatic_errors")["values"]
            @test abs(last(pressure)[4] - first(pressure)[4]) > 1e-3
        end
        forward = histories[(method, :combined, 3, false)]
        reversed = histories[(method, :combined, 3, true)]
        for name in ("hydrostatic_errors", "plate_deflection")
            a, b = correction_series(forward, name), correction_series(reversed, name)
            @test a["time"] == b["time"]
            @test reduce(hcat, a["values"])≈reduce(hcat, b["values"]) rtol=1e-6 atol=1e-10
        end
    end
end

@trixi_testset "Hydrostatic reference and correction controls" begin
    using LinearAlgebra: norm
    include(joinpath(validation_dir(), "hydrostatic_water_column_2d", "correction_util.jl"))
    balances = Dict{Symbol, Any}()
    for (name, fluid_correction, boundary_correction) in
        ((:none, nothing, nothing),
         (:fluid_only, MixedKernelGradientCorrection(), nothing),
         (:mixed, MixedKernelGradientCorrection(), MixedKernelGradientCorrection()))
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(examples_dir(), "fsi",
                                                  "hydrostatic_water_column_2d.jl");
                                         use_edac=true, n_particles_plate_y=3,
                                         fluid_density_calculator=ContinuityDensity(),
                                         density_correction=nothing,
                                         fluid_gradient_correction=fluid_correction,
                                         boundary_gradient_correction=boundary_correction,
                                         parallelization_backend=SerialBackend(),
                                         sol=nothing) [
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]
        fluid = only(filter(system -> system isa TrixiParticles.AbstractFluidSystem,
                            ode.p.semi.systems))
        @test TrixiParticles.correction_gradient(fluid.correction) === fluid_correction
        for system in filter(system -> !(system isa TrixiParticles.AbstractFluidSystem),
                   ode.p.semi.systems)
            @test TrixiParticles.correction_gradient(system.boundary_model.correction) ===
                  boundary_correction
        end
        global hydrostatic_surface_height = maximum(view(tank.fluid.coordinates, 2, :)) +
                                            fluid_particle_spacing / 2
        @test hydrostatic_reference(fluid, hydrostatic_surface_height).pressure == 0
        @test hydrostatic_reference(fluid, maximum(view(tank.fluid.coordinates, 2, :))).pressure ≈
              fluid_density * gravity * fluid_particle_spacing / 2
        balances[name] = initial_hydrostatic_balance(ode, fluid)
    end
    @test balances[:mixed].initial_affine_gradient_error < 1e-12
    @test balances[:mixed].initial_momentum_error_interior < 1e-10
    @test balances[:none].initial_affine_gradient_error > 1e-3

    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(examples_dir(), "fsi",
                                              "hydrostatic_water_column_2d.jl");
                                     use_edac=false, n_particles_plate_y=3,
                                     parallelization_backend=SerialBackend(), sol=nothing) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    fluid = only(filter(system -> system isa WeaklyCompressibleSPHSystem,
                        ode.p.semi.systems))
    global hydrostatic_surface_height = maximum(view(tank.fluid.coordinates, 2, :)) +
                                        fluid_particle_spacing / 2
    for depth in (0.1, 0.5, 1.5)
        y = hydrostatic_surface_height - depth
        reference = hydrostatic_reference(fluid, y)
        @test fluid.state_equation(reference.density) ≈ reference.pressure rtol=1e-11
        step = 1e-4
        derivative = (hydrostatic_reference(fluid, y-step).pressure -
                      hydrostatic_reference(fluid, y+step).pressure) / (2step)
        @test derivative ≈ gravity * reference.density rtol=1e-8
    end
end

@trixi_testset "Corrected EDAC hydrostatic pressure and plate" begin
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "validation_corrections.jl");
                                     resolutions=(3,), methods=(:edac,),
                                     correction_cases=(:none, :mixed),
                                     check_reverse_order=false,
                                     tspan=(0.0, 0.35), sample_dt=0.01, save_solution=false,
                                     averaging_interval=(0.25, 0.35),
                                     output_root=mktempdir()) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    @test sol.retcode == ReturnCode.Success
    baseline = only(filter(row -> row.correction == :none, summaries))
    corrected = only(filter(row -> row.correction == :mixed, summaries))
    @test corrected.final_pressure_error < 0.01
    @test corrected.final_pressure_error < 0.75baseline.final_pressure_error
    @test corrected.max_pressure_error <= 1.05baseline.max_pressure_error
    @test corrected.deflection_error < 0.05
    @test corrected.interface_residual < 2e-12
end
