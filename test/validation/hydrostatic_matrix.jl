@trixi_testset "Complete hydrostatic density-gradient matrix" begin
    using Base64: base64decode
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "validation_long.jl");
                                     resolutions=(3,), tspan=(0.0, 0.0005),
                                     averaging_interval=(0.00025, 0.0005),
                                     sample_dt=0.00025,
                                     reinitialization_dt=0.00025,
                                     output_root=mktempdir()) [
        r"\[ Info: To create the self-interaction neighborhood search.*\n"
    ]
    @test length(summaries) == 23
    @test length(order_comparisons) == 2
    @test all(row -> isfinite(row.late_density_error) && isfinite(row.late_pressure_error),
              summaries)
    @test all(row -> isfinite(row.late_deflection_span) &&
                     isfinite(row.late_deflection_drift),
              summaries)
    @test all(row -> row.interface_residual < 2e-12, summaries)
    settings = JSON.parsefile(joinpath(output_root, "parameters.json"))
    reference = CSV.read(joinpath(validation_dir(), "hydrostatic_water_column_2d",
                                  "validation_reference_long.csv"), DataFrame)
    @test nrow(reference) == 46
    for method in methods, case in HydrostaticCorrectionMethods.MATRIX_CASES
        row = only(filter(row -> row.method == method && row.correction == case &&
                                 !row.reverse_order, summaries))
        published = only(eachrow(filter(row -> row.method == string(method) &&
                                               row.correction == string(case) &&
                                               row.resolution == 3 && !row.reverse_order,
                                        reference)))
        @test row.initial_density_error≈published.initial_density_error rtol=1e-6 atol=1e-12
        @test row.initial_pressure_error≈published.initial_pressure_error rtol=1e-6 atol=1e-12
        spec = HydrostaticCorrectionMethods.definition(case)
        actual = settings["cases"]["$(method)_$(case)_3"]
        @test actual["definition"]["shepard"] == spec.shepard
        @test actual["definition"]["gradient"] == string(spec.gradient)
        @test actual["reinitialize_density"] == (method == :wcsph && spec.shepard)
        @test actual["fluid_gradient_correction"] ==
              string(hydrostatic_gradient_correction(case))
        @test actual["fluid_gradient_correction"] == actual["boundary_gradient_correction"]
        if spec.gradient in (:gradient, :mixed)
            @test row.initial_affine_gradient_error < 1e-12
        end
        if method == :wcsph && spec.shepard
            @test row.reinitialization_count == 2
            @test row.reinitialization_volume_residual < 1e-12
        end
    end

    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(),
                                              "hydrostatic_water_column_2d",
                                              "plot_correction_results.jl");
                                     output_directory=output_root, operator_file=nothing,
                                     structure_file=nothing) [
        r"GKS: cannot open display - headless operation mode active\n"
    ]
    @test report.coupled_runs == 23
    @test report.coupled_error_statistic == "late mean"
    runs, _ = CorrectionValidationPlots.load_runs((output_root,))
    groups = CorrectionValidationPlots.run_groups(runs)
    for (index, field, peak) in ((1, :late_density_error, :max_density_error),
         (2, :late_pressure_error, :max_pressure_error))
        panel = report.figures["coupled"].subplots[index]
        matrix = CorrectionValidationPlots.correction_matrix(runs, groups, field, peak)
        @test size(matrix.final) == (2, 11)
        @test count(!isempty, matrix.members) == 21
        @test sort(vcat(vec(matrix.members)...)) == collect(1:23)
        heatmap = only(filter(series -> series[:seriestype] == :heatmap, panel.series_list))
        values = hasproperty(heatmap[:z], :surf) ? heatmap[:z].surf : heatmap[:z]
        @test isequal(Matrix(values), permutedims(matrix.final))
        labels = last(CorrectionValidationPlots.Plots.yticks(panel))
        @test any(occursin("Kernel · no Shepard"), labels)
        @test any(occursin("Gradient · Shepard"), labels)
        @test any(occursin("Blend 0.5 · Shepard"), labels)
    end
    @test occursin("continuity resets every", report.html)
    @test occursin("late-window mean", report.html)
    @test occursin("bands span all forward correction methods", report.html)
    encoded = match(r"download='hydrostatic_method_definitions.json' href='data:application/json;base64,([^']+)'",
                    report.html).captures[1]
    @test length(JSON.parse(String(base64decode(encoded)))) == 11
end
