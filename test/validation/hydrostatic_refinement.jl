@trixi_testset "Hydrostatic refinement comparisons" begin
    include(joinpath(validation_dir(), "hydrostatic_water_column_2d",
                     "refinement_metrics.jl"))
    reference = CSV.read(joinpath(validation_dir(), "hydrostatic_water_column_2d",
                                  "validation_reference_long.csv"), DataFrame)
    samples = filter(row -> !row.reverse_order, reference)
    samples.spacing = 0.05 ./ (samples.resolution .- 1)
    samples.protocol = fill("matched 1-second study", size(samples, 1))
    samples.density_model = [row.method == "wcsph" ||
                             row.correction == "continuity_reinit" ?
                             "continuity" : "summation" for row in eachrow(samples)]
    refinement = HydrostaticRefinementMetrics.observed_refinement(samples)
    @test size(refinement, 1) == 63
    @test all(==(2.0), refinement.refinement_ratio)
    @test Set(refinement.metric) == Set(["late_density_error", "late_pressure_error",
                  "deflection_error"])
    deflection = filter(row -> row.metric == "deflection_error", refinement)
    # The measured n=3 coincidence must remain visible as deterioration towards n=5.
    @test all(<(0), deflection.observed_order)
    @test all(>(1), deflection.error_ratio)
    available = filter(row -> row.method == "edac" && row.correction in ("none", "mixed") &&
                              !(row.correction == "mixed" && row.resolution == 5), samples)
    common = HydrostaticRefinementMetrics.common_method_samples(available)
    @test Set(common.correction) == Set(["none"])
    @test Set(common.resolution) == Set([3, 5])
    # Do not infer refinement from incompatible protocols, even with matching case names.
    samples.protocol = string.(samples.resolution)
    @test isempty(HydrostaticRefinementMetrics.observed_refinement(samples))
end

@trixi_testset "Hydrostatic refinement worker and resume" begin
    using Base64
    include(joinpath(validation_dir(), "hydrostatic_water_column_2d", "refinement_util.jl"))
    root = mktempdir()
    options = (; resolutions=(3,), methods=(:edac,), density_modes=(:summation,),
               correction_cases=(:none,), workers=1,
               tspan=(0.0, 0.0005), averaging_interval=(0.00025, 0.0005),
               sample_dt=0.00025, reinitialization_dt=0.00025, generate_report=false)
    result = HydrostaticRefinement.run_refinement(; options..., output_root=root)
    @test result["status"] == "completed"
    @test result["expected_runs"] == 1
    @test only(result["jobs"])["status"] == "completed"
    status = HydrostaticRefinement.refinement_status(root)
    @test status.validated_runs == status.expected_runs == 1
    @test status.levels[3].validated_runs == 1
    directory = only(result["jobs"])["directory"]
    marker = joinpath(directory, "complete.json")
    modified = mtime(marker)
    original = CSV.read(joinpath(directory, "hydrostatic_corrections.csv"), DataFrame)

    resumed = HydrostaticRefinement.run_refinement(; options..., output_root=root)
    @test resumed["status"] == "completed"
    @test only(resumed["jobs"])["status"] == "reused"
    @test mtime(marker) == modified
    @test isequal(CSV.read(joinpath(only(resumed["campaign_directories"]),
                                    "hydrostatic_corrections.csv"), DataFrame), original)

    imported = HydrostaticRefinement.run_refinement(; options..., output_root=mktempdir(),
                                                    reuse_directories=(directory,))
    @test imported["status"] == "completed"
    @test only(imported["jobs"])["status"] == "reused"
    @test isequal(CSV.read(joinpath(only(imported["campaign_directories"]),
                                    "hydrostatic_corrections.csv"), DataFrame), original)
    variant_file = joinpath(directory, "diagnostic_variant.json")
    HydrostaticRefinement.write_json(variant_file, (; diagnostic=true))
    @test_throws r"diagnostic variations" HydrostaticRefinement.validate_output(directory,
                                                                                only(result["jobs"]),
                                                                                result["protocol"])
    rm(variant_file)
    @test_throws r"resume protocol differs" HydrostaticRefinement.run_refinement(;
                                                                                 options...,
                                                                                 output_root=root,
                                                                                 sample_dt=0.0001)
    @test_throws ArgumentError HydrostaticRefinement.run_refinement(; options...,
                                                                    output_root=mktempdir(),
                                                                    methods=(:unknown,))
    progress_file = joinpath(root, "refinement_progress.json")
    saved = JSON.parsefile(progress_file)
    active = Dict("status" => "running",
                  "jobs" => [Dict("status" => "running", "pid" => getpid(),
                                  "directory" => mktempdir())])
    HydrostaticRefinement.write_json(progress_file, active)
    @test_throws r"workers are still running" HydrostaticRefinement.run_refinement(;
                                                                                   options...,
                                                                                   output_root=root)
    HydrostaticRefinement.write_json(progress_file, saved)

    # A failed forward trajectory blocks its reversed control. It must remain
    # visible in the matrix and must never acquire a zero-valued error statistic.
    failed_directory = mktempdir()
    write(joinpath(failed_directory, "run.log"),
          "Running hydrostatic correction case edac_combined_3\n" *
          "ERROR: LoadError: particle coordinates are NaN or outside the domain bounds of the cell list\n" *
          "cell (1, 4) is out of bounds for cell grid of size [2:10, 2:20]\n")
    failed_job = Dict{String, Any}("id" => "edac_combined_3", "method" => "edac",
                                   "correction" => "combined", "resolution" => 3,
                                   "directory" => failed_directory, "status" => "failed")
    snapshot_root = mktempdir()
    directories = HydrostaticRefinement.assemble_campaigns!([
                                                                only(result["jobs"]),
                                                                failed_job
                                                            ],
                                                            result["protocol"],
                                                            snapshot_root;
                                                            allow_incomplete=true)
    settings = JSON.parsefile(joinpath(only(directories), "parameters.json"))
    @test !settings["campaign_complete"]
    @test settings["case_status"]["edac_combined_3"]["status"] == "failed"
    @test settings["case_status"]["edac_combined_3"]["failure_kind"] == "cell_grid_bounds"
    @test settings["case_status"]["edac_combined_3_reversed"]["status"] == "blocked"
    diagnostic_directory = mktempdir()
    diagnostic_state = Dict("integrator_time" => 0.25, "exception" => nothing,
                            "previous_state" => Dict("max_speed" => 0.2,
                                                     "min_pressure" => -1.0,
                                                     "outside_nominal_sides" => 0))
    HydrostaticRefinement.write_json(joinpath(diagnostic_directory,
                                              "diagnostic_variant.json"),
                                     (; diagnostic=true, method="edac", correction="mixed",
                                      resolution=3, grid="dictionary", final_time=1.0,
                                      purpose="diagnostic attachment test fixture"))
    HydrostaticRefinement.write_json(joinpath(diagnostic_directory, "diagnostic.json"),
                                     diagnostic_state)
    include(joinpath(validation_dir(), "corrections", "plot_util.jl"))
    report = CorrectionValidationPlots.build_report(; hydrostatic_directories=directories,
                                                    diagnostic_directories=(diagnostic_directory,))
    @test report.diagnostic_runs == 1
    encoded = match(r"download='hydrostatic_stability_diagnostics.json' href='data:application/json;base64,([^']+)'",
                    report.html).captures[1]
    diagnostics = JSON.parse(String(Base64.base64decode(encoded)))
    @test only(diagnostics)["state"] == diagnostic_state
    @test only(diagnostics)["outcome"] == "stopped_before_target"
    @test report.coupled_runs == 1
    @test report.planned_coupled_runs == 3
    @test report.incomplete_coupled_runs == 2
    @test Set(report.case_status.status) == Set(["completed", "failed", "blocked"])
    @test all(ismissing,
              filter(row -> row.status != "completed", report.case_status).final_time)
    runs, _ = CorrectionValidationPlots.load_runs(directories)
    matrix = CorrectionValidationPlots.correction_matrix(runs,
                                                         CorrectionValidationPlots.run_groups(runs),
                                                         :density_error, :max_density_error)
    column = only(findall(==("combined"), matrix.corrections))
    @test isnan(matrix.final[1, column])
    @test isempty(matrix.members[1, column])
    @test occursin("FAIL = failed trajectory", report.html)
    @test occursin("order controls are unfinished/failed, not zero residuals", report.html)

    # Identical case names must not reuse a result from a different density model.
    cross = HydrostaticRefinement.run_refinement(; options..., output_root=mktempdir(),
                                                 density_modes=(:continuity, :summation),
                                                 reuse_directories=(directory,),
                                                 execute=false)
    @test cross["expected_runs"] == 2
    @test count(job -> job["status"] == "reused", cross["jobs"]) == 1
    @test only(filter(job -> job["density_mode"] == "continuity", cross["jobs"]))["status"] ==
          "pending"
    @test length(cross["campaign_directories"]) == 2
    cross_runs,
    cross_sources = CorrectionValidationPlots.load_runs(Tuple(cross["campaign_directories"]))
    cross_groups = CorrectionValidationPlots.run_groups(cross_runs; sources=cross_sources)
    @test length(cross_runs) == 1
    @test length(cross_groups) == 2
    empty_group = only(filter(group -> group.density_mode == "continuity", cross_groups))
    @test isempty(empty_group.first_index:empty_group.last_index)
    cross_report = CorrectionValidationPlots.build_report(;
                                                          hydrostatic_directories=Tuple(cross["campaign_directories"]))
    @test cross_report.planned_coupled_runs == 2
    @test cross_report.incomplete_coupled_runs == 1

    # Reuse a terminal outcome without rerunning it, keeping the primary data empty.
    pending = only(filter(job -> job["density_mode"] == "continuity", cross["jobs"]))
    mkpath(pending["directory"])
    write(joinpath(pending["directory"], "run.log"),
          "Running hydrostatic correction case edac_none_3\nERROR: LoadError: recorded test failure\n")
    pending["status"] = "failed"
    terminal_directories = HydrostaticRefinement.assemble_campaigns!(cross["jobs"],
                                                                     cross["protocol"],
                                                                     mktempdir();
                                                                     allow_incomplete=true)
    terminal = HydrostaticRefinement.run_refinement(; options..., output_root=mktempdir(),
                                                    density_modes=(:continuity, :summation),
                                                    reuse_directories=Tuple(terminal_directories),
                                                    execute=false)
    @test count(job -> get(job, "reused_failure", false), terminal["jobs"]) == 1
    @test count(job -> job["status"] == "reused", terminal["jobs"]) == 1
    @test count(job -> job["status"] == "pending", terminal["jobs"]) == 0

    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(), "corrections",
                                              "validation_corrections.jl");
                                     resolutions=(12,), output_directory=mktempdir())
    expanded_report = CorrectionValidationPlots.build_report(; operator_file=csv_file,
                                                             hydrostatic_directories=Tuple(terminal_directories),
                                                             expanded=true)
    @test Set(keys(expanded_report.figures)) ==
          Set(["operators", "coupled", "summation", "cross_density"])
    @test all(figure -> length(figure.subplots) == 4, values(expanded_report.figures))
    @test length(collect(eachmatch(r"<section class='figure'", expanded_report.html))) == 4
    @test expanded_report.coupled_runs == 1
    @test expanded_report.planned_coupled_runs == 2
    @test occursin("Continuity-density hydrostatic benchmark", expanded_report.html)
    @test occursin("Summation-density hydrostatic benchmark", expanded_report.html)
    @test startswith(expanded_report.figures["summation"].subplots[1][:title], "I")
    @test startswith(expanded_report.figures["cross_density"].subplots[4][:title], "P")
end
