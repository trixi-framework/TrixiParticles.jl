@trixi_testset "Scheme-density cross-product plan" begin
    include(joinpath(validation_dir(), "hydrostatic_water_column_2d", "refinement_util.jl"))
    jobs = HydrostaticRefinement.plan_jobs(mktempdir())
    protocol = Dict("check_reverse_order" => true)
    @test length(jobs) == 160
    @test length(unique(job["id"] for job in jobs)) == 160
    @test sum(length(HydrostaticRefinement.case_names(job, protocol)) for job in jobs) ==
          176
    affected = only(filter(job -> job["method"]=="wcsph" &&
                                  job["density_mode"]=="summation" &&
                                  job["correction"]=="shepard" && job["resolution"]==3,
                           jobs))
    @test !HydrostaticRefinement.compatible_density_boundary_policy(Dict(), affected)
    @test HydrostaticRefinement.compatible_density_boundary_policy(Dict("density_boundary_policy"=>"raw-pressure-predictor"),
                                                                   affected)
    unaffected = only(filter(job -> job["method"]=="wcsph" &&
                                    job["density_mode"]=="summation" &&
                                    job["correction"]=="none" && job["resolution"]==3,
                             jobs))
    @test HydrostaticRefinement.compatible_density_boundary_policy(Dict(), unaffected)
    @test !HydrostaticRefinement.compatible_initialization_policy(Dict(), affected)
    @test !HydrostaticRefinement.compatible_initialization_policy(Dict(), unaffected)
    @test !HydrostaticRefinement.compatible_pressure_boundary_policy(Dict(), affected)
    @test HydrostaticRefinement.compatible_pressure_boundary_policy(Dict("wcsph_extrapolated_pressure_gradient" => "fluid-sided"),
                                                                    affected)
    edac_job = only(filter(job -> job["method"] == "edac" &&
                                  job["density_mode"] == "summation" &&
                                  job["correction"] == "none" && job["resolution"] == 3,
                           jobs))
    @test HydrostaticRefinement.compatible_pressure_boundary_policy(Dict(), edac_job)
    @test HydrostaticRefinement.compatible_initialization_policy(Dict("wcsph_summation_initialization" => "operator-offset-hydrostatic"),
                                                                 affected)
    for method in ("edac", "wcsph"), mode in ("continuity", "summation")
        block = filter(job -> job["method"] == method && job["density_mode"] == mode, jobs)
        @test length(block) == 40
        @test Set(job["resolution"] for job in block) == Set([3, 5, 9, 17])
        @test Set(job["correction"] for job in block) ==
              Set(string.(HydrostaticRefinement.HCM.MATRIX_CASES))
    end
end

@trixi_testset "Alternate hydrostatic density formulations" begin
    for (method, mode) in ((:wcsph, :summation), (:edac, :continuity))
        @trixi_test_nowarn trixi_include(@__MODULE__,
                                         joinpath(validation_dir(),
                                                  "hydrostatic_water_column_2d",
                                                  "validation_corrections.jl");
                                         resolutions=(3,), methods=(method,),
                                         wcsph_density_mode=mode, edac_density_mode=mode,
                                         correction_cases=(:none, :kernel, :gradient,
                                                           :blended, :mixed,
                                                           :shepard, :shepard_kernel,
                                                           :shepard_gradient,
                                                           :shepard_blended, :combined),
                                         tspan=(0.0, 0.00002),
                                         averaging_interval=(0.00001, 0.00002),
                                         sample_dt=0.00001, reinitialization_dt=0.00001,
                                         save_solution=false, output_root=mktempdir()) [
            r"\[ Info: To create the self-interaction neighborhood search.*\n"
        ]
        @test length(summaries) == 11
        @test all(row -> row.base_density_mode == mode && row.density_mode == mode,
                  summaries)
        @test all(row -> row.interface_residual < 2e-12, summaries)
        @test length(order_comparisons) == 1
        settings = JSON.parsefile(joinpath(output_root, "parameters.json"))
        expected = mode == :continuity ? "ContinuityDensity" : "SummationDensity"
        for row in summaries
            name = "$(method)_$(row.correction)_3" * (row.reverse_order ? "_reversed" : "")
            case = settings["cases"][name]
            spec = HydrostaticCorrectionMethods.definition(row.correction)
            @test occursin(expected, case["density_calculator"])
            @test case["reinitialize_density"] == (mode == :continuity && spec.shepard)
            @test !case["pressure_clipping"]
            @test case["viscosity"] == "nothing"
            if mode == :continuity && spec.shepard
                @test row.reinitialization_count == 2
                @test row.edac_reinitialization_pressure_change == 0
            else
                @test row.reinitialization_count == 0
            end
        end
    end
end
