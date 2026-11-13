@trixi_testset "Correction operator scaling" begin
    using Base64: base64decode
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(), "corrections",
                                              "validation_corrections.jl");
                                     resolutions=(12, 24, 48), output_directory=mktempdir())
    @test all(result -> isfinite(result.error), results)

    function finest(method, operator, region)
        return last(filter(result -> result.method == method &&
                                     result.operator == operator &&
                                     result.region == region,
                           results))
    end

    raw_interpolation_boundary = finest(:none, :interpolation, :boundary)
    shepard_interpolation_boundary = finest(:shepard, :interpolation, :boundary)
    raw_difference_boundary = finest(:none, :difference_gradient, :boundary)
    gradient_difference_boundary = finest(:gradient, :difference_gradient, :boundary)
    blended_difference_boundary = finest(:blended, :difference_gradient, :boundary)
    mixed_difference_boundary = finest(:mixed, :difference_gradient, :boundary)
    raw_direct_boundary = finest(:none, :direct_gradient, :boundary)
    kernel_direct_boundary = finest(:kernel, :direct_gradient, :boundary)
    mixed_direct_boundary = finest(:mixed, :direct_gradient, :boundary)

    @test shepard_interpolation_boundary.order > raw_interpolation_boundary.order + 0.9
    @test gradient_difference_boundary.order > raw_difference_boundary.order + 0.9
    @test mixed_difference_boundary.order > raw_difference_boundary.order + 0.9
    @test kernel_direct_boundary.order > raw_direct_boundary.order + 0.9
    @test mixed_direct_boundary.order > kernel_direct_boundary.order + 0.9
    @test blended_difference_boundary.error < raw_difference_boundary.error

    shepard_interpolation_interior = finest(:shepard, :interpolation, :interior)
    gradient_difference_interior = finest(:gradient, :difference_gradient, :interior)
    mixed_difference_interior = finest(:mixed, :difference_gradient, :interior)
    mixed_direct_interior = finest(:mixed, :direct_gradient, :interior)
    raw_density_boundary = finest(:none, :summation_density, :boundary)
    shepard_density_boundary = finest(:shepard, :summation_density, :boundary)
    reinitialized_density_boundary = finest(:shepard, :density_reinitialization, :boundary)
    reinitialized_density_interior = finest(:shepard, :density_reinitialization, :interior)

    @test shepard_interpolation_interior.order > 1.8
    @test gradient_difference_interior.order > 1.8
    @test mixed_difference_interior.order > 1.8
    @test mixed_direct_interior.order > 1.8
    @test shepard_density_boundary.error < raw_density_boundary.error
    @test abs(shepard_density_boundary.order) < 0.1
    @test reinitialized_density_boundary.order > 0.9
    @test reinitialized_density_interior.order > 1.8

    pressure_operators = (:pressure_summation,
                          :pressure_interparticle_summation,
                          :pressure_continuity,
                          :pressure_interparticle_continuity)
    for operator in pressure_operators
        @test finest(:gradient, operator, :interior).order > 1.8
        @test finest(:mixed, operator, :interior).order > 1.8
    end
    for operator in (:pressure_summation, :pressure_interparticle_summation)
        @test finest(:shepard_mixed, operator, :interior).order > 1.8
    end

    constant_pressure_results = filter(results) do result
        startswith(string(result.operator), "constant_pressure_") &&
            result.region == :interior && result.resolution == 48
    end
    @test !isempty(constant_pressure_results)
    @test maximum(result -> result.error, constant_pressure_results) < 1e-7

    for region in (:boundary, :interior)
        tensile = finest(:none, :pressure_tensile_positive, region)
        continuity = finest(:none, :pressure_continuity, region)
        @test tensile.error ≈ continuity.error rtol = 5e-13
    end

    saved_results = CSV.read(csv_file, DataFrame)
    @test saved_results.l2_error == getproperty.(results, :error)
    @test saved_results.resolution == getproperty.(results, :resolution)
    @test occursin("not full-discretization convergence orders", read(report_file, String))

    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(), "corrections",
                                              "plot_correction_results.jl");
                                     output_directory, hydrostatic_directories=()) [
        r"GKS: cannot open display - headless operation mode active\n"
    ]
    @test Set(keys(report.figures)) == Set(["operators"])
    @test length(report.figures["operators"].subplots) == 4
    @test sum(envelope -> envelope.count, report.operator_envelopes) == length(results)
    @test all(eachrow(saved_results)) do row
        matches = filter(envelope -> envelope.method == row.method &&
                                     envelope.region == row.region &&
                                     row.operator in envelope.operators,
                         report.operator_envelopes)
        length(matches) == 1 || return false
        envelope = only(matches)
        index = only(findall(==(row.resolution), envelope.resolution))
        return envelope.lower[index] <= row.l2_error <= envelope.upper[index]
    end
    @test report.operator_rows == length(results)
    @test occursin("zero acceleration", report.html)
    @test occursin("production one-pass density can retain a floor", report.html)
    @test !occursin(r"(?:src|href)='https?://", report.html)
    @test read(report_file, String) == report.html
    encoded = match(r"download='correction_convergence.csv' href='data:text/csv;base64,([^']+)'",
                    report.html).captures[1]
    embedded = CSV.read(IOBuffer(base64decode(encoded)), DataFrame)
    @test embedded.l2_error == saved_results.l2_error
    @test embedded.operator == saved_results.operator
    for (order, name) in ((1, "O(h) guide"), (2, "O(h²) guide"))
        reference = only(filter(series -> series[:label] == name,
                                report.figures["operators"].subplots[1].series_list))
        x, y = reference[:x], reference[:y]
        @test log(first(y) / last(y)) / log(last(x) / first(x)) ≈ order
    end
    @test length(collect(eachmatch(r"<img ", report.html))) == 1
    @test !any(endswith(".png"), readdir(output_directory))
end
