@trixi_testset "Periodic correction consistency" begin
    using Base64: base64decode
    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(), "corrections",
                                              "validation_corrections.jl");
                                     resolutions=(12, 24, 48), output_directory=mktempdir())

    # From the first particle, see neighbors through every combination of periodic faces.
    for d in 1:3
        (; system, semi, u_ode) = CorrectionConvergence.setup_operator(12, nothing, Val(d))
        GC.@preserve u_ode begin
            u = TrixiParticles.wrap_u(u_ode, system, semi)
            coords = TrixiParticles.current_coordinates(u, system)
            @test size(coords) == (d, 12^d)
            wrapped = Set{NTuple{d, Bool}}()
            TrixiParticles.foreach_point_neighbor(system, system, coords, coords, semi;
                                                  points=1:1) do a, b, pos_diff, distance
                if distance < 1.8 / 12
                    push!(wrapped, Tuple(abs.(coords[:, a] - coords[:, b]) .> 0.5))
                end
            end
            @test length(wrapped) == 2^d
        end
    end

    @test all(r -> r.region == :periodic && isfinite(r.error), results)
    finest(method,
           operator) = only(r
                            for r in results
                            if r.method == method &&
                               r.operator == operator && r.resolution == 48)
    for (method, operator) in
        ((:shepard, :interpolation), (:gradient, :difference_gradient),
         (:mixed, :direct_gradient), (:shepard, :density_reinitialization),
         (:gradient, :pressure_continuity), (:mixed, :pressure_interparticle_continuity),
         (:shepard_mixed, :pressure_summation),
         (:shepard_mixed, :pressure_interparticle_summation))
        @test 1.9 < finest(method, operator).order < 2.1
    end
    @test finest(:mixed, :direct_gradient).error < finest(:none, :direct_gradient).error
    @test finest(:none, :pressure_tensile_positive).error ≈
          finest(:none, :pressure_continuity).error
    constant_results = filter(r -> startswith(string(r.operator), "constant_pressure_"),
                              results)
    @test maximum(r -> r.error, constant_results) < 1e-12

    saved = CSV.read(csv_file, DataFrame)
    @test saved.l2_error == getproperty.(results, :error)
    @test unique(saved.region) == ["periodic"]
    @test unique(saved.dimension) == [2]
    @test occursin("2D periodic case", read(report_file, String))

    @trixi_test_nowarn trixi_include(@__MODULE__,
                                     joinpath(validation_dir(), "corrections",
                                              "plot_correction_results.jl");
                                     output_directory) [r"GKS: cannot open display - headless operation mode active\n"]
    @test length(figure.subplots) == 4
    html = read(report_file, String)
    encoded = match(r"href='data:text/csv;base64,([^']+)'", html).captures[1]
    @test base64decode(encoded) == read(csv_file)

    for d in (1, 3)
        rows = CorrectionConvergence.run_convergence(dimension=d, resolutions=(12, 16))
        @test all(r -> r.dimension == d && isfinite(r.error), rows)
        for (method, operator) in
            ((:shepard, :interpolation), (:gradient, :difference_gradient),
             (:mixed, :pressure_continuity))
            row = only(r
                       for r in rows
                       if r.method == method && r.operator == operator &&
                          r.resolution == 16)
            @test 1.85 < row.order < 2.1
        end
        @test maximum(r.error
                      for r in rows if startswith(string(r.operator), "constant_pressure_")) <
              1e-12
        x = fill(0.125, d)
        exact = TrixiParticles.ForwardDiff.gradient(CorrectionConvergence.field, x)
        @test all(!iszero, exact) && CorrectionConvergence.field_gradient(x, Val(d)) ≈ exact
    end
end
