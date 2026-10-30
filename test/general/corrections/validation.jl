@testset "Analytical operator scaling" begin
    include(joinpath(validation_dir(), "corrections", "convergence.jl"))
    results = CorrectionConvergence.run_convergence(; resolutions=(12, 24, 48))
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
end
