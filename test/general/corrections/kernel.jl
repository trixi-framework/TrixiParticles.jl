@testset "Kernel correction" begin
    for edac in (false, true)
        setup = correction_setup(KernelCorrection(); edac,
                                 pressure_acceleration=nothing)
        update_correction!(setup)

        @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
        @test all(isfinite, setup.system.cache.dw_gamma)
    end

    for perturbation in (false, true)
        setup = update_correction!(correction_setup(KernelCorrection(); perturbation))
        moments = correction_moments(setup)
        @test maximum(abs, moments.zeroth_gradient_moment) < 2e-12
    end

    density32 = fill(1000.0f0, 4)
    mass32 = fill(10.0f0, 4)
    state_equation = StateEquationCole(; sound_speed=10.0f0,
                                       reference_density=1000.0f0, exponent=1)
    boundary = BoundaryModelDummyParticles(density32, mass32, SummationDensity(),
                                           WendlandC6Kernel{2}(), 0.2f0;
                                           state_equation,
                                           correction=KernelCorrection())
    @test eltype(boundary.cache.dw_gamma) == Float32

    for edac in (false, true),
        density_calculator in (SummationDensity(),
                               ContinuityDensity())
        result = correction_restart_result(KernelCorrection(); edac, density_calculator)
        @test result.state_equal
        @test result.rhs_equal
        @test result.cache_finite
    end
end
