@testset "Mixed kernel-gradient correction" begin
    identity_matrix = Matrix{Float64}(I, 2, 2)
    linear_field(pos) = 2.0 + 3.0 * pos[1] - 2.0 * pos[2]
    exact_gradient = [3.0, -2.0]

    # Both fluid formulations allocate finite caches for all mixed correction components.
    for edac in (false, true)
        setup = correction_setup(MixedKernelGradientCorrection(); edac,
                                 pressure_acceleration=nothing)
        update_correction!(setup)
        @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
        @test all(isfinite, setup.system.cache.dw_gamma)
        @test all(isfinite, setup.system.cache.correction_matrix)
    end

    # The composed correction reproduces constants and affine fields on regular and perturbed grids.
    for perturbation in (false, true)
        setup = update_correction!(correction_setup(MixedKernelGradientCorrection();
                                                    perturbation))
        moments = correction_moments(setup; field=linear_field)
        @test maximum(abs, moments.zeroth_gradient_moment) < 3e-12
        @test maximum(particle -> norm(moments.first_gradient_moment[:, :, particle] -
                                       identity_matrix),
                      TrixiParticles.eachparticle(setup.system)) < 3e-12
        @test maximum(particle -> norm(moments.direct_gradient[:, particle] -
                                       exact_gradient),
                      TrixiParticles.eachparticle(setup.system)) < 1e-11
    end

    # A uniform fluid state has the analytic continuity-density rate of -2000.
    setup = correction_setup(MixedKernelGradientCorrection())
    dv_ode = zero(setup.v_ode)
    TrixiParticles.kick!(dv_ode, setup.v_ode, setup.u_ode,
                         (; semi=setup.semi, split_integration_data=nothing), 0.0)
    dv = TrixiParticles.wrap_v(dv_ode, setup.system, setup.semi)
    density_error = dv[end, :] .+ 2000.0
    @test sqrt(sum(abs2, density_error) / length(density_error)) < 2e-10

    # Restarting preserves state, RHS, and correction caches for every density formulation.
    for edac in (false, true),
        density_calculator in (SummationDensity(),
                               ContinuityDensity())
        result = correction_restart_result(MixedKernelGradientCorrection();
                                           edac, density_calculator)
        @test result.state_equal
        @test result.rhs_equal
        @test result.cache_finite
    end

    # Boundary cache arrays retain the system scalar type.
    density32 = fill(1000.0f0, 4)
    mass32 = fill(10.0f0, 4)
    state_equation = StateEquationCole(; sound_speed=10.0f0,
                                       reference_density=1000.0f0, exponent=1)
    boundary = BoundaryModelDummyParticles(density32, mass32, SummationDensity(),
                                           WendlandC6Kernel{2}(), 0.2f0;
                                           state_equation,
                                           correction=MixedKernelGradientCorrection())
    @test eltype(boundary.cache.dw_gamma) == Float32
    @test eltype(boundary.cache.correction_matrix) == Float32
end
