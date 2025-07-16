@testset verbose=true "Viscosity" begin
    particle_spacing = 0.2
    smoothing_length = 1.2 * particle_spacing
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    sound_speed = 10 * sqrt(9.81 * 0.9)

    state_equation = StateEquationCole(; sound_speed, reference_density=1000.0,
                                       exponent=7, clip_negative_pressure=false)

    fluid = rectangular_patch(particle_spacing, (3, 3), seed=1)

    v_diff = [0.3, -1.0]
    pos_diff = [-0.25 * smoothing_length, 0.375 * smoothing_length]
    distance = norm(pos_diff)
    rho_a = rho_b = rho_mean = 1000.0

    # We only test here that the values don't change
    @testset verbose=true "`ArtificialViscosityMonaghan`" begin
        viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

        system_wcsph = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity)

        grad_kernel = TrixiParticles.smoothing_kernel_grad(system_wcsph, pos_diff,
                                                           distance, 1)

        dv = viscosity(sound_speed, v_diff, pos_diff, distance,
                       rho_mean, rho_a, rho_b, smoothing_length,
                       grad_kernel, 0.0, 0.0)

        @test isapprox(dv[1], -0.02049217623299368, atol=6e-15)
        @test isapprox(dv[2], 0.03073826434949052, atol=6e-15)
    end
    @testset verbose=true "`ViscosityMorris`" begin
        nu = 7e-3
        viscosity = ViscosityMorris(nu=nu)
        system_wcsph = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity)

        grad_kernel = TrixiParticles.smoothing_kernel_grad(system_wcsph, pos_diff,
                                                           distance, 1)
        v = fluid.velocity

        m_a = 0.01
        m_b = 0.01

        v[1, 1] = v_diff[1]
        v[2, 1] = v_diff[2]
        v[1, 2] = 0.0
        v[2, 2] = 0.0

        dv = viscosity(system_wcsph, system_wcsph,
                       v, v, 1, 2, pos_diff, distance,
                       sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)

        @test isapprox(dv[1], -1.0895602048035404e-5, atol=6e-15)
        @test isapprox(dv[2], 3.631867349345135e-5, atol=6e-15)
    end
    @testset verbose=true "`ViscosityAdami`" begin
        viscosity = ViscosityAdami(nu=7e-3)
        system_wcsph = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity)

        grad_kernel = TrixiParticles.smoothing_kernel_grad(system_wcsph, pos_diff,
                                                           distance, 1)
        v = fluid.velocity

        m_a = 0.01
        m_b = 0.01

        v[1, 1] = v_diff[1]
        v[2, 1] = v_diff[2]
        v[1, 2] = 0.0
        v[2, 2] = 0.0

        dv = viscosity(system_wcsph, system_wcsph,
                       v, v, 1, 2, pos_diff, distance,
                       sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)

        @test isapprox(dv[1], -1.089560204803541e-5, atol=6e-15)
        @test isapprox(dv[2], 3.6318673493451364e-5, atol=6e-15)
    end
    @testset verbose=true "`ViscosityMorrisSGS`" begin
        nu = 7e-3
        viscosity = ViscosityMorrisSGS(nu=nu)
        system_wcsph = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity)

        grad_kernel = TrixiParticles.smoothing_kernel_grad(system_wcsph, pos_diff,
                                                           distance, 1)

        v = fluid.velocity

        m_a = 0.01
        m_b = 0.01

        v[1, 1] = v_diff[1]
        v[2, 1] = v_diff[2]
        v[1, 2] = 0.0
        v[2, 2] = 0.0

        dv = viscosity(system_wcsph, system_wcsph,
                       v, v, 1, 2, pos_diff, distance,
                       sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)

        @test isapprox(dv[1], -2.032835697804103e-5, atol=6e-15)
        @test isapprox(dv[2], 6.776118992680343e-5, atol=6e-15)
    end
    @testset verbose=true "`ViscosityAdamiSGS`" begin
        viscosity = ViscosityAdamiSGS(nu=7e-3)
        system_wcsph = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length, viscosity=viscosity)

        grad_kernel = TrixiParticles.smoothing_kernel_grad(system_wcsph, pos_diff,
                                                           distance, 1)
        v = fluid.velocity

        m_a = 0.01
        m_b = 0.01

        v[1, 1] = v_diff[1]
        v[2, 1] = v_diff[2]
        v[1, 2] = 0.0
        v[2, 2] = 0.0

        dv = viscosity(system_wcsph, system_wcsph,
                       v, v, 1, 2, pos_diff, distance,
                       sound_speed, m_a, m_b, rho_a, rho_b, grad_kernel)

        @test isapprox(dv[1], -2.0328356978041036e-5, atol=6e-15)
        @test isapprox(dv[2], 6.776118992680346e-5, atol=6e-15)
    end
end
