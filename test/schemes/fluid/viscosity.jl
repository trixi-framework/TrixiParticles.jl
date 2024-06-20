@testset verbose=true "Viscosity" begin
    particle_spacing = 0.2
    smoothing_length = 1.2 * particle_spacing
    smoothing_kernel = SchoenbergCubicSplineKernel{2}()
    sound_speed = 10 * sqrt(9.81 * 0.9)

    state_equation = StateEquationCole(; sound_speed, reference_density=1000.0,
                                       exponent=7, clip_negative_pressure=false)

    fluid =  rectangular_patch(particle_spacing, (3, 3), seed=1)

    system_wcsph = WeaklyCompressibleSPHSystem(fluid, ContinuityDensity(),
    state_equation, smoothing_kernel, smoothing_length)

    v_diff = [0.1, -0.75]
    pos_diff = [-0.5 * smoothing_length, 0.75 * smoothing_length]
    distance = norm(pos_diff)
    rho_a = rho_b = rho_mean = 1000.0

    grad_kernel = TrixiParticles.smoothing_kernel_grad(system_wcsph, pos_diff,
    distance)

    @testset verbose=true "`ArtificialViscosityMonaghan`" begin
        viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)


        vr = dot(v_diff, pos_diff)

        dv = viscosity(sound_speed, v_diff, pos_diff, distance,
            rho_mean, rho_a, rho_b, smoothing_length,
            grad_kernel)

        @test isapprox(dv[1], -0.05659079310795717, atol=6e-15)
        @test isapprox(dv[2], 0.08488618966193576, atol=6e-15)

    end
    @testset verbose=true "`ViscosityMorris`" begin
        viscosity = ViscosityMorris(nu=7e-3)

        dv = viscosity(sound_speed, v_diff, pos_diff, distance,
            rho_mean, rho_a, rho_b, smoothing_length,
            grad_kernel)

        @test isapprox(dv[1], -0.00294750186361511, atol=6e-15)
        @test isapprox(dv[2], 0.022106263977113322, atol=6e-15)
    end
    @testset verbose=true "`ViscosityAdami`" begin
    viscosity = ViscosityAdami(nu=7e-3)

    dv = viscosity(sound_speed, v_diff, pos_diff, distance,
        rho_mean, rho_a, rho_b, smoothing_length,
        grad_kernel)

    @test isapprox(dv[1], -0.00294750186361511, atol=6e-15)
    @test isapprox(dv[2], 0.022106263977113322, atol=6e-15)
end
end
