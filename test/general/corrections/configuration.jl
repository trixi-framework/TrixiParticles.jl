@testset "Independent correction configuration" begin
    for edac in (false, true)
        setup = correction_setup(; density_calculator=SummationDensity(), edac,
                                 density_correction=ShepardKernelCorrection(),
                                 gradient_correction=MixedKernelGradientCorrection())
        update_correction!(setup)
        @test setup.system.correction isa CorrectionConfiguration
        @test all(isfinite, setup.system.cache.kernel_correction_coefficient)
        @test all(isfinite, setup.system.cache.dw_gamma)
        @test all(isfinite, setup.system.cache.correction_matrix)
    end

    @test_throws ArgumentError correction_setup(GradientCorrection();
                                                gradient_correction=GradientCorrection())
    @test_throws ArgumentError correction_setup(;
                                                density_calculator=ContinuityDensity(),
                                                density_correction=ShepardKernelCorrection())
    @test_throws ArgumentError CorrectionConfiguration(; density=GradientCorrection())
    @test_throws ArgumentError CorrectionConfiguration(; gradient=ShepardKernelCorrection())

    particles = RectangularShape(0.1, (2, 2), (0.0, 0.0); density=1000.0)
    @test_throws ArgumentError ImplicitIncompressibleSPHSystem(particles;
                                                               smoothing_kernel=WendlandC6Kernel{2}(),
                                                               smoothing_length=0.2,
                                                               reference_density=1000.0,
                                                               time_step=0.01,
                                                               correction=GradientCorrection())

    n = 5
    spacing = 1.0 / n
    kernel = WendlandC6Kernel{2}()
    boundary_particles = RectangularShape(spacing, (n, n), (0.0, 0.0); density=1000.0)
    state_equation = StateEquationCole(; sound_speed=10.0,
                                       reference_density=1000.0, exponent=1)
    boundary_model = BoundaryModelDummyParticles(boundary_particles.density,
                                                 boundary_particles.mass,
                                                 SummationDensity(), kernel, 2spacing;
                                                 state_equation,
                                                 density_correction=ShepardKernelCorrection(),
                                                 gradient_correction=MixedKernelGradientCorrection())
    boundary = WallBoundarySystem(boundary_particles, boundary_model)
    semi = Semidiscretization(boundary; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v_ode = Array(ode.u0.x[1])
    u_ode = Array(ode.u0.x[2])
    boundary = first(ode.p.semi.systems)
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)
    @test boundary.boundary_model.correction isa CorrectionConfiguration
    @test all(isfinite, boundary.boundary_model.cache.kernel_correction_coefficient)
    @test all(isfinite, boundary.boundary_model.cache.dw_gamma)
    @test all(isfinite, boundary.boundary_model.cache.correction_matrix)

    combined = update_correction!(correction_setup(;
                                                   density_calculator=SummationDensity(),
                                                   density_correction=ShepardKernelCorrection(),
                                                   gradient_correction=MixedKernelGradientCorrection()))
    moments = correction_moments(combined; field=pos -> 2.0 + 3.0pos[1] - 2.0pos[2])
    identity_matrix = Matrix{Float64}(I, 2, 2)
    @test maximum(abs, moments.zeroth_gradient_moment) < 3e-12
    @test maximum(particle -> norm(moments.first_gradient_moment[:, :, particle] -
                                   identity_matrix),
                  TrixiParticles.eachparticle(combined.system)) < 3e-12
end
