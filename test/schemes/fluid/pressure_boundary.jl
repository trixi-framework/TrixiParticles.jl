@testset "WCSPH reconstructed-pressure boundary consistency" begin
    function closed_pressure_result(calculator, correction, boundary_correction;
                                    kind=:wall, extrapolation=AdamiPressureExtrapolation())
        spacing = 0.1
        n = 5
        kernel = WendlandC2Kernel{2}()
        h = sqrt(2) * spacing
        density = 1000.0
        equation = StateEquationCole(; sound_speed=50.0, reference_density=density,
                                     exponent=1, background_pressure=10000.0)
        initial = RectangularShape(spacing, (n, n), (0.0, 0.0); density)
        outer = RectangularShape(spacing, (n + 6, n + 6), (-3spacing, -3spacing); density)
        geometry = setdiff(outer, initial)
        fluid = WeaklyCompressibleSPHSystem(initial; smoothing_kernel=kernel,
                                            smoothing_length=h, state_equation=equation,
                                            density_calculator=calculator, correction)
        model = BoundaryModelDummyParticles(geometry.density, geometry.mass,
                                            extrapolation, kernel, h;
                                            state_equation=equation,
                                            correction=kind == :rigid ? nothing :
                                                       boundary_correction)
        boundary = kind == :wall ? WallBoundarySystem(geometry, model) :
                   kind == :rigid ?
                   RigidBodySystem(geometry; boundary_model=model,
                                   particle_spacing=spacing) :
                   TotalLagrangianSPHSystem(geometry; smoothing_kernel=kernel,
                                            smoothing_length=h, young_modulus=0.0,
                                            poisson_ratio=0.0, boundary_model=model)
        ode = semidiscretize(Semidiscretization(fluid, boundary;
                                                neighborhood_search=nothing,
                                                parallelization_backend=SerialBackend()),
                             (0.0, 1.0); reset_threads=false)
        semi = ode.p.semi
        fluid, boundary = semi.systems
        v, u = ode.u0.x
        # Bring the algebraic fluid density, reconstructed boundary density and
        # correction moments to the same frozen state before measuring its force.
        for _ in 1:3
            TrixiParticles.update_systems_and_nhs(v, u, semi, 0.0)
        end
        dv = zero(v)
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        acceleration = TrixiParticles.wrap_v(dv, fluid, semi)
        # A closed constant-pressure vessel must have zero fluid acceleration,
        # independently of correction moments on the ghost support.
        @test minimum(fluid.pressure) >= 9999.0
        @test maximum(fluid.pressure) - minimum(fluid.pressure) < 1e-6
        @test maximum(abs, view(acceleration, 1:2, :)) < 1e-8
        @test all(isfinite, dv)
        if kind != :wall
            fluid_force = vec(sum(fluid.mass' .* view(acceleration, 1:2, :); dims=2))
            forces = kind == :rigid ? boundary.force_per_particle :
                     boundary.mass' .* TrixiParticles.wrap_v(dv, boundary, semi)
            boundary_force = vec(sum(forces; dims=2))
            force_scale = max(sum(norm, eachcol(forces)), 1.0)
            @test norm(fluid_force + boundary_force) / force_scale < 1e-12
        end
    end

    for calculator in (SummationDensity(), ContinuityDensity()),
        extrapolation in (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation()),
        kind in (:wall, :rigid, :tlsph)
        closed_pressure_result(calculator, MixedKernelGradientCorrection(),
                               KernelCorrection(); kind, extrapolation)
    end
    # Zero blending must retain rest even when the ghost-side correction differs.
    for correction in (nothing, KernelCorrection(), GradientCorrection(),
         BlendedGradientCorrection(0.0), MixedKernelGradientCorrection())
        closed_pressure_result(SummationDensity(), correction,
                               MixedKernelGradientCorrection())
    end
    closed_pressure_result(SummationDensity(), KernelCorrection(), nothing)
end
