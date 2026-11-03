@testset "WCSPH reconstructed-pressure boundary consistency" begin
    function closed_pressure_result(calculator, correction, boundary_correction;
                                    kind=:wall, extrapolation=AdamiPressureExtrapolation(),
                                    averaged=false)
        spacing = 0.1
        n = 5
        kernel = WendlandC2Kernel{2}()
        h = sqrt(2) * spacing
        equation = StateEquationCole(; sound_speed=50.0, reference_density=1000.0,
                                     exponent=7)
        density = TrixiParticles.inverse_state_equation(equation, 10000.0)
        initial = RectangularShape(spacing, (n, n), (0.0, 0.0); density)
        outer = RectangularShape(spacing, (n + 6, n + 6), (-3spacing, -3spacing);
                                 density)
        geometry = setdiff(outer, initial)
        fluid = WeaklyCompressibleSPHSystem(initial; smoothing_kernel=kernel,
                                            smoothing_length=h, state_equation=equation,
                                            density_calculator=calculator,
                                            pressure_acceleration=averaged ?
                                                                  TrixiParticles.inter_particle_averaged_pressure :
                                                                  nothing,
                                            initial_density_correction=calculator isa
                                                                       SummationDensity,
                                            gradient_correction=correction)
        model = BoundaryModelDummyParticles(geometry.density, geometry.mass,
                                            extrapolation, kernel, h;
                                            state_equation=equation,
                                            gradient_correction=kind == :rigid ? nothing :
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
        dv = zero(v)
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        acceleration = TrixiParticles.wrap_v(dv, fluid, semi)
        # All particles have the same physical pressure and complete kernel support.
        # The correct fluid acceleration is zero, independent of ghost-side moments.
        @test maximum(abs.(fluid.pressure .- 10000.0)) < 1e-6
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

    for calculator in (SummationDensity(), ContinuityDensity()), averaged in (false, true),
        extrapolation in (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation()),
        kind in (:wall, :rigid, :tlsph)
        closed_pressure_result(calculator, MixedKernelGradientCorrection(),
                               KernelCorrection(); kind, extrapolation, averaged)
    end
    for correction in (nothing, KernelCorrection(), GradientCorrection(),
         BlendedGradientCorrection(0.5), MixedKernelGradientCorrection())
        closed_pressure_result(SummationDensity(), correction,
                               MixedKernelGradientCorrection())
    end
    closed_pressure_result(SummationDensity(), KernelCorrection(), nothing)
end
