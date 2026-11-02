@testset "Shepard summation with EOS pressure boundaries is history independent" begin
    function predictor_setup(kind; reverse_order=false,
                             calculator=AdamiPressureExtrapolation())
        spacing = 0.1
        kernel = WendlandC6Kernel{2}()
        equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                     exponent=1)
        initial = RectangularShape(spacing, (7, 7), (0.0, 0.0); density=1100.0,
                                   velocity=pos -> SVector(0.1+pos[1], -0.2-pos[2]))
        fluid = WeaklyCompressibleSPHSystem(initial; smoothing_kernel=kernel,
                                            smoothing_length=2spacing,
                                            density_calculator=SummationDensity(),
                                            state_equation=equation,
                                            acceleration=(0.0, -1.0),
                                            density_correction=ShepardKernelCorrection())
        geometry = RectangularShape(spacing, (7, 3), (0.0, -0.3); density=1200.0)
        model = BoundaryModelDummyParticles(fill(1000.0, nparticles(geometry)),
                                            fill(1000spacing^2, nparticles(geometry)),
                                            calculator, kernel, 2spacing;
                                            state_equation=equation)
        boundary = kind == :wall ? WallBoundarySystem(geometry, model) :
                   kind == :rigid ?
                   RigidBodySystem(geometry; boundary_model=model,
                                   particle_spacing=spacing) :
                   TotalLagrangianSPHSystem(geometry; smoothing_kernel=kernel,
                                            smoothing_length=2spacing, young_modulus=0.0,
                                            poisson_ratio=0.0, boundary_model=model)
        systems = reverse_order ? (boundary, fluid) : (fluid, boundary)
        semi = Semidiscretization(systems...; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        runtime = ode.p.semi
        fluid = only(filter(s->s isa WeaklyCompressibleSPHSystem, runtime.systems))
        boundary = only(filter(s->!(s isa WeaklyCompressibleSPHSystem), runtime.systems))
        return (; semi=runtime, fluid, boundary, v=Array(ode.u0.x[1]), u=Array(ode.u0.x[2]))
    end
    function predictor_rhs(setup)
        (; semi, fluid, boundary, v, u)=setup
        dv=zero(v)
        TrixiParticles.kick!(dv, v, u, (; semi, split_integration_data=nothing), 0.0)
        return (; density=copy(fluid.cache.density), pressure=copy(fluid.pressure),
                coefficient=copy(fluid.cache.kernel_correction_coefficient),
                boundary_density=copy(boundary.boundary_model.cache.density),
                boundary_pressure=copy(boundary.boundary_model.pressure), rhs=copy(dv))
    end
    for kind in (:wall, :rigid, :tlsph),
        calculator in (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation())
        setup = predictor_setup(kind; calculator)
        baseline = predictor_rhs(setup)
        @test any(!iszero, baseline.pressure)
        # Boundary density is an output, not hidden state of the next normalizer.
        setup.boundary.boundary_model.cache.density .= 700.0
        setup.boundary.boundary_model.pressure .= -1234.0
        repeated = predictor_rhs(setup)
        for name in propertynames(baseline)
            @test getproperty(repeated, name) == getproperty(baseline, name)
        end
        # Simulate a rejected trial evaluation, then restore the exact input state.
        coordinates = copy(setup.u)
        fluid_u = TrixiParticles.wrap_u(setup.u, setup.fluid, setup.semi)
        fluid_u[1, 1] += 0.01
        predictor_rhs(setup)
        copyto!(setup.u, coordinates)
        restored = predictor_rhs(setup)
        for name in propertynames(baseline)
            @test getproperty(restored, name) == getproperty(baseline, name)
        end
        reversed = predictor_rhs(predictor_setup(kind; reverse_order=true, calculator))
        for name in
            (:density, :pressure, :coefficient, :boundary_density, :boundary_pressure)
            @test getproperty(reversed, name)≈getproperty(baseline, name) rtol=5e-12 atol=1e-9
        end
        @test setup.fluid.pressure == setup.fluid.state_equation.(setup.fluid.cache.density)
    end
end
