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

@testset "Zhang inlet (wall first: $wall_first)" for wall_first in (false, true)
    particle_spacing = 0.1
    smoothing_length = 2particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    fluid_particles = RectangularShape(particle_spacing, (7, 7), (0.0, 0.0);
                                       density=1100.0)
    fluid = WeaklyCompressibleSPHSystem(fluid_particles; smoothing_kernel, smoothing_length,
                                        state_equation,
                                        density_calculator=SummationDensity(),
                                        density_correction=ShepardKernelCorrection(),
                                        buffer_size=10)
    zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 0.7]), face_normal=[1.0, 0.0],
                        boundary_type=InFlow(), open_boundary_layers=10, density=1000.0,
                        particle_spacing)
    inlet = OpenBoundarySystem(zone; fluid_system=fluid, buffer_size=10,
                               boundary_model=BoundaryModelDynamicalPressureZhang())

    # The wall shares its pressure extrapolation neighborhood with the fluid and inlet.
    wall_particles = RectangularShape(particle_spacing, (17, 3), (-1.0, -0.3);
                                      density=1000.0)
    boundary_model = BoundaryModelDummyParticles(wall_particles.density,
                                                 wall_particles.mass,
                                                 AdamiPressureExtrapolation(),
                                                 smoothing_kernel,
                                                 smoothing_length; state_equation)
    wall = WallBoundarySystem(wall_particles, boundary_model)
    systems = wall_first ? (fluid, wall, inlet) : (fluid, inlet, wall)
    cell_list = FullGridCellList(; min_corner=[-2.0, -1.0], max_corner=[2.0, 2.0])
    neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                    update_strategy=ParallelUpdate())
    semi = Semidiscretization(systems...; neighborhood_search,
                              parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    (; semi) = ode.p
    fluid, system_a, system_b = semi.systems
    wall, inlet = wall_first ? (system_a, system_b) : (system_b, system_a)
    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    # This test calls the RHS directly, without taking an integrator step.
    semi.update_callback_used[] = true
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    density = copy(fluid.cache.density)
    wall_pressure = copy(wall.boundary_model.pressure)
    inlet_pressure = copy(inlet.cache.pressure)
    rhs = copy(dv_ode)

    # A rejected trial with a different inlet density must not leave stale inlet pressure.
    initial_v = copy(v_ode)
    GC.@preserve v_ode begin
        v_inlet = TrixiParticles.wrap_v(v_ode, inlet, semi)
        inlet_density = TrixiParticles.current_density(v_inlet, inlet)
        inlet_density[TrixiParticles.each_integrated_particle(inlet)] .+= 100.0
    end
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    @test inlet.cache.pressure != inlet_pressure
    copyto!(v_ode, initial_v)
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    @test fluid.cache.density == density
    @test wall.boundary_model.pressure == wall_pressure
    @test dv_ode == rhs
end
