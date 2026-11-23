@testset "Two-stage Adami boundary pressure with fallback" begin
    spacing = 0.1
    h = 2spacing
    kernel = WendlandC6Kernel{2}()
    rho0 = 1000.0
    c = 10.0
    exponent = 7
    gravity = 2.0
    equation = StateEquationCole(; sound_speed=c, reference_density=rho0, exponent)
    viscosity = ViscosityAdami(; nu=1e-6)
    velocity = SVector(0.5, -0.2)
    fluid_particles = RectangularShape(spacing, (1, 1), (0.0, 0.0); density=rho0,
                                       velocity)
    fluid = WeaklyCompressibleSPHSystem(fluid_particles; smoothing_kernel=kernel,
                                        smoothing_length=h, state_equation=equation,
                                        viscosity, acceleration=(0.0, -gravity),
                                        density_calculator=SummationDensity(),
                                        density_correction=ShepardKernelCorrection())
    wall_heights = (-spacing, spacing)
    initial_wall_density = (rho0, 980.0)
    walls = map(wall_heights, initial_wall_density) do y, density
        particles = RectangularShape(spacing, (1, 1), (0.0, y); density=rho0)
        model = BoundaryModelDummyParticles([density], particles.mass,
                                            AdamiPressureExtrapolation(),
                                            kernel, h; state_equation=equation, viscosity)
        return WallBoundarySystem(particles, model)
    end
    semi = Semidiscretization(fluid, walls...; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    v, u = ode.u0.x
    dv = zero(v)
    TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
    fluid, walls... = ode.p.semi.systems

    # Each wall sees exactly one fluid neighbor, so the Adami kernel weight cancels.
    # The hydrostatic term puts the upper wall below the Cole limit in the provisional
    # pass only. The lower wall has a valid inverse in both passes.
    B = rho0 * c^2 / exponent
    fluid_pressure(rho) = B * ((rho / rho0)^exponent - 1)
    wall_pressure(rho, y) = fluid_pressure(rho) - gravity * rho * y
    density_from_pressure(p) = rho0 * (1 + p / B)^(1 / exponent)
    self_term = only(fluid_particles.mass) * TrixiParticles.kernel(kernel, 0.0, h)
    wall_term = only(first(walls).boundary_model.hydrodynamic_mass) *
                TrixiParticles.kernel(kernel, spacing, h)
    raw_density = self_term + 2wall_term
    p_lower, p_upper = map(y -> wall_pressure(raw_density, y), wall_heights)
    @test p_upper < -B < p_lower
    coefficient = self_term / raw_density + wall_term / density_from_pressure(p_lower) +
                  wall_term / last(initial_wall_density)
    corrected_density = raw_density / coefficient

    # Re-run with poisoned wall caches to verify that both the inverse and the fallback
    # are independent of the previous RHS evaluation.
    rhs = copy(dv)
    for wall in walls
        wall.boundary_model.cache.density .= 700.0
        wall.boundary_model.pressure .= -1234.0
    end
    TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
    @test all(isfinite, dv)
    @test dv == rhs
    @test only(fluid.cache.kernel_correction_coefficient) ≈ coefficient
    @test only(fluid.cache.density) ≈ corrected_density
    @test only(fluid.pressure) ≈ fluid_pressure(corrected_density)
    for (wall, y) in zip(walls, wall_heights)
        final_wall_pressure = wall_pressure(corrected_density, y)
        @test -B < final_wall_pressure < 0
        @test only(wall.boundary_model.pressure) ≈ final_wall_pressure
        @test only(wall.boundary_model.cache.density) ≈
              density_from_pressure(final_wall_pressure)
        @test vec(wall.boundary_model.cache.wall_velocity) ≈ -velocity
    end

    # The helper warms up both RHS functions and excludes neighborhood search allocations.
    saved_state = (; t=(0.0,), u=(ode.u0,), prob=ode)
    @test count_rhs_allocations(saved_state) == 0
end

@testset "Shepard summation with pressure-extrapolated boundaries is history independent" begin
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
        # Report derivatives in physical system order, independent of storage order.
        fluid_rhs = dv[semi.ranges_v[TrixiParticles.system_indices(fluid, semi)]]
        boundary_rhs = dv[semi.ranges_v[TrixiParticles.system_indices(boundary, semi)]]
        return (; density=copy(fluid.cache.density), pressure=copy(fluid.pressure),
                boundary_density=copy(boundary.boundary_model.cache.density),
                boundary_pressure=copy(boundary.boundary_model.pressure), fluid_rhs,
                boundary_rhs)
    end
    calculators = (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation())
    cases = Iterators.product((:wall, :rigid, :tlsph), calculators)
    @testset "$kind / $(nameof(typeof(calculator)))" for (kind, calculator) in cases
        setup = predictor_setup(kind; calculator)
        baseline = predictor_rhs(setup)
        # Boundary density is an output, not hidden state of the next normalizer.
        setup.boundary.boundary_model.cache.density .= 700.0
        setup.boundary.boundary_model.pressure .= -1234.0
        @test predictor_rhs(setup) == baseline
        # Simulate a rejected trial evaluation, then restore the exact input state.
        coordinates = copy(setup.u)
        fluid_index = TrixiParticles.system_indices(setup.fluid, setup.semi)
        setup.u[first(setup.semi.ranges_u[fluid_index])] += 0.01
        @test predictor_rhs(setup).boundary_density != baseline.boundary_density
        copyto!(setup.u, coordinates)
        @test predictor_rhs(setup) == baseline
        reversed = predictor_rhs(predictor_setup(kind; reverse_order=true, calculator))
        @test all(isapprox(a, b; rtol=5e-12, atol=1e-9)
                  for (a, b) in zip(reversed, baseline))
    end
end

function test_zhang_inlet_pressure_ordering(; shepard, wall_before_inlet)
    particle_spacing = 0.1
    smoothing_length = 2particle_spacing
    smoothing_kernel = WendlandC6Kernel{2}()
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=1)
    fluid_particles = RectangularShape(particle_spacing, (7, 7), (0.0, 0.0);
                                       density=1100.0)
    density_correction = shepard ? ShepardKernelCorrection() : nothing
    fluid = WeaklyCompressibleSPHSystem(fluid_particles; smoothing_kernel, smoothing_length,
                                        state_equation,
                                        density_calculator=SummationDensity(),
                                        density_correction, buffer_size=10)
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
    systems = wall_before_inlet ? (fluid, wall, inlet) : (fluid, inlet, wall)
    cell_list = FullGridCellList(; min_corner=[-2.0, -1.0], max_corner=[2.0, 2.0])
    neighborhood_search = GridNeighborhoodSearch{2}(; cell_list,
                                                    update_strategy=ParallelUpdate())
    semi = Semidiscretization(systems...; neighborhood_search,
                              parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    (; semi) = ode.p
    fluid, system_a, system_b = semi.systems
    wall, inlet = wall_before_inlet ? (system_a, system_b) : (system_b, system_a)
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
    @test wall.boundary_model.pressure != wall_pressure
    copyto!(v_ode, initial_v)
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    @test fluid.cache.density == density
    @test wall.boundary_model.pressure == wall_pressure
    @test dv_ode == rhs
end

@testset "Zhang inlet pressure ordering" begin
    for shepard in (false, true), wall_before_inlet in (false, true)
        @testset "Shepard: $shepard, wall before inlet: $wall_before_inlet" begin
            test_zhang_inlet_pressure_ordering(; shepard, wall_before_inlet)
        end
    end
end
