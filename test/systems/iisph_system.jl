@testset verbose=true "ImplicitIncompressibleSPHSystem" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors" begin
        # Verify constructor wiring, default values, and argument validation across 2D/3D
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0]
        ]
        omegas = [
            0.5,
            0.6
        ]
        max_errors = [
            0.1,
            0.05
        ]
        min_iterations_ = [
            2,
            10
        ]
        max_iterations_ = [
            20,
            30
        ]
        time_steps_ = [
            0.001,
            0.0001
        ]
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            mass = [1.25, 1.5]
            density = [990.0, 1000.0]
            reference_density = 1000.0
            smoothing_kernel = Val(:smoothing_kernel)
            omega = omegas[i]
            max_error = max_errors[i]
            min_iterations = min_iterations_[i]
            max_iterations = max_iterations_[i]
            time_step = time_steps_[i]
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = i + 1
            smoothing_kernel2 = Val(:smoothing_kernel2)
            # The wrong dimension. 2 -> 3, 3 -> 2.
            TrixiParticles.ndims(::Val{:smoothing_kernel2}) = i % 2 + 2
            smoothing_length = 0.362

            initial_condition = InitialCondition(; coordinates, mass, density)
            system = ImplicitIncompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                     smoothing_length, reference_density,
                                                     omega, max_error, min_iterations,
                                                     max_iterations, time_step)

            # Constructor copies input fields, applies defaults, and respects the requested dimensionality
            @test system isa ImplicitIncompressibleSPHSystem{NDIMS}
            @test system.initial_condition == initial_condition
            @test system.mass == mass
            @test system.reference_density == reference_density
            @test system.smoothing_kernel == smoothing_kernel
            @test system.omega == omega
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.viscosity === nothing
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test system.max_error == max_error
            @test system.min_iterations == min_iterations
            @test system.max_iterations == max_iterations
            @test system.time_step == time_step
            @test length(system.density) == size(coordinates, 2)

            # A too-short acceleration vector triggers dimension validation
            error_str1 = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str1) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density=density,
                                                                                   acceleration=(0.0),
                                                                                   time_step=0.001)

            # Smoothing kernel dimensionality must match the problem dimension
            error_str2 = "smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str2) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel=smoothing_kernel2,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   time_step=0.001)

            # Reference density must be positive
            error_str3 = "`reference_density` must be a positive number"
            @test_throws ArgumentError(error_str3) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density=0.0,
                                                                                   time_step=0.001)

            # max_error is a percentage and must be in (0, 100]
            error_str4 = "`max_error` is given in percentage, so it must be a number between 0 and 100"
            @test_throws ArgumentError(error_str4) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   max_error=0.0,
                                                                                   time_step=0.001)

            # min_iterations must be strictly positive
            error_str5 = "`min_iterations` must be a positive number"
            @test_throws ArgumentError(error_str5) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=0,
                                                                                   time_step=0.001)

            # min_iterations must not exceed max_iterations
            error_str6 = "`min_iterations` must be smaller or equal to `max_iterations`"
            @test_throws ArgumentError(error_str6) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=10,
                                                                                   max_iterations=5,
                                                                                   time_step=0.001)

            # time_step must be strictly positive
            error_str7 = "`time_step` must be a positive number"
            @test_throws ArgumentError(error_str7) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                   smoothing_kernel,
                                                                                   smoothing_length,
                                                                                   reference_density,
                                                                                   min_iterations=1,
                                                                                   max_iterations=5,
                                                                                   time_step=0)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "Constructors with Setups" begin
        # Validate setup-based construction for several geometries and ensure validation hooks persist
        setups = [
            RectangularShape(0.123, (2, 3), (-1.0, 0.1), density=1.0),
            RectangularShape(0.123, (2, 3, 2), (-1.0, 0.1, 2.1), density=1.0),
            RectangularTank(0.123, (0.369, 0.246), (0.369, 0.369), 1020.0).fluid,
            RectangularTank(0.123, (0.369, 0.246, 0.246), (0.369, 0.492, 0.492),
                            1020.0).fluid,
            SphereShape(0.52, 0.1, (-0.2, 0.123), 1.0)
        ]
        setup_names = [
            "RectangularShape 2D",
            "RectangularShape 3D",
            "RectangularTank 2D",
            "RectangularTank 3D",
            "SphereShape 2D"
        ]
        NDIMS_ = [2, 3, 2, 3, 2]
        reference_densities = [1.0, 1.0, 1020.0, 1020.0, 1.0]
        @testset "$(setup_names[i])" for i in eachindex(setups)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            smoothing_kernel = Val(:smoothing_kernel)
            reference_density = reference_densities[i]
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362

            density_calculator = SummationDensity()

            system = ImplicitIncompressibleSPHSystem(setup;
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density,
                                                     time_step=0.001)

            # Constructor from setup copies geometry properties, sets defaults, and preserves dimension
            @test system isa ImplicitIncompressibleSPHSystem{NDIMS}
            @test system.initial_condition == setup
            @test system.mass == setup.mass
            @test system.smoothing_kernel == smoothing_kernel
            @test system.reference_density == reference_density
            @test TrixiParticles.initial_smoothing_length(system) == smoothing_length
            @test system.viscosity === nothing
            @test system.acceleration == [0.0 for _ in 1:NDIMS]
            @test length(system.mass) == size(setup.coordinates, 2)
            @test length(system.density) == size(setup.coordinates, 2)
        end

        # wrong dimension of acceleration
        NDIMS_ = [2, 3]
        @testset "Wrong acceleration dimension" for i in eachindex(NDIMS_)
            setup = setups[i]
            NDIMS = NDIMS_[i]
            reference_density = 1.0
            smoothing_kernel = Val(:smoothing_kernel)
            TrixiParticles.ndims(::Val{:smoothing_kernel}) = NDIMS
            smoothing_length = 0.362

            density_calculator = SummationDensity()

            # Acceleration vector length validation also applies to setup-based constructors
            error_str = "`acceleration` must be of length $NDIMS for a $(NDIMS)D problem"
            @test_throws ArgumentError(error_str) ImplicitIncompressibleSPHSystem(setup;
                                                                                  smoothing_kernel,
                                                                                  smoothing_length,
                                                                                  reference_density,
                                                                                  acceleration=(0.0),
                                                                                  time_step=0.001)
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.001)

        # repr without MIME should show compact one-line summary.
        show_compact = "ImplicitIncompressibleSPHSystem{2}(1000.0, Val{:smoothing_kernel}(), nothing, [0.0, 0.0], 0.5, 0.1, 2, 20) with 2 particles"
        @test repr(system) == show_compact
        # repr("text/plain", ...) should emit the detailed boxed summary.
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ImplicitIncompressibleSPHSystem{2}                                                               │
        │ ══════════════════════════════════                                                               │
        │ #particles: ………………………………………………… 2                                                                │
        │ reference density: ……………………………… 1000.0                                                           │
        │ density calculator: …………………………… SummationDensity                                                 │
        │ smoothing kernel: ………………………………… Val                                                              │
        │ viscosity: …………………………………………………… nothing                                                          │
        │ acceleration: …………………………………………… [0.0, 0.0]                                                       │
        │ omega: ……………………………………………………………… 0.5                                                              │
        │ max_error: …………………………………………………… 0.1                                                              │
        │ min_iterations: ……………………………………… 2                                                                │
        │ max_iterations: ……………………………………… 20                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    @testset verbose=true "write_u0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel,
                                                 smoothing_length, reference_density,
                                                 time_step=0.001)

        # `write_u0!` must copy initial coordinates into the solver state array
        u0 = zeros(TrixiParticles.u_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_u0!(u0, system)

        @test u0 == coordinates
    end

    @testset verbose=true "write_v0!" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        velocity = 2 * coordinates
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]
        reference_density = 1000.0
        smoothing_kernel = Val(:smoothing_kernel)
        smoothing_length = 0.362

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)

        # SummationDensity (is always in use)
        system = ImplicitIncompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.001)

        # write_v0! must copy initial velocities into the solver state array
        v0 = zeros(TrixiParticles.v_nvariables(system),
                   TrixiParticles.n_integrated_particles(system))
        TrixiParticles.write_v0!(v0, system)

        @test v0 == velocity
    end

    @testset "Extrapolated wall density is independent of RHS history" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        h = 1.0
        dt = 1e-3
        rho = 1000.0
        viscosity = ViscosityAdami(nu=0.1)
        x = (0.0, 1.5, 1.75)
        weights = [TrixiParticles.kernel(kernel, abs(a - b), h) for a in x, b in x]
        masses = rho / (weights[1, 1] + 2weights[1, 2] + 3weights[1, 3]) .* [1, 2, 3]

        function make_wall(i, calculator; state_equation=nothing, wall_viscosity=nothing)
            ic = InitialCondition(; coordinates=reshape([x[i], 0.0], 2, 1), density=rho,
                                  mass=masses[i], particle_spacing=h)
            model = BoundaryModelDummyParticles(ic.density, ic.mass, calculator, kernel, h;
                                                viscosity=wall_viscosity, state_equation)
            motion = PrescribedMotion((coords, t) -> coords + SVector(0.0, t), t -> true)
            return WallBoundarySystem(ic, model; prescribed_motion=motion)
        end

        calculators = (AdamiPressureExtrapolation(pressure_offset=50.0),
                       BernoulliPressureExtrapolation(pressure_offset=50.0))
        configurations = Iterators.product(calculators, (false, true))
        @testset "$calculator, wall_first=$wall_first" for (calculator, wall_first) in
                                                           configurations

            ic = InitialCondition(; coordinates=zeros(2, 1), density=rho, mass=masses[1],
                                  particle_spacing=h, velocity=reshape([0.005, 0.0], 2, 1))
            fluid = ImplicitIncompressibleSPHSystem(ic; smoothing_kernel=kernel,
                                                    smoothing_length=h,
                                                    reference_density=rho,
                                                    time_step=dt, viscosity)
            eos = StateEquationCole(sound_speed=5.0, reference_density=rho, exponent=1)
            wall = make_wall(2, calculator; state_equation=eos, wall_viscosity=viscosity)
            other_wall = make_wall(3, PressureZeroing())
            systems = wall_first ? (wall, other_wall, fluid) : (fluid, wall, other_wall)
            semi = Semidiscretization(systems...; parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01))
            v, u = ode.u0.x
            dv = zero(v)
            model = wall.boundary_model

            function wall_rhs()
                TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
                return (; fluid_pressure=fluid.pressure[1], wall_pressure=model.pressure[1],
                        wall_density=model.cache.density[1],
                        predicted_y=fluid.advection_velocity[2, 1],
                        dv=copy(TrixiParticles.wrap_v(dv, fluid, ode.p.semi)))
            end

            baseline = wall_rhs()
            @test baseline.wall_pressure > calculator.pressure_offset
            @test baseline.wall_density > rho
            @test eos(baseline.wall_density) ≈ baseline.wall_pressure
            @test baseline.dv[2, 1] > 0
            @test baseline.predicted_y ≈ dt * baseline.dv[2, 1]
            @test wall_rhs() == baseline

            fluid.pressure[1] = 1234.0
            model.cache.density[1] = 2rho
            model.pressure[1] = -1234.0
            fill!(model.cache.wall_velocity, 1234.0)
            @test wall_rhs() == baseline

            # Simulate a rejected trial evaluation and restore the exact ODE state.
            coords = TrixiParticles.wrap_u(u, fluid, ode.p.semi)
            coords[1, 1] = 0.025
            @test wall_rhs() != baseline
            coords[1, 1] = 0.0
            @test wall_rhs() == baseline
        end
    end

    @testset "IISPH wall pressure and normal motion in the density balance" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        rho0 = 1000.0
        h = 1.0
        weight = TrixiParticles.kernel(kernel, 0.0, h) +
                 TrixiParticles.kernel(kernel, h, h)

        function evaluate_pair(calculator; dt=0.005, rho=rho0, wall_speed=0.0,
                               gravity=0.0, eos=nothing, reverse_order=false,
                               min_iterations=2)
            # Equal masses give both particles the specified kernel-summation density.
            mass = rho / weight
            fluid_ic = InitialCondition(; coordinates=zeros(2, 1), density=rho0, mass,
                                        particle_spacing=h)
            wall_ic = InitialCondition(; coordinates=reshape([h, 0.0], 2, 1),
                                       density=rho0, mass, particle_spacing=h)
            # Configure a distinct ghost velocity without adding a viscous force: zero
            # wall viscosity makes the harmonic viscosity of the wall pair zero.
            fluid_viscosity = eos === nothing ? nothing : ViscosityAdami(nu=0.1)
            wall_viscosity = eos === nothing ? nothing : ViscosityAdami(nu=0.0)
            fluid = ImplicitIncompressibleSPHSystem(fluid_ic; smoothing_kernel=kernel,
                                                    smoothing_length=h, time_step=dt,
                                                    reference_density=rho0, omega=0.4,
                                                    min_iterations, max_iterations=100,
                                                    max_error=1e-9,
                                                    viscosity=fluid_viscosity,
                                                    acceleration=(gravity, 0.0))
            model = BoundaryModelDummyParticles(wall_ic.density, wall_ic.mass, calculator,
                                                kernel, h; state_equation=eos,
                                                viscosity=wall_viscosity)
            motion = PrescribedMotion((coords, t) -> coords + SVector(wall_speed * t, 0.0),
                                      t -> true)
            wall = WallBoundarySystem(wall_ic, model; prescribed_motion=motion)
            systems = reverse_order ? (wall, fluid) : (fluid, wall)
            semi = Semidiscretization(systems...; parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, dt))
            v, u = ode.u0.x
            dv = zero(v)
            TrixiParticles.kick!(dv, v, u, ode.p, 0.0)

            grad_x = TrixiParticles.smoothing_kernel_grad(fluid, SVector(-h, 0.0), h, 1)[1]
            acceleration = TrixiParticles.wrap_v(dv, fluid, ode.p.semi)[1, 1]
            # Apply the continuity linearization to the actual fluid RHS and wall motion.
            updated_density = rho + dt * mass * (dt * acceleration - wall_speed) * grad_x
            return (; predicted_density=only(fluid.predicted_density),
                    fluid_pressure=only(fluid.pressure), wall_pressure=only(model.pressure),
                    wall_density=only(model.cache.density), acceleration, updated_density,
                    density_weight=mass * grad_x,
                    ghost_speed=wall_viscosity === nothing ? 0.0 :
                                model.cache.wall_velocity[1, 1])
        end

        @testset "Invalid seed, admissible solved pressure" begin
            # With zero fluid pressure, the hydrostatic term alone is below the Cole
            # inverse's lower limit. The provisional fluid pressure makes it admissible.
            rho = 1005.0
            gravity = -10.0
            c = 5.0
            exponent = 7
            eos = StateEquationCole(; sound_speed=c, reference_density=rho0, exponent)
            @test gravity * rho * h < -rho0 * c^2 / exponent
            for calculator in
                (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation())
                result = evaluate_pair(calculator; dt=0.023, rho, gravity, eos)
                @test -rho0 * c^2 / exponent < result.wall_pressure < 0.0
                @test eos(result.wall_density) ≈ result.wall_pressure
                @test result.updated_density≈rho0 atol=2e-8 rtol=0
            end
        end

        @testset "Normal wall motion and frozen pressure" begin
            wall_speed = -1.0
            dt = 0.005
            eos = StateEquationCole(; sound_speed=40.0, reference_density=rho0, exponent=7)
            for calculator in
                (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation())
                result = evaluate_pair(calculator; dt, eos, wall_speed)
                reversed = evaluate_pair(calculator; dt, eos, wall_speed,
                                         reverse_order=true)
                @test all(isapprox.(values(result), values(reversed)))
                @test result.wall_pressure > 0
                @test result.fluid_pressure > 0
                @test result.ghost_speed ≈ 2wall_speed
                # The default wall interpolation supplies pressure but not a movable
                # pressure DOF. Its frozen acceleration is known during the final solve.
                wall_velocity_increment = -dt * result.density_weight *
                                          result.wall_pressure / result.wall_density^2
                predicted_density = rho0 +
                                    dt * result.density_weight *
                                    (wall_velocity_increment - wall_speed)
                @test result.predicted_density≈predicted_density atol=1e-10 rtol=0
                @test result.updated_density≈rho0 atol=2e-8 rtol=0
            end
        end

        @testset "Active PressureBoundaries in both system orders" begin
            wall_speed = -1.0
            dt = 0.005
            calculator = PressureBoundaries(; time_step=dt)
            # Force enough iterations to test the ordering independently of the legacy
            # stopping criterion used when there are no staged extrapolating walls.
            result = evaluate_pair(calculator; dt, wall_speed, min_iterations=60)
            reversed = evaluate_pair(calculator; dt, wall_speed, min_iterations=60,
                                     reverse_order=true)
            @test all(isapprox.(values(result), values(reversed)))
            @test result.wall_pressure > 0
            @test result.fluid_pressure > 0
            @test result.predicted_density ≈ rho0 - dt * result.density_weight * wall_speed
            @test result.updated_density≈rho0 atol=2e-8 rtol=0
        end
    end

    @testset "EOS wall contact uses the fluid kernel and the current positions" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        rho = 1000.0
        mass = (rho + 1) / (TrixiParticles.kernel(kernel, 0.0, 1.0) +
                TrixiParticles.kernel(kernel, 1.0, 1.0))
        fluid_ic = InitialCondition(; coordinates=zeros(2, 1), particle_spacing=1.0,
                                    density=rho, mass)
        wall_ic = InitialCondition(; coordinates=reshape([1.5, 0.0], 2, 1),
                                   particle_spacing=1.0, density=rho)
        fluid = ImplicitIncompressibleSPHSystem(fluid_ic; smoothing_kernel=kernel,
                                                smoothing_length=1.0, reference_density=rho,
                                                time_step=0.001, min_iterations=1,
                                                max_iterations=1)
        eos = StateEquationCole(; sound_speed=5.0, reference_density=rho, exponent=7)
        model = BoundaryModelDummyParticles(wall_ic.density, wall_ic.mass,
                                            AdamiPressureExtrapolation(), kernel, 0.5;
                                            state_equation=eos)
        motion = PrescribedMotion((coords, t) -> coords + SVector(t, 0.0), t -> true)
        wall = WallBoundarySystem(wall_ic, model; prescribed_motion=motion)
        # Keep the pressure diagonal nonzero even when the moving wall is out of range.
        fixed_ic = InitialCondition(; coordinates=reshape([-1.0, 0.0], 2, 1),
                                    particle_spacing=1.0, density=rho, mass)
        fixed_model = BoundaryModelDummyParticles(fixed_ic.density, fixed_ic.mass,
                                                  PressureZeroing(), kernel, 1.0)
        fixed_wall = WallBoundarySystem(fixed_ic, fixed_model)
        semi = Semidiscretization(fluid, wall, fixed_wall;
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0))
        v, u = ode.u0.x
        dv = zero(v)

        TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
        @test TrixiParticles.has_iisph_wall_contact(fluid)
        @test iszero(only(model.cache.volume)) # The wall kernel cannot see the fluid.

        fluid.pressure .= 0.0
        TrixiParticles.kick!(dv, v, u, ode.p, 1.0)
        cold_pressure = only(fluid.pressure)
        @test !TrixiParticles.has_iisph_wall_contact(fluid)
        # A single warm-started iteration retains this fraction of the old pressure.
        # Running a provisional solve would reset it and discard this increment.
        fluid.pressure .= 40.0
        TrixiParticles.kick!(dv, v, u, ode.p, 1.0)
        @test only(fluid.pressure) - cold_pressure ≈ (1 - fluid.omega) * 20.0

        TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
        @test TrixiParticles.has_iisph_wall_contact(fluid)
        ode.p.semi.interaction_matrix[1, 2] = false
        TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
        @test !TrixiParticles.has_iisph_wall_contact(fluid)
    end

    # Validate helper routines that build the IISPH linear system and update pressures
    @trixi_testset "Solver helper calculations" begin
        smoothing_kernel = Val(:smoothing_kernel)
        TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
        smoothing_length = 0.5

        coordinates = zeros(2, 1)
        velocity = zeros(2, 1)

        @testset "Fluid-fluid coefficients (d_ij, d_ii, sum term)" begin
            mass_a = [2.0]
            density_a = [5.0]
            pressure_a = [0.8]
            ic_a = InitialCondition(; coordinates, velocity, mass=mass_a, density=density_a,
                                    pressure=pressure_a)
            system = ImplicitIncompressibleSPHSystem(ic_a;
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density=6.0,
                                                     time_step=0.5)

            mass_b = [3.0]
            density_b = [4.0]
            pressure_b = [1.2]
            ic_b = InitialCondition(; coordinates, velocity, mass=mass_b, density=density_b,
                                    pressure=pressure_b)
            neighbor_system = ImplicitIncompressibleSPHSystem(ic_b;
                                                              smoothing_kernel,
                                                              smoothing_length,
                                                              reference_density=5.0,
                                                              time_step=0.5)

            system.sum_d_ij_pj[:, 1] .= (0.4, -0.2)
            neighbor_system.sum_d_ij_pj[:, 1] .= (0.05, 0.07)
            neighbor_system.d_ii[:, 1] .= (0.3, 0.1)

            grad_kernel = SVector(1.0, -0.5)

            # d_ij uses neighbor mass/density with kernel gradient sign; sum_term combines
            # both particles’ contributions per Ihmsen eq. 13.
            # With dt=0.5, m_b=3, rho_b=4, and grad_kernel=(1,-0.5), d_ij should be
            # (-0.5^2 * 3 / 4^2)*grad_kernel = (-0.046875, 0.0234375)
            @test isapprox(TrixiParticles.calculate_d_ij(system, neighbor_system, 1,
                                                         grad_kernel, system.time_step),
                           SVector(-0.046875, 0.0234375))
            # sum_term expands to m_j * dot(sum_dik_pk - d_jj * p_j - (sum_djk_pk - d_ji * p_i), grad_kernel)
            # with the prepared values this yields 3 * dot((0.4,-0.2) - (0.3,0.1)*1.2 - ((0.05,0.07) -
            # (-0.046875,0.0234375)*0.8), (1,-0.5)) = 0.615
            @test isapprox(TrixiParticles.calculate_sum_term(system, neighbor_system, 1,
                                                             1, grad_kernel,
                                                             system.time_step),
                           0.615; atol=1.0e-12)
            # d_ii uses particle i density and neighbor mass (eq. 9)
            # Using m_b=3, rho_a=5, grad_kernel[1]=1, dt=0.5 gives -dt^2 * m_b / rho_a^2 * grad = -0.03
            @test isapprox(TrixiParticles.calculate_d_ii(neighbor_system, mass_b[1],
                                                         density_a[1], grad_kernel[1],
                                                         system.time_step),
                           -0.03)
        end

        @testset "Boundary coefficients (PressureMirroring doubles a_ii)" begin
            mass_a = [2.0]
            density_a = [5.0]
            pressure_a = [0.8]
            ic_a = InitialCondition(; coordinates, velocity, mass=mass_a, density=density_a,
                                    pressure=pressure_a)
            system = ImplicitIncompressibleSPHSystem(ic_a;
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density=6.0,
                                                     time_step=0.5)
            system.sum_d_ij_pj[:, 1] .= (0.4, -0.2)

            grad_kernel = SVector(1.0, -0.5)

            struct MockBoundaryModel{DC}
                density_calculator::DC
            end
            struct MockBoundarySystem <: TrixiParticles.AbstractBoundarySystem{2}
                boundary_model :: MockBoundaryModel
                mass           :: Vector{Float64}
                density        :: Vector{Float64}
            end
            Base.ndims(::MockBoundarySystem) = 2
            TrixiParticles.hydrodynamic_mass(system::MockBoundarySystem,
                                             particle) = system.mass[particle]
            TrixiParticles.current_density(::Nothing, system::MockBoundarySystem,
                                           particle) = system.density[particle]
            TrixiParticles.current_density(_, system::MockBoundarySystem,
                                           particle) = system.density[particle]
            TrixiParticles.nparticles(system::MockBoundarySystem) = length(system.mass)

            boundary_system_mirroring = MockBoundarySystem(MockBoundaryModel(PressureMirroring()),
                                                           [1.5], [7.0])
            boundary_system_regular = MockBoundarySystem(MockBoundaryModel(AdamiPressureExtrapolation()),
                                                         [1.5], [7.0])

            # PressureMirroring removes off-diagonal pressure coupling and doubles diagonal term.
            # For the mirroring path: -dt^2 * m_b / rho_b^2 * grad = -0.25^2 * 1.5 / 7^2 * (1,-0.5)
            @test isapprox(TrixiParticles.calculate_d_ij(system, boundary_system_mirroring,
                                                         1,
                                                         grad_kernel, system.time_step),
                           SVector(-0.007653061224489796, 0.003826530612244898))
            regular_d_ii = TrixiParticles.calculate_d_ii(boundary_system_regular,
                                                         boundary_system_regular.boundary_model,
                                                         boundary_system_regular.boundary_model.density_calculator,
                                                         boundary_system_regular.mass[1],
                                                         density_a[1], grad_kernel[2],
                                                         system.time_step)
            # Regular boundary path (no mirroring) uses -dt^2 * m_b / rho_a^2 * grad_kernel[2] = 0.0075
            @test isapprox(regular_d_ii, 0.0075)
            @test isapprox(TrixiParticles.calculate_d_ii(boundary_system_mirroring,
                                                         boundary_system_mirroring.boundary_model,
                                                         boundary_system_mirroring.boundary_model.density_calculator,
                                                         boundary_system_mirroring.mass[1],
                                                         density_a[1], grad_kernel[2],
                                                         system.time_step),
                           2 * regular_d_ii; atol=1.0e-12)
            # Boundary sum_term should still accumulate the fluid-side sum_dij_pj contribution
            @test isapprox(TrixiParticles.calculate_sum_term(system,
                                                             boundary_system_mirroring, 1,
                                                             1, grad_kernel,
                                                             system.time_step),
                           0.75; atol=1.0e-12)
        end

        @testset "pressure_update handles zero diagonal entries" begin
            mass = [1.0, 1.0]
            density = [1000.0, 950.0]
            pressure = [0.0, 10.0]
            coordinates = [0.0 0.1
                           0.0 0.2]
            velocity = zeros(2, 2)
            ic = InitialCondition(; coordinates, velocity, mass, density, pressure)
            system_pressure = ImplicitIncompressibleSPHSystem(ic;
                                                              smoothing_kernel,
                                                              smoothing_length,
                                                              reference_density=1000.0,
                                                              omega=0.4,
                                                              time_step=0.5)
            system_pressure.predicted_density .= [990.0, 1010.0]
            system_pressure.sum_term .= [5.0, -2.0]
            system_pressure.a_ii .= [0.5, 1.0e-10]
            system_pressure.density_error .= [0.0, 99.0]

            semi = DummySemidiscretization()
            # First particle uses standard Jacobi update; second hits the safeguarded zero-a_ii path.
            # For particle 1: (1-omega)*0 + omega/a_ii * (source - sum_term) with omega=0.4,
            # source=(1000-990)=10, a_ii=0.5, sum_term=5 gives pressure 4 and
            # density_error = 997-1000 = -3 (signed deviation).
            # For particle 2: the pressure is clamped to zero, so the stale buffer
            # entry 99.0 from the previous iteration is explicitly cleared.
            mode = TrixiParticles.StandardIISPHPressure()
            relative_error = TrixiParticles.pressure_update(mode, system_pressure,
                                                            system_pressure.pressure,
                                                            system_pressure.reference_density,
                                                            system_pressure.a_ii,
                                                            system_pressure.sum_term,
                                                            system_pressure.omega,
                                                            system_pressure.density_error,
                                                            semi)

            @test isapprox(relative_error, -0.003) # -3 / 1000
            @test isapprox(system_pressure.pressure, [4.0, 0.0])
            @test isapprox(system_pressure.density_error, [-3.0, 0.0])
        end

        @testset "Source term and iteration limits" begin
            mass = [1.0, 1.0]
            density = [1000.0, 950.0]
            pressure = [0.0, 0.0]
            coordinates = [0.0 0.1
                           0.0 0.2]
            velocity = zeros(2, 2)
            ic = InitialCondition(; coordinates, velocity, mass, density, pressure)
            system_iters = ImplicitIncompressibleSPHSystem(ic;
                                                           smoothing_kernel,
                                                           smoothing_length,
                                                           reference_density=1000.0,
                                                           omega=0.6,
                                                           max_error=0.25,
                                                           min_iterations=3,
                                                           max_iterations=7,
                                                           time_step=0.25)
            system_iters.predicted_density .= [1005.0, 990.0]

            # Source term equals reference density minus predicted density for each particle
            # Here: 1000-1005 = -5 and 1000-990 = 10, matching the solver source definition
            @test TrixiParticles.iisph_source_term(system_iters, 1) == -5.0
            @test TrixiParticles.iisph_source_term(system_iters, 2) == 10.0
            # Accessors expose IISPH-specific particle counts and solver thresholds
            @test TrixiParticles.n_iisph_particles(system_iters) == 2
            @test TrixiParticles.maximum_iisph_error(system_iters) == 0.25
            @test TrixiParticles.minimum_iisph_iterations(system_iters) == 3
            @test TrixiParticles.maximum_iisph_iterations(system_iters) == 7
        end
    end
end
