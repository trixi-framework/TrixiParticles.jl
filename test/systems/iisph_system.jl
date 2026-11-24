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

            # The relaxed Jacobi scheme can diverge for omega >= 1
            error_str8 = "`omega` must be a number between 0 and 1 (exclusive), since the relaxed Jacobi scheme can diverge for `omega >= 1`"
            for invalid_omega in (0.0, 1.0, 1.5)
                @test_throws ArgumentError(error_str8) ImplicitIncompressibleSPHSystem(initial_condition;
                                                                                       smoothing_kernel,
                                                                                       smoothing_length,
                                                                                       reference_density,
                                                                                       omega=invalid_omega,
                                                                                       time_step=0.001)
            end

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

    @testset "Wall pressure solve is independent of RHS history" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        h = 1.0
        dt = 1e-3
        rho = 1000.0
        viscosity = ViscosityAdami(nu=0.1)
        x = (0.0, 1.5, 1.75)
        weights = [TrixiParticles.kernel(kernel, abs(a - b), h) for a in x, b in x]
        masses = rho / (weights[1, 1] + 2weights[1, 2] + 3weights[1, 3]) .* [1, 2, 3]

        function make_wall(i, calculator; wall_viscosity=nothing)
            ic = InitialCondition(; coordinates=reshape([x[i], 0.0], 2, 1), density=rho,
                                  mass=masses[i], particle_spacing=h)
            model = BoundaryModelDummyParticles(ic.density, ic.mass, calculator, kernel, h;
                                                viscosity=wall_viscosity)
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
            wall = make_wall(2, calculator; wall_viscosity=viscosity)
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
                        dv=copy(TrixiParticles.wrap_v(dv, fluid, ode.p.semi)))
            end

            # The wall samples only the fluid particle. The normal component of the
            # relative velocity between wall and fluid is -0.005.
            dynamic_pressure = calculator isa BernoulliPressureExtrapolation ?
                               rho * 0.005^2 / 2 : 0.0

            baseline = wall_rhs()
            @test baseline.wall_pressure > calculator.pressure_offset
            # The wall pressure is extrapolated from the solved fluid pressure
            @test baseline.wall_pressure ≈
                  baseline.fluid_pressure + calculator.pressure_offset + dynamic_pressure
            # Without a state equation, the wall density is constant
            @test baseline.wall_density == rho
            # Viscous force of the wall moving in y-direction with the no-slip velocity
            # 2 * (0, 1) - (0.005, 0)
            @test baseline.dv[2, 1] > 0
            @test model.cache.wall_velocity[:, 1] ≈ [-0.005, 2.0]
            @test wall_rhs() == baseline

            # Neither the pressures nor the no-slip velocity of the previous evaluation
            # affect the result
            fluid.pressure[1] = 1234.0
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
                               gravity=0.0, reverse_order=false, min_iterations=2,
                               max_iterations=100, max_error=1e-9, omega=0.4,
                               fluid_viscosity=nothing, wall_viscosity=nothing)
            # Equal masses give both particles the specified kernel-summation density.
            mass = rho / weight
            fluid_ic = InitialCondition(; coordinates=zeros(2, 1), density=rho0, mass,
                                        particle_spacing=h)
            wall_ic = InitialCondition(; coordinates=reshape([h, 0.0], 2, 1),
                                       density=rho0, mass, particle_spacing=h)
            fluid = ImplicitIncompressibleSPHSystem(fluid_ic; smoothing_kernel=kernel,
                                                    smoothing_length=h, time_step=dt,
                                                    reference_density=rho0, omega,
                                                    min_iterations, max_iterations,
                                                    max_error, viscosity=fluid_viscosity,
                                                    acceleration=(gravity, 0.0))
            model = BoundaryModelDummyParticles(wall_ic.density, wall_ic.mass, calculator,
                                                kernel, h; viscosity=wall_viscosity)
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

        @testset "Hydrostatic wall pressure" begin
            # Gravity along the pair, away from and towards the wall at distance `h`.
            # The compressed fluid has a positive pressure in both cases.
            rho = 1005.0
            for calculator in
                (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation()),
                gravity in (-10.0, 10.0)

                result = evaluate_pair(calculator; rho, gravity)
                @test result.fluid_pressure > 0
                # Hydrostatic extrapolation from the only fluid neighbor at distance `h`
                @test result.wall_pressure ≈ result.fluid_pressure + rho * gravity * h
                @test result.wall_density == rho0
                @test result.updated_density≈rho0 atol=2e-8 rtol=0
            end
        end

        @testset "Normal wall motion and self-consistent wall pressure" begin
            wall_speed = -1.0
            dt = 0.005
            # Configure a no-slip velocity without a viscous force: zero wall viscosity
            # makes the harmonic viscosity of the wall pair zero.
            options = (; dt, wall_speed, fluid_viscosity=ViscosityAdami(nu=0.1),
                       wall_viscosity=ViscosityAdami(nu=0.0))
            for calculator in
                (AdamiPressureExtrapolation(), BernoulliPressureExtrapolation())
                result = evaluate_pair(calculator; options...)
                reversed = evaluate_pair(calculator; options..., reverse_order=true)
                @test all(isapprox.(values(result), values(reversed)))
                @test result.wall_pressure > 0
                @test result.fluid_pressure > 0
                @test result.ghost_speed ≈ 2wall_speed
                # The wall pressure is part of the pressure solve, so the prediction
                # only contains the wall motion.
                @test result.predicted_density ≈
                      rho0 - dt * result.density_weight * wall_speed
                # With one fluid neighbor, no gravity and no offset, the extrapolated
                # wall pressure equals the solved fluid pressure (plus the dynamic
                # pressure of the normal wall motion with Bernoulli).
                dynamic_pressure = calculator isa BernoulliPressureExtrapolation ?
                                   rho0 * wall_speed^2 / 2 : 0.0
                @test result.wall_pressure ≈ result.fluid_pressure + dynamic_pressure
                @test result.wall_density == rho0
                # With equal densities, the density balance determines the pressure sum
                @test result.fluid_pressure + result.wall_pressure ≈
                      -wall_speed * rho0^2 / (dt * result.density_weight)
                @test result.updated_density≈rho0 atol=2e-8 rtol=0
            end
        end

        @testset "Viscous wall force in the density balance" begin
            # The no-slip velocity -2 has a component along the pair, so the viscous
            # force of the wall enters the density balance.
            wall_speed = -1.0
            dt = 0.005
            calculator = AdamiPressureExtrapolation()
            inviscid = evaluate_pair(calculator; dt, wall_speed)
            options = (; dt, wall_speed, fluid_viscosity=ViscosityAdami(nu=10.0),
                       wall_viscosity=ViscosityAdami(nu=10.0))
            viscous = evaluate_pair(calculator; options...)
            reversed = evaluate_pair(calculator; options..., reverse_order=true)
            @test all(isapprox.(values(viscous), values(reversed)))
            # The pressure compensates the viscous force
            @test !isapprox(viscous.fluid_pressure, inviscid.fluid_pressure, rtol=0.05)
            @test viscous.wall_pressure ≈ viscous.fluid_pressure
            @test viscous.updated_density≈rho0 atol=2e-8 rtol=0
        end

        @testset "Active PressureBoundaries in both system orders" begin
            wall_speed = -1.0
            dt = 0.005
            calculator = PressureBoundaries(; time_step=dt)
            # Force enough iterations to test the ordering independently of the standard
            # stopping criterion used without pressure-extrapolating walls.
            result = evaluate_pair(calculator; dt, wall_speed, min_iterations=60)
            reversed = evaluate_pair(calculator; dt, wall_speed, min_iterations=60,
                                     reverse_order=true)
            @test all(isapprox.(values(result), values(reversed)))
            @test result.wall_pressure > 0
            @test result.fluid_pressure > 0
            @test result.predicted_density ≈ rho0 - dt * result.density_weight * wall_speed
            @test result.updated_density≈rho0 atol=2e-8 rtol=0
        end

        @testset "Extrapolated wall force in a PressureBoundaries row" begin
            tp = TrixiParticles
            dt = 0.005
            w0 = tp.kernel(kernel, 0.0, h)
            w1 = tp.kernel(kernel, h, h)
            extrapolated_mass = 1200.0
            # Set the raw summation densities to rho0 + 9 in the fluid row and
            # rho0 + 5 in the solved wall row, with noncollinear wall forces.
            fluid_mass = (rho0 + 9 - extrapolated_mass * w1 - (rho0 + 5) * w1 / w0) /
                         (w0 - w1^2 / w0)
            solved_wall_mass = (rho0 + 5 - fluid_mass * w1) / w0
            wall_position = SVector(0.6, 0.8)
            solved_wall_position = SVector(1.0, 0.0)

            function point(coords, mass)
                return InitialCondition(; coordinates=reshape(collect(coords), 2, 1),
                                        density=rho0, mass, particle_spacing=h)
            end
            initial = point(SVector(0.0, 0.0), fluid_mass)
            fluid = ImplicitIncompressibleSPHSystem(initial; smoothing_kernel=kernel,
                                                    smoothing_length=h, time_step=dt,
                                                    reference_density=rho0, omega=0.4,
                                                    max_iterations=1500, max_error=1e-9)
            function make_wall(coords, wall_mass, calculator)
                ic = point(coords, wall_mass)
                model = BoundaryModelDummyParticles(ic.density, ic.mass, calculator,
                                                    kernel, h)
                return WallBoundarySystem(ic, model)
            end
            solved_wall = make_wall(solved_wall_position, solved_wall_mass,
                                    PressureBoundaries(; time_step=dt))
            extrapolated_wall = make_wall(wall_position, extrapolated_mass,
                                          AdamiPressureExtrapolation())
            interactions = Bool[1 1 1; 1 1 0; 1 0 1]
            semi = Semidiscretization(fluid, solved_wall, extrapolated_wall;
                                      parallelization_backend=SerialBackend(),
                                      interaction_matrix=interactions)
            ode = semidiscretize(semi, (0.0, dt))
            v, u = ode.u0.x
            dv = zero(v)
            tp.kick!(dv, v, u, ode.p, 0.0)

            model = extrapolated_wall.boundary_model
            grad_extrapolated = tp.smoothing_kernel_grad(fluid, -wall_position, h, 1)
            grad_solved = tp.smoothing_kernel_grad(fluid, -solved_wall_position, h, 1)
            cache = solved_wall.boundary_model.cache
            @test only(model.pressure) > 0
            @test only(solved_wall.boundary_model.pressure) > 0
            # The extrapolated wall samples only the fluid particle at distance `h`.
            @test only(model.pressure) ≈ only(fluid.pressure)
            @test only(model.cache.density) == rho0
            # At rest, the prediction does not change the density of the solved wall row.
            # The force of the extrapolated wall enters through the fluid displacement.
            @test only(cache.predicted_density) ≈ only(cache.density)
            # Measure both constraints using the actual fluid force, not the PPE sums.
            fluid_acceleration = SVector{2}(tp.wrap_v(dv, fluid, ode.p.semi))
            wall_row_density = only(cache.density) +
                               dt^2 * fluid_mass * dot(fluid_acceleration, grad_solved)
            @test wall_row_density≈rho0 atol=2e-8 rtol=0
            fluid_row_density = only(fluid.density) +
                                dt^2 * dot(fluid_acceleration,
                                    solved_wall_mass * grad_solved +
                                    extrapolated_mass * grad_extrapolated)
            @test fluid_row_density≈rho0 atol=2e-8 rtol=0
        end
    end

    @testset "Wall state equations and custom interactions are rejected" begin
        tp = TrixiParticles
        kernel = SchoenbergCubicSplineKernel{2}()
        rho = 1000.0
        h = 1.0
        function point(x)
            return InitialCondition(; coordinates=reshape([x, 0.0], 2, 1), density=rho,
                                    mass=rho, particle_spacing=h)
        end
        fluid = ImplicitIncompressibleSPHSystem(point(0.0); smoothing_kernel=kernel,
                                                smoothing_length=h, time_step=0.001,
                                                reference_density=rho)
        eos = StateEquationCole(; sound_speed=40.0, reference_density=rho, exponent=7)
        function make_wall(calculator; state_equation=nothing)
            ic = point(1.0)
            model = BoundaryModelDummyParticles(ic.density, ic.mass, calculator, kernel, h;
                                                state_equation)
            return WallBoundarySystem(ic, model)
        end
        custom_interaction(args...; kwargs...) = nothing
        fluid_ignores_wall = Bool[1 0; 1 1]

        @testset "$(nameof(typeof(calculator)))" for calculator in
                                                     (AdamiPressureExtrapolation(),
                                                      BernoulliPressureExtrapolation())
            eos_wall = make_wall(calculator; state_equation=eos)
            @test_throws "must not have a state equation" Semidiscretization(fluid,
                                                                             eos_wall)
            # Without forces on the fluid, the wall does not enter the pressure solve
            semi = Semidiscretization(fluid, eos_wall;
                                      interaction_matrix=fluid_ignores_wall)
            @test !tp.has_iisph_wall_extrapolation(semi)

            wall = make_wall(calculator)
            @test_throws "must use the default interaction" Semidiscretization(fluid, wall;
                                                                               interaction_matrix=[true custom_interaction
                                                                                                   true true])
            @test tp.has_iisph_wall_extrapolation(Semidiscretization(fluid, wall))
        end

        semi = Semidiscretization(fluid, make_wall(PressureZeroing()))
        @test !tp.has_iisph_wall_extrapolation(semi)
    end

    @testset "Zero start with pressure-extrapolating walls" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        rho = 1000.0
        mass = (rho + 1) / (TrixiParticles.kernel(kernel, 0.0, 1.0) +
                TrixiParticles.kernel(kernel, 1.0, 1.0))
        function point(x)
            return InitialCondition(; coordinates=reshape([x, 0.0], 2, 1),
                                    particle_spacing=1.0, density=rho, mass)
        end
        fluid = ImplicitIncompressibleSPHSystem(point(0.0); smoothing_kernel=kernel,
                                                smoothing_length=1.0, reference_density=rho,
                                                time_step=0.001, min_iterations=1,
                                                max_iterations=1)
        function make_wall(x, calculator)
            ic = point(x)
            model = BoundaryModelDummyParticles(ic.density, ic.mass, calculator, kernel,
                                                1.0)
            return WallBoundarySystem(ic, model)
        end
        # Asymmetric wall positions, so that the kernel gradients do not cancel in a_ii
        semi = Semidiscretization(fluid, make_wall(1.5, AdamiPressureExtrapolation()),
                                  make_wall(-1.0, PressureZeroing());
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0))
        v, u = ode.u0.x
        dv = zero(v)

        function single_iteration(initial_pressure)
            fluid.pressure .= initial_pressure
            TrixiParticles.kick!(dv, v, u, ode.p, 0.0)
            return only(fluid.pressure)
        end

        # The previous pressure does not affect the solve with an extrapolating wall
        @test single_iteration(0.0) > 0
        @test single_iteration(40.0) == single_iteration(0.0)

        # Without the interaction with the extrapolating wall, a single iteration
        # retains this fraction of the warm start with half the previous pressure.
        ode.p.semi.interaction_matrix[1, 2] = false
        cold_pressure = single_iteration(0.0)
        @test cold_pressure > 0
        @test single_iteration(40.0) - cold_pressure ≈ (1 - fluid.omega) * 20.0
    end

    @testset "Wall shared with a Shepard-corrected EDAC system" begin
        # The wall extrapolates the hydrostatic pressure with the EDAC density, which is
        # only final after the density correction. The pressure solve must use this density.
        tp = TrixiParticles
        kernel = SchoenbergCubicSplineKernel{2}()
        h = 1.0
        rho = 1000.0
        dt = 0.005
        gravity = (-9.81, 0.0)
        wall_coords = [0.0 0.0 0.0
                       -1.0 0.0 1.0]
        edac_coords = [1.0 1.0 1.0 2.0 2.0 2.0
                       -1.0 0.0 1.0 -1.0 0.0 1.0]
        iisph_position = SVector(-1.0, 0.0)
        wall_weights = [tp.kernel(kernel, norm(iisph_position - wall_coords[:, b]), h)
                        for b in axes(wall_coords, 2)]
        # Compress the IISPH particle, so that it has a positive pressure
        iisph_mass = (rho + 5 - rho * sum(wall_weights)) / tp.kernel(kernel, 0.0, h)
        iisph_ic = InitialCondition(; coordinates=reshape(collect(iisph_position), 2, 1),
                                    density=rho, mass=iisph_mass, particle_spacing=h)
        fluid = ImplicitIncompressibleSPHSystem(iisph_ic; smoothing_kernel=kernel,
                                                smoothing_length=h, reference_density=rho,
                                                time_step=dt, acceleration=gravity,
                                                max_error=1e-10, max_iterations=1000)
        edac_ic = InitialCondition(; coordinates=edac_coords, density=rho, mass=rho,
                                   particle_spacing=h, pressure=100.0)
        edac = EntropicallyDampedSPHSystem(edac_ic; smoothing_kernel=kernel,
                                           smoothing_length=h, sound_speed=10.0,
                                           density_calculator=SummationDensity(),
                                           density_correction=ShepardKernelCorrection(),
                                           acceleration=gravity)
        wall_ic = InitialCondition(; coordinates=wall_coords, density=rho, mass=rho,
                                   particle_spacing=h)
        model = BoundaryModelDummyParticles(wall_ic.density, wall_ic.mass,
                                            AdamiPressureExtrapolation(), kernel, h)
        wall = WallBoundarySystem(wall_ic, model)
        # The two fluids only interact through the wall
        interactions = Bool[1 0 1; 0 1 1; 1 1 1]
        semi = Semidiscretization(fluid, edac, wall; interaction_matrix=interactions,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, dt))
        v, u = ode.u0.x
        dv = zero(v)
        tp.kick!(dv, v, u, ode.p, 0.0)

        @test only(fluid.pressure) > 0
        # Apply the continuity linearization to the actual fluid RHS
        acceleration = SVector{2}(tp.wrap_v(dv, fluid, ode.p.semi))
        updated_density = only(fluid.density)
        for b in axes(wall_coords, 2)
            pos_diff = iisph_position - wall_coords[:, b]
            grad = tp.kernel_grad(kernel, pos_diff, norm(pos_diff), h)
            updated_density += dt^2 * rho * dot(acceleration, grad)
        end
        @test updated_density≈rho atol=1e-8 rtol=0
    end

    @testset "Wall pressure is the extrapolation of the solved fluid pressure" begin
        tp = TrixiParticles
        rho = 1000.0
        g = 9.81
        dt = 1e-3
        h = 0.1
        kernel = SchoenbergCubicSplineKernel{2}()
        tank = RectangularTank(h, (0.5, 0.5), (0.5, 0.8), rho; n_layers=2,
                               acceleration=(0.0, -g), coordinates_eltype=Float64)
        # Compress the fluid slightly to get pressure in the bulk as well
        tank.fluid.coordinates .*= 0.995
        @testset "$(nameof(typeof(calculator)))" for calculator in
                                                     (AdamiPressureExtrapolation(pressure_offset=10.0),
                                                      BernoulliPressureExtrapolation(pressure_offset=10.0))
            fluid = ImplicitIncompressibleSPHSystem(tank.fluid; smoothing_kernel=kernel,
                                                    smoothing_length=h, time_step=dt,
                                                    reference_density=rho,
                                                    acceleration=(0.0, -g),
                                                    max_error=1e-8, max_iterations=5000,
                                                    viscosity=ViscosityAdami(nu=1e-2))
            model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                calculator, kernel, h;
                                                viscosity=ViscosityAdami(nu=1e-2))
            # A resting wall flagged as moving activates the Bernoulli dynamic pressure
            motion = PrescribedMotion((coords, t) -> coords, t -> true)
            wall = WallBoundarySystem(tank.boundary, model; prescribed_motion=motion)
            semi = Semidiscretization(fluid, wall; parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, dt))
            v, u = ode.u0.x
            # Nonzero velocities activate the viscous and the Bernoulli wall terms
            v_fluid = tp.wrap_v(v, fluid, ode.p.semi)
            v_fluid[1, :] .= 0.1 .* sin.(1:size(v_fluid, 2))
            dv = zero(v)
            tp.kick!(dv, v, u, ode.p, 0.0)
            @test tp.has_iisph_wall_extrapolation(ode.p.semi)

            coords = tp.wrap_u(u, fluid, ode.p.semi)
            wall_coords = wall.coordinates
            # Extrapolate the solved fluid pressure to every wall particle
            for b in axes(wall_coords, 2)
                numerator = 0.0
                volume = 0.0
                for f in axes(coords, 2)
                    pos_diff = wall_coords[:, b] - coords[:, f]
                    weight = tp.kernel(kernel, norm(pos_diff), h)
                    weight > 0 || continue
                    normal_velocity = dot(-v_fluid[:, f], pos_diff) / norm(pos_diff)
                    dynamic = calculator isa BernoulliPressureExtrapolation ?
                              fluid.density[f] * normal_velocity^2 / 2 : 0.0
                    extrapolated = calculator.pressure_offset + fluid.pressure[f] +
                                   dynamic +
                                   fluid.density[f] * dot(SVector(0.0, -g), pos_diff)
                    numerator += extrapolated * weight
                    volume += weight
                end
                expected = volume > eps() ? numerator / volume : 0.0
                @test model.pressure[b]≈expected rtol=1e-12 atol=1e-9
            end
            # The wall density is constant
            @test model.cache.density == tank.boundary.density

            # The actual RHS satisfies the linearized density balance
            velocity = v_fluid .+ dt .* tp.wrap_v(dv, fluid, ode.p.semi)
            for i in axes(coords, 2)
                density = fluid.density[i]
                for j in axes(coords, 2)
                    pos_diff = coords[:, i] - coords[:, j]
                    distance = norm(pos_diff)
                    grad = tp.smoothing_kernel_grad(fluid, pos_diff, distance, i)
                    density += dt * fluid.mass[j] *
                               dot(velocity[:, i] - velocity[:, j], grad)
                end
                for b in axes(wall_coords, 2)
                    pos_diff = coords[:, i] - wall_coords[:, b]
                    distance = norm(pos_diff)
                    grad = tp.smoothing_kernel_grad(fluid, pos_diff, distance, i)
                    density += dt * model.hydrodynamic_mass[b] * dot(velocity[:, i], grad)
                end
                residual = density - rho
                if fluid.pressure[i] > 0
                    @test abs(residual) < 1e-6
                else
                    @test residual < 1e-6
                end
            end
        end
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
            mode = TrixiParticles.StandardIISPHSolve()
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

        @testset "Density residuals of both solve variants" begin
            # Particles with positive pressure, and with zero pressure that are
            # overdense or underdense after the update
            n_particles = 3
            ic = InitialCondition(; coordinates=zeros(2, n_particles),
                                  velocity=zeros(2, n_particles), mass=ones(n_particles),
                                  density=fill(1000.0, n_particles))
            system = ImplicitIncompressibleSPHSystem(ic; smoothing_kernel,
                                                     smoothing_length,
                                                     reference_density=1000.0,
                                                     omega=0.4, time_step=0.5)
            semi = DummySemidiscretization()

            function update(mode)
                system.pressure .= [2.0, 0.0, 0.0]
                system.predicted_density .= [990.0, 1010.0, 990.0]
                system.sum_term .= [5.0, -2.0, -2.0]
                system.a_ii .= [0.5, -1.0, -1.0]
                return TrixiParticles.pressure_update(mode, system, system.pressure,
                                                      system.reference_density,
                                                      system.a_ii, system.sum_term,
                                                      system.omega, system.density_error,
                                                      semi)
            end

            # Signed density error of Ihmsen et al. (2013) with the updated pressure on
            # the diagonal, ignoring particles with zero pressure
            @test update(TrixiParticles.StandardIISPHSolve()) ≈ (-2.4 + 4.8) / 1000
            @test system.pressure ≈ [5.2, 3.2, 0.0]
            @test system.density_error ≈ [-2.4, 4.8, 0.0]

            # The projected residual at the old iterate counts the absolute error with
            # positive pressure, and only overdensity with zero pressure
            @test update(TrixiParticles.WallExtrapolationIISPHSolve()) ≈ (4 + 8) / 1000
            @test system.pressure ≈ [5.2, 3.2, 0.0]
            @test system.density_error ≈ [4.0, 8.0, 0.0]
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
