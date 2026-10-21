@testset verbose=true "Dummy Particles" begin
    @testset "show" begin
        boundary_model = BoundaryModelDummyParticles([1000.0],
                                                     [1.0],
                                                     SummationDensity(),
                                                     SchoenbergCubicSplineKernel{2}(),
                                                     0.1)

        expected_repr = "BoundaryModelDummyParticles(SummationDensity, Nothing)"
        @test repr(boundary_model) == expected_repr
    end

    @testset "Pressure clipping" begin
        state_equation = StateEquationCole(sound_speed=10.0,
                                           reference_density=1000.0,
                                           exponent=7,
                                           clip_negative_pressure=false)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.1

        boundary_model_clip = BoundaryModelDummyParticles([1000.0], [1.0],
                                                          SummationDensity(),
                                                          smoothing_kernel,
                                                          smoothing_length,
                                                          state_equation=state_equation,
                                                          clip_negative_pressure=true)
        boundary_model_no_clip = BoundaryModelDummyParticles([1000.0], [1.0],
                                                             SummationDensity(),
                                                             smoothing_kernel,
                                                             smoothing_length,
                                                             state_equation=state_equation)

        @test TrixiParticles.clip_negative_pressure(boundary_model_clip)
        @test !TrixiParticles.clip_negative_pressure(boundary_model_no_clip)

        TrixiParticles.apply_state_equation!(boundary_model_clip, 900.0, 1)
        TrixiParticles.apply_state_equation!(boundary_model_no_clip, 900.0, 1)

        @test boundary_model_clip.pressure[1] == 0
        @test boundary_model_no_clip.pressure[1] < 0

        # Test that clipping behavior is preserved through `adapt_structure`.
        adapted_model_clip = TrixiParticles.Adapt.adapt(Array, boundary_model_clip)
        adapted_model_no_clip = TrixiParticles.Adapt.adapt(Array, boundary_model_no_clip)
        @test TrixiParticles.clip_negative_pressure(adapted_model_clip)
        @test !TrixiParticles.clip_negative_pressure(adapted_model_no_clip)

        boundary_model_adami_clip = BoundaryModelDummyParticles([1000.0], [1.0],
                                                                AdamiPressureExtrapolation(),
                                                                smoothing_kernel,
                                                                smoothing_length,
                                                                state_equation=state_equation,
                                                                clip_negative_pressure=true)
        boundary_model_adami_no_clip = BoundaryModelDummyParticles([1000.0], [1.0],
                                                                   AdamiPressureExtrapolation(),
                                                                   smoothing_kernel,
                                                                   smoothing_length,
                                                                   state_equation=state_equation)

        for model in (boundary_model_adami_clip, boundary_model_adami_no_clip)
            model.cache.volume[1] = 1
            model.pressure[1] = -1
            TrixiParticles.compute_adami_density!(model, nothing, nothing, 1)
        end

        @test boundary_model_adami_clip.pressure[1] == 0
        @test boundary_model_adami_no_clip.pressure[1] == -1
    end

    @testset "Bernoulli dynamic pressure" begin
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0

        boundary = InitialCondition(; coordinates=[0.0; 0.0;;],
                                    density=[3.0], mass=[1.0],
                                    particle_spacing=1.0)
        movement = PrescribedMotion((x, t) -> x, t -> true)
        boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                     BernoulliPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)
        boundary_system = WallBoundarySystem(boundary, boundary_model;
                                             prescribed_motion=movement)
        boundary_system.cache.velocity[:, 1] .= [5.0, 0.0]

        fluid = InitialCondition(; coordinates=[2.0; 0.0;;],
                                 velocity=[1.0; 0.0;;],
                                 density=[3.0], mass=[1.0],
                                 particle_spacing=1.0)
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length,
                                                   density_calculator=ContinuityDensity(),
                                                   state_equation=nothing)
        v_fluid = vcat(fluid.velocity, fluid.density')

        dynamic_pressure = TrixiParticles.dynamic_pressure(BernoulliPressureExtrapolation(),
                                                           zeros(1, 1), v_fluid,
                                                           boundary_system, fluid_system,
                                                           1, 1, SVector(2.0, 0.0),
                                                           2.0, 3.0)

        # 0.5 * rho * v_normal^2 with v_normal = 5 - 1.
        @test dynamic_pressure == 24.0
    end

    @testset verbose=true "Viscosity Adami/Bernoulli: Wall Velocity" begin
        particle_spacing = 0.1

        # Boundary particles in fluid compact support
        boundary_1 = RectangularShape(particle_spacing,
                                      (10, 1),
                                      (0.0, 0.2),
                                      density=257.0)
        boundary_2 = RectangularShape(particle_spacing,
                                      (10, 1),
                                      (0.0, 0.1),
                                      density=257.0)

        # Boundary particles out of fluid compact support
        boundary_3 = RectangularShape(particle_spacing,
                                      (10, 1),
                                      (0.0, 0.0),
                                      density=257.0)

        boundary = union(boundary_1, boundary_2, boundary_3)
        v_boundary = zeros(2, TrixiParticles.nparticles(boundary))

        particles_in_compact_support = length(boundary_1.mass) + length(boundary_2.mass)

        # Define fluid particles
        fluid = RectangularShape(particle_spacing,
                                 (16, 5),
                                 (-0.3, 0.3),
                                 density=257.0,
                                 loop_order=:x_first)

        # Simulation parameters
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2 * particle_spacing
        viscosity = ViscosityAdami(nu=1e-6)
        state_equation = StateEquationCole(sound_speed=10.0,
                                           reference_density=257.0,
                                           exponent=7)

        # Define pressure extrapolation methods to test
        boundary_model_adami = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                                           AdamiPressureExtrapolation(),
                                                           smoothing_kernel,
                                                           smoothing_length; state_equation,
                                                           viscosity)
        boundary_model_bernoulli = BoundaryModelDummyParticles(boundary.density,
                                                               boundary.mass,
                                                               BernoulliPressureExtrapolation(),
                                                               smoothing_kernel,
                                                               smoothing_length;
                                                               state_equation, viscosity)

        boundary_systems = [
            WallBoundarySystem(boundary, boundary_model_adami),
            WallBoundarySystem(boundary, boundary_model_bernoulli),
            RigidBodySystem(boundary; boundary_model=boundary_model_adami,
                            particle_spacing),
            RigidBodySystem(boundary; boundary_model=boundary_model_bernoulli,
                            particle_spacing),
            TotalLagrangianSPHSystem(boundary; smoothing_kernel,
                                     smoothing_length, young_modulus=1e6,
                                     poisson_ratio=0.3,
                                     boundary_model=boundary_model_adami),
            TotalLagrangianSPHSystem(boundary; smoothing_kernel,
                                     smoothing_length, young_modulus=1e6,
                                     poisson_ratio=0.3,
                                     boundary_model=boundary_model_bernoulli)
        ]

        # Create fluid system
        fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                                   smoothing_length,
                                                   density_calculator=SummationDensity(),
                                                   state_equation)

        velocities = [
            [0.0; -1.0],
            [1.0; 1.0],
            [-1.0; 0.0],
            [0.7; 0.2],
            [0.3; 0.8]
        ]

        semi = DummySemidiscretization()

        for boundary_system in boundary_systems
            system_name = boundary_system |> typeof |> nameof
            density_calculator = boundary_system.boundary_model.density_calculator |>
                                 typeof |> nameof
            @testset "$system_name with $density_calculator" begin
                boundary_model = boundary_system.boundary_model

                @testset "Wall Velocity $v_fluid" for v_fluid in velocities
                    viscosity = boundary_model.viscosity
                    pressure = boundary_model.pressure
                    volume = boundary_model.cache.volume

                    # Reset cache and perform pressure extrapolation
                    TrixiParticles.reset_cache!(boundary_model.cache,
                                                boundary_model.viscosity)
                    TrixiParticles.boundary_pressure_extrapolation!(Val(true),
                                                                    boundary_model,
                                                                    boundary_system,
                                                                    fluid_system,
                                                                    boundary.coordinates,
                                                                    fluid.coordinates,
                                                                    v_boundary,
                                                                    v_fluid .*
                                                                    ones(size(fluid.coordinates)),
                                                                    semi)

                    # Compute wall velocities
                    for particle in TrixiParticles.eachparticle(boundary_system)
                        if volume[particle] > eps()
                            pressure[particle] /= volume[particle]
                            TrixiParticles.compute_wall_velocity!(viscosity,
                                                                  boundary_system,
                                                                  v_boundary,
                                                                  particle)
                        end
                    end

                    # Expected wall velocities
                    v_wall = zeros(size(boundary.coordinates))
                    v_wall[:, 1:particles_in_compact_support] .= -v_fluid

                    @test isapprox(boundary_model.cache.wall_velocity,
                                   v_wall)
                end

                scale_factors = [1.0, 0.5, 0.7, 1.8, 67.5]

                # For a constant velocity profile (each fluid particle has the same velocity),
                # the wall velocity is `v_wall = -v_fluid` (see eq. 22 in Adami_2012).
                # With a staggered velocity profile, we can test the smoothed velocity field
                # for a variable velocity profile.
                @testset "Wall Velocity Staggered: Factor $scale" for scale in scale_factors
                    viscosity = boundary_model.viscosity
                    pressure = boundary_model.pressure
                    volume = boundary_model.cache.volume

                    # Create a staggered velocity profile
                    v_fluid = zeros(size(fluid.coordinates))
                    for i in TrixiParticles.eachparticle(fluid_system)
                        if mod(i, 2) == 1
                            v_fluid[:, i] .= scale
                        end
                    end

                    # Reset cache and perform pressure extrapolation
                    TrixiParticles.reset_cache!(boundary_model.cache,
                                                boundary_model.viscosity)
                    TrixiParticles.boundary_pressure_extrapolation!(Val(true),
                                                                    boundary_model,
                                                                    boundary_system,
                                                                    fluid_system,
                                                                    boundary.coordinates,
                                                                    fluid.coordinates,
                                                                    v_boundary, v_fluid,
                                                                    semi)

                    # Compute wall velocities
                    for particle in TrixiParticles.eachparticle(boundary_system)
                        if volume[particle] > eps()
                            pressure[particle] /= volume[particle]
                            TrixiParticles.compute_wall_velocity!(viscosity,
                                                                  boundary_system,
                                                                  v_boundary, particle)
                        end
                    end

                    # Expected wall velocities
                    v_wall = zeros(size(boundary.coordinates))

                    # First boundary row
                    for i in 1:length(boundary_1.mass)
                        if mod(i, 2) == 1
                            # Particles with a diagonal distance to a fluid particle with v_fluid > 0.0
                            v_wall[:, i] .= -0.42040669416720744 * scale
                        else
                            # Particles with an orthogonal distance to a fluid particle with v_fluid > 0.0
                            v_wall[:, i] .= -0.5795933058327924 * scale
                        end
                    end

                    # Second boundary row
                    for i in (length(boundary_1.mass) + 1):particles_in_compact_support
                        if mod(i, 2) == 1
                            # Particles with a diagonal distance to a fluid particle with v_fluid > 0.0
                            v_wall[:, i] .= -0.12101100073462243 * scale
                        else
                            # Particles with an orthogonal distance to a fluid particle with v_fluid > 0.0
                            v_wall[:, i] .= -0.8789889992653775 * scale
                        end
                    end

                    @test isapprox(boundary_system.boundary_model.cache.wall_velocity,
                                   v_wall)
                end
            end
        end
    end

    @testset "Pressure Extrapolation Adami" begin
        particle_spacing = 0.1
        n_particles = 10
        n_layers = 2
        width = particle_spacing * n_particles
        height = particle_spacing * n_particles
        density = 257

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.5 * particle_spacing
        state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                           exponent=7)

        tank1 = RectangularTank(particle_spacing, (width, height), (width, height), density;
                                n_layers, faces=(true, true, true, false))

        boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                     tank1.boundary.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length;
                                                     state_equation)

        boundary_system = WallBoundarySystem(tank1.boundary, boundary_model)
        viscosity = boundary_system.boundary_model.viscosity

        semi = DummySemidiscretization()

        # In this testset, we verify that pressures in a static tank are extrapolated correctly.
        # Use constant density equal to the reference density of the state equation,
        # so that the pressure is constant zero. Then we test that the extrapolation also yields zero.
        @testset "Constant Zero Pressure" begin
            fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid; smoothing_kernel,
                                                        smoothing_length,
                                                        density_calculator=SummationDensity(),
                                                        state_equation)
            fluid_system1.cache.density .= tank1.fluid.density
            v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system1))

            TrixiParticles.compute_pressure!(fluid_system1, v_fluid, semi)

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity)

            TrixiParticles.boundary_pressure_extrapolation!(Val(true), boundary_model,
                                                            boundary_system,
                                                            fluid_system1,
                                                            tank1.boundary.coordinates,
                                                            tank1.fluid.coordinates,
                                                            v_fluid,
                                                            nothing, # Not used
                                                            semi)

            for particle in TrixiParticles.eachparticle(boundary_system)
                TrixiParticles.compute_adami_density!(boundary_model, boundary_system,
                                                      tank1.boundary.coordinates, particle)
            end

            @test all(boundary_system.boundary_model.pressure .== 0.0)
            @test all(fluid_system1.pressure .== 0.0)
        end

        # Test whether the pressure is constant if the density of the state equation
        # and in the tank are not the same.
        # Then we test that the extrapolation yields some constant value.
        @testset "Constant Non-Zero Pressure" begin
            density = 260
            tank2 = RectangularTank(particle_spacing, (width, height), (width, height),
                                    density; n_layers, faces=(true, true, true, false))

            fluid_system2 = WeaklyCompressibleSPHSystem(tank2.fluid; smoothing_kernel,
                                                        smoothing_length,
                                                        density_calculator=SummationDensity(),
                                                        state_equation)

            fluid_system2.cache.density .= tank2.fluid.density
            v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system2))
            TrixiParticles.compute_pressure!(fluid_system2, v_fluid, semi)

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity)

            TrixiParticles.boundary_pressure_extrapolation!(Val(true), boundary_model,
                                                            boundary_system,
                                                            fluid_system2,
                                                            tank2.boundary.coordinates,
                                                            tank2.fluid.coordinates,
                                                            v_fluid,
                                                            nothing, # Not used
                                                            semi)

            for particle in TrixiParticles.eachparticle(boundary_system)
                TrixiParticles.compute_adami_density!(boundary_model, boundary_system,
                                                      tank2.boundary.coordinates, particle)
            end

            # Test that pressure of the fluid is indeed constant
            @test all(isapprox.(fluid_system2.pressure, fluid_system2.pressure[1]))
            # Test that boundary pressure equals fluid pressure
            @test all(isapprox.(boundary_system.boundary_model.pressure,
                                fluid_system2.pressure[1], atol=1.0e-12))
        end

        @testset "Interaction Matrix Filters Boundary Pressure" begin
            density_low = 260
            density_high = 320
            tank_low = RectangularTank(particle_spacing, (width, height), (width, height),
                                       density_low; n_layers,
                                       faces=(true, true, true, false))
            tank_high = RectangularTank(particle_spacing, (width, height), (width, height),
                                        density_high; n_layers,
                                        faces=(true, true, true, false))

            fluid_low = WeaklyCompressibleSPHSystem(tank_low.fluid;
                                                    smoothing_kernel,
                                                    smoothing_length,
                                                    density_calculator=SummationDensity(),
                                                    state_equation)
            fluid_high = WeaklyCompressibleSPHSystem(tank_high.fluid;
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     density_calculator=SummationDensity(),
                                                     state_equation)
            fluid_low.cache.density .= tank_low.fluid.density
            fluid_high.cache.density .= tank_high.fluid.density

            filtered_boundary_model = BoundaryModelDummyParticles(tank_low.boundary.density,
                                                                  tank_low.boundary.mass,
                                                                  AdamiPressureExtrapolation(),
                                                                  smoothing_kernel,
                                                                  smoothing_length;
                                                                  state_equation)
            filtered_boundary_system = WallBoundarySystem(tank_low.boundary,
                                                          filtered_boundary_model)

            included_boundary_model = BoundaryModelDummyParticles(tank_low.boundary.density,
                                                                  tank_low.boundary.mass,
                                                                  AdamiPressureExtrapolation(),
                                                                  smoothing_kernel,
                                                                  smoothing_length;
                                                                  state_equation)
            included_boundary_system = WallBoundarySystem(tank_low.boundary,
                                                          included_boundary_model)

            function update_boundary_pressure!(boundary_system, fluid_low, fluid_high,
                                               interaction_matrix)
                semi_local = Semidiscretization(fluid_low, fluid_high, boundary_system;
                                                neighborhood_search=nothing,
                                                interaction_matrix)

                v_ode = zeros(sum(TrixiParticles.v_nvariables(system) *
                                  TrixiParticles.n_integrated_particles(system)
                                  for system in semi_local.systems))
                u_ode = zeros(sum(TrixiParticles.u_nvariables(system) *
                                  TrixiParticles.n_integrated_particles(system)
                                  for system in semi_local.systems))

                TrixiParticles.foreach_system_wrapped(semi_local, v_ode,
                                                      u_ode) do system, v, u
                    TrixiParticles.write_v0!(v, system)
                    TrixiParticles.write_u0!(u, system)
                end

                for fluid_system in (fluid_low, fluid_high)
                    v_fluid_system = TrixiParticles.wrap_v(v_ode, fluid_system,
                                                           semi_local)
                    TrixiParticles.compute_pressure!(fluid_system, v_fluid_system,
                                                     semi_local)
                end

                v_boundary = TrixiParticles.wrap_v(v_ode, boundary_system, semi_local)
                u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi_local)
                boundary_model_local = boundary_system.boundary_model
                TrixiParticles.compute_pressure!(boundary_model_local,
                                                 boundary_model_local.density_calculator,
                                                 boundary_system, v_boundary, u_boundary,
                                                 v_ode, u_ode, semi_local)

                return copy(boundary_model_local.pressure), semi_local
            end

            all_enabled_pressure,
            _ = update_boundary_pressure!(included_boundary_system,
                                          fluid_low, fluid_high,
                                          trues(3, 3))

            filtered_matrix = trues(3, 3)
            filtered_matrix[3, 2] = false # wall skips pressure extrapolation from high fluid
            filtered_pressure,
            semi_filtered = update_boundary_pressure!(filtered_boundary_system,
                                                      fluid_low, fluid_high,
                                                      filtered_matrix)

            @test !TrixiParticles.has_system_interaction(semi_filtered.systems[3],
                                                         semi_filtered.systems[2],
                                                         semi_filtered)
            @test !all(isapprox.(all_enabled_pressure, fluid_low.pressure[1],
                                 atol=1.0e-12))
            @test all(isapprox.(filtered_pressure, fluid_low.pressure[1], atol=1.0e-12))
        end

        @testset "Interaction Matrix Filters Boundary Density" begin
            density_low = 260
            density_high = 320
            tank_low = RectangularTank(particle_spacing, (width, height), (width, height),
                                       density_low; n_layers,
                                       faces=(true, true, true, false))
            tank_high = RectangularTank(particle_spacing, (width, height), (width, height),
                                        density_high; n_layers,
                                        faces=(true, true, true, false))

            fluid_low = WeaklyCompressibleSPHSystem(tank_low.fluid;
                                                    smoothing_kernel,
                                                    smoothing_length,
                                                    density_calculator=SummationDensity(),
                                                    state_equation)
            fluid_high = WeaklyCompressibleSPHSystem(tank_high.fluid;
                                                     smoothing_kernel,
                                                     smoothing_length,
                                                     density_calculator=SummationDensity(),
                                                     state_equation)

            function boundary_density_with_systems(boundary_system, systems,
                                                   interaction_matrix)
                semi_local = Semidiscretization(systems..., boundary_system;
                                                neighborhood_search=nothing,
                                                interaction_matrix)

                v_ode = zeros(sum(TrixiParticles.v_nvariables(system) *
                                  TrixiParticles.n_integrated_particles(system)
                                  for system in semi_local.systems))
                u_ode = zeros(sum(TrixiParticles.u_nvariables(system) *
                                  TrixiParticles.n_integrated_particles(system)
                                  for system in semi_local.systems))

                TrixiParticles.foreach_system_wrapped(semi_local, v_ode,
                                                      u_ode) do system, v, u
                    TrixiParticles.write_v0!(v, system)
                    TrixiParticles.write_u0!(u, system)
                end

                v_boundary = TrixiParticles.wrap_v(v_ode, boundary_system, semi_local)
                u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi_local)
                boundary_model_local = boundary_system.boundary_model
                TrixiParticles.compute_density!(boundary_model_local,
                                                boundary_model_local.density_calculator,
                                                boundary_system, v_boundary, u_boundary,
                                                v_ode, u_ode, semi_local)

                return copy(boundary_model_local.cache.density), semi_local
            end

            all_enabled_boundary_model = BoundaryModelDummyParticles(tank_low.boundary.density,
                                                                     tank_low.boundary.mass,
                                                                     SummationDensity(),
                                                                     smoothing_kernel,
                                                                     smoothing_length;
                                                                     state_equation)
            all_enabled_boundary = WallBoundarySystem(tank_low.boundary,
                                                      all_enabled_boundary_model)
            all_enabled_density,
            _ = boundary_density_with_systems(all_enabled_boundary,
                                              (fluid_low, fluid_high),
                                              trues(3, 3))

            filtered_boundary_model = BoundaryModelDummyParticles(tank_low.boundary.density,
                                                                  tank_low.boundary.mass,
                                                                  SummationDensity(),
                                                                  smoothing_kernel,
                                                                  smoothing_length;
                                                                  state_equation)
            filtered_boundary = WallBoundarySystem(tank_low.boundary,
                                                   filtered_boundary_model)
            filtered_matrix = trues(3, 3)
            filtered_matrix[3, 2] = false # wall skips density summation from high fluid
            filtered_density,
            semi_filtered = boundary_density_with_systems(filtered_boundary,
                                                          (fluid_low,
                                                           fluid_high),
                                                          filtered_matrix)

            reference_boundary_model = BoundaryModelDummyParticles(tank_low.boundary.density,
                                                                   tank_low.boundary.mass,
                                                                   SummationDensity(),
                                                                   smoothing_kernel,
                                                                   smoothing_length;
                                                                   state_equation)
            reference_boundary = WallBoundarySystem(tank_low.boundary,
                                                    reference_boundary_model)
            reference_density,
            _ = boundary_density_with_systems(reference_boundary,
                                              (fluid_low,), trues(2, 2))

            @test !TrixiParticles.has_system_interaction(semi_filtered.systems[3],
                                                         semi_filtered.systems[2],
                                                         semi_filtered)
            @test any(all_enabled_density .> filtered_density)
            @test all(isapprox.(filtered_density, reference_density))
        end

        # In this test, we initialize a fluid with a hydrostatic pressure gradient
        # and check that this gradient is extrapolated correctly.
        @testset "Hydrostatic Pressure Gradient" begin
            for flipped_condition in (Val(true), Val(false))
                tank3 = RectangularTank(particle_spacing, (width, height), (width, height),
                                        density; acceleration=[0.0, -9.81], state_equation,
                                        n_layers, faces=(true, true, true, false))

                fluid_system3 = WeaklyCompressibleSPHSystem(tank3.fluid;
                                                            smoothing_kernel,
                                                            smoothing_length,
                                                            density_calculator=SummationDensity(),
                                                            state_equation,
                                                            acceleration=[0.0, -9.81])
                fluid_system3.cache.density .= tank3.fluid.density
                v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system3))
                TrixiParticles.compute_pressure!(fluid_system3, v_fluid, semi)

                TrixiParticles.set_zero!(boundary_model.pressure)
                TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                            viscosity)

                TrixiParticles.boundary_pressure_extrapolation!(flipped_condition,
                                                                boundary_model,
                                                                boundary_system,
                                                                fluid_system3,
                                                                tank3.boundary.coordinates,
                                                                tank3.fluid.coordinates,
                                                                v_fluid,
                                                                nothing, # Not used
                                                                semi)

                for particle in TrixiParticles.eachparticle(boundary_system)
                    TrixiParticles.compute_adami_density!(boundary_model, boundary_system,
                                                          tank3.boundary.coordinates,
                                                          particle)
                end

                width_reference = particle_spacing * (n_particles + 2 * n_layers)
                height_reference = particle_spacing * (n_particles + n_layers)

                # Define another tank without a boundary, where the fluid has the same size
                # as fluid plus boundary in the other tank.
                # The pressure gradient of this fluid should be the same as the extrapolated pressure
                # of the boundary in the first tank.
                tank_reference = RectangularTank(particle_spacing,
                                                 (width_reference, height_reference),
                                                 (width_reference, height_reference),
                                                 density; acceleration=[0.0, -9.81],
                                                 state_equation, n_layers=0,
                                                 faces=(true, true, true, false))

                # Because it is a pain to deal with the linear indices of the pressure arrays,
                # we convert the matrices to Cartesian indices based on the coordinates.
                function set_pressure!(pressure, coordinates, offset, system,
                                       system_pressure)
                    for particle in TrixiParticles.eachparticle(system)
                        # Coordinates as integer indices
                        coords = coordinates[:, particle] ./ particle_spacing
                        # Move bottom left corner to (1, 1)
                        coords .+= offset
                        # Round to integer index
                        index = round.(Int, coords)
                        pressure[index...] = system_pressure[particle]
                    end
                end

                # Set up the combined pressure matrix to store the pressure values of fluid
                # and boundary together.
                pressure = zeros(n_particles + 2 * n_layers, n_particles + n_layers)

                # The fluid starts at -0.5 * particle_spacing from (0, 0),
                # so the boundary starts at -(n_layers + 0.5) * particle_spacing
                set_pressure!(pressure, boundary_system.coordinates, n_layers + 0.5,
                              boundary_system, boundary_system.boundary_model.pressure)

                # The fluid starts at -0.5 * particle_spacing from (0, 0),
                # so the boundary starts at -(n_layers + 0.5) * particle_spacing
                set_pressure!(pressure, tank3.fluid.coordinates, n_layers + 0.5,
                              fluid_system3, fluid_system3.pressure)
                pressure_reference = similar(pressure)

                # The fluid starts at -0.5 * particle_spacing from (0, 0)
                set_pressure!(pressure_reference, tank_reference.fluid.coordinates, 0.5,
                              tank_reference.fluid, tank_reference.fluid.pressure)

                @test all(isapprox.(pressure, pressure_reference, atol=4.0))
            end
        end
    end

    @testset verbose=true "Pressure Mirroring: No-Slip Wall Velocity" begin
        # `PressureMirroring` does not extrapolate the boundary pressure, but the no-slip
        # condition by Adami et al. (2012) requires the same wall velocity
        # `v_w = 2 * v_boundary - ṽ_fluid` as with `AdamiPressureExtrapolation`.
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2
        fluid_density = 1000.0
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=fluid_density, exponent=1)
        viscosity = ViscosityAdami(nu=0.1)

        # Fluid particles with different velocities above a row of boundary particles.
        # The last boundary particle has no fluid neighbors.
        fluid_ic = InitialCondition(; coordinates=[-1.0 0.0 -1.0 0.0; 1.0 1.0 2.0 2.0],
                                    velocity=[1.0 0.0 -1.0 0.5; 0.0 1.0 0.5 -0.5],
                                    mass=fill(fluid_density, 4),
                                    density=fill(fluid_density, 4), particle_spacing=1.0)
        n_boundary = 5
        boundary_coordinates = [collect(-1.0:3.0)'; zeros(1, n_boundary)]

        function create_systems(boundary_type, density_calculator, boundary_velocity)
            fluid_system = WeaklyCompressibleSPHSystem(fluid_ic; smoothing_kernel,
                                                       smoothing_length, state_equation,
                                                       density_calculator=ContinuityDensity(),
                                                       viscosity)

            velocity = repeat(collect(boundary_velocity), 1, n_boundary)
            boundary_ic = InitialCondition(; coordinates=boundary_coordinates,
                                           velocity, mass=fill(fluid_density, n_boundary),
                                           density=fill(fluid_density, n_boundary),
                                           particle_spacing=1.0)
            boundary_model = BoundaryModelDummyParticles(boundary_ic.density,
                                                         boundary_ic.mass,
                                                         density_calculator,
                                                         smoothing_kernel,
                                                         smoothing_length;
                                                         state_equation, viscosity)

            boundary_system = if boundary_type === WallBoundarySystem
                WallBoundarySystem(boundary_ic, boundary_model)
            elseif boundary_type === TotalLagrangianSPHSystem
                TotalLagrangianSPHSystem(boundary_ic; smoothing_kernel, smoothing_length,
                                         young_modulus=1e5, poisson_ratio=0.3,
                                         boundary_model)
            else
                RigidBodySystem(boundary_ic; boundary_model)
            end

            return fluid_system, boundary_system
        end

        # Run the regular update step, which computes the wall velocity
        function initialize(fluid_system, boundary_system)
            semi = if boundary_system isa TotalLagrangianSPHSystem
                @test_logs (:info,
                            r"^To create the self-interaction neighborhood search") Semidiscretization(fluid_system,
                                                                                                       boundary_system;
                                                                                                       parallelization_backend=SerialBackend())
            else
                Semidiscretization(fluid_system, boundary_system;
                                   parallelization_backend=SerialBackend())
            end
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)

            return ode.p.semi, v_ode, u_ode
        end

        # Walls are at rest, structures move
        boundary_configurations = ((WallBoundarySystem, (0.0, 0.0)),
                                   (TotalLagrangianSPHSystem, (0.2, -0.1)),
                                   (RigidBodySystem, (0.2, -0.1)))

        @testset "`$boundary_type`" for (boundary_type,
                                         boundary_velocity) in boundary_configurations

            semi, _,
            _ = initialize(create_systems(boundary_type,
                                          AdamiPressureExtrapolation(),
                                          boundary_velocity)...)
            wall_velocity_adami = semi.systems[2].boundary_model.cache.wall_velocity

            # The wall velocity is only computed for boundary particles with fluid neighbors
            for particle in 1:(n_boundary - 1)
                @test !iszero(wall_velocity_adami[:, particle])
            end
            @test iszero(wall_velocity_adami[:, n_boundary])

            semi, _,
            _ = initialize(create_systems(boundary_type, PressureMirroring(),
                                          boundary_velocity)...)
            wall_velocity = semi.systems[2].boundary_model.cache.wall_velocity

            @test wall_velocity ≈ wall_velocity_adami
        end

        # A structure moving through fluid at rest must drag the fluid along.
        # The fluid pressure is zero, so both models have the reference density and zero
        # pressure. Therefore, the pair force is purely viscous and must be the same.
        @testset "Viscous Force of a Moving `$structure_type`" for structure_type in
                                                                   (TotalLagrangianSPHSystem,
                                                                    RigidBodySystem)
            # A single fluid particle at rest and a single structure particle at `(1.5h, 0)`
            fluid_ic_ = InitialCondition(; coordinates=zeros(2, 1), velocity=zeros(2, 1),
                                         mass=[fluid_density], density=[fluid_density],
                                         particle_spacing=1.0)
            structure_velocity = SVector(1.0, 0.5)
            structure_ic = InitialCondition(;
                                            coordinates=reshape([1.5 * smoothing_length,
                                                                    0.0], 2, 1),
                                            velocity=reshape(collect(structure_velocity),
                                                             2, 1),
                                            mass=[2 * fluid_density],
                                            density=[2 * fluid_density],
                                            particle_spacing=1.0)

            function viscous_force(density_calculator)
                fluid_system = WeaklyCompressibleSPHSystem(fluid_ic_; smoothing_kernel,
                                                           smoothing_length, state_equation,
                                                           density_calculator=ContinuityDensity(),
                                                           viscosity)
                boundary_model = BoundaryModelDummyParticles([fluid_density],
                                                             [fluid_density],
                                                             density_calculator,
                                                             smoothing_kernel,
                                                             smoothing_length;
                                                             state_equation, viscosity)
                structure_system = if structure_type === TotalLagrangianSPHSystem
                    TotalLagrangianSPHSystem(structure_ic; smoothing_kernel,
                                             smoothing_length, young_modulus=1e5,
                                             poisson_ratio=0.3, boundary_model)
                else
                    RigidBodySystem(structure_ic; boundary_model)
                end

                semi, v_ode, u_ode = initialize(fluid_system, structure_system)
                fluid, structure = semi.systems
                v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
                u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
                v_structure = TrixiParticles.wrap_v(v_ode, structure, semi)
                u_structure = TrixiParticles.wrap_u(u_ode, structure, semi)

                dv_fluid = zero(v_fluid)
                TrixiParticles.interact!(dv_fluid, v_fluid, u_fluid, v_structure,
                                         u_structure, fluid, structure, semi)

                return fluid.mass[1] * dv_fluid[1:2, 1]
            end

            force_adami = viscous_force(AdamiPressureExtrapolation())
            @test dot(force_adami, structure_velocity) > 0
            @test viscous_force(PressureMirroring()) ≈ force_adami
        end
    end

    @testset verbose=true "Pressure Mirroring with EDAC" begin
        # Two fluid particles with different pressures next to a wall particle.
        # With pressure mirroring, fluid particle `a` sees the wall pressure `p_a`.
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.0
        fluid_density = 1000.0
        pressures = [500.0, 100.0]

        # Both fluid particles are neighbors of each other and of the wall particle
        fluid_ic = InitialCondition(; coordinates=[0.0 0.5; 0.0 0.0], velocity=zeros(2, 2),
                                    mass=fill(fluid_density, 2),
                                    density=fill(fluid_density, 2), pressure=pressures,
                                    particle_spacing=1.0)
        distances = (1.5, 1.0)
        wall_ic = InitialCondition(; coordinates=reshape([1.5, 0.0], 2, 1),
                                   mass=[fluid_density], density=[fluid_density],
                                   particle_spacing=1.0)

        # Independent cubic-spline gradient ∇_a W_ab for `b` to the right of `a`
        function grad_kernel(distance)
            derivative = 10 / (7pi) *
                         (distance < 1 ? -3distance + 2.25distance^2 :
                          -0.75 * (2 - distance)^2)
            return SVector(-derivative, 0.0)
        end

        @testset "average_pressure_reduction=$average_pressure_reduction" for average_pressure_reduction in
                                                                              (false,
                                                                               true)
            fluid_system = EntropicallyDampedSPHSystem(fluid_ic; smoothing_kernel,
                                                       smoothing_length, sound_speed=10.0,
                                                       density_calculator=ContinuityDensity(),
                                                       average_pressure_reduction)
            boundary_model = BoundaryModelDummyParticles(wall_ic.density, wall_ic.mass,
                                                         PressureMirroring(),
                                                         smoothing_kernel, smoothing_length)
            wall_system = WallBoundarySystem(wall_ic, boundary_model)
            semi = Semidiscretization(fluid_system, wall_system;
                                      parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.01))
            v_ode, u_ode = ode.u0.x
            TrixiParticles.update_systems_and_nhs(v_ode, u_ode, ode.p.semi, 0.0)

            semi = ode.p.semi
            fluid, wall = semi.systems

            if average_pressure_reduction
                # Each particle averages over itself, the other fluid particle and the wall
                # particle, which has the mirrored pressure of the particle itself.
                @test fluid.cache.pressure_average ≈
                      [2 * pressures[1] + pressures[2], pressures[1] + 2 * pressures[2]] /
                      3
            end

            v_fluid = TrixiParticles.wrap_v(v_ode, fluid, semi)
            u_fluid = TrixiParticles.wrap_u(u_ode, fluid, semi)
            v_wall = TrixiParticles.wrap_v(v_ode, wall, semi)
            u_wall = TrixiParticles.wrap_u(u_ode, wall, semi)
            dv = zero(v_fluid)
            TrixiParticles.interact!(dv, v_fluid, u_fluid, v_wall, u_wall, fluid, wall,
                                     semi)

            for particle in 1:2
                # `inter_particle_averaged_pressure` with the mirrored wall pressure
                # `p_b = p_a` and the volumes `m_a / rho_a = m_b / rho_b = 1`.
                # Without average pressure reduction, `average_pressure` returns zero.
                reduced_pressure = pressures[particle] -
                                   TrixiParticles.average_pressure(fluid, particle)
                expected = -2 / fluid_density * reduced_pressure *
                           grad_kernel(distances[particle])

                @test dv[1:2, particle] ≈ expected
            end

            # The divergence term of the pressure evolution is zero for particles at rest,
            # and the mirrored wall pressure makes the pressure diffusion term zero.
            @test iszero(dv[3, :])
        end
    end
end

include("rhs.jl")
