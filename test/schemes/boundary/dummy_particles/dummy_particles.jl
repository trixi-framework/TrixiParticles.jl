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
        boundary_model_adami = BoundaryModelDummyParticles(boundary.density,
                                                           boundary.mass,
                                                           state_equation=state_equation,
                                                           AdamiPressureExtrapolation(),
                                                           smoothing_kernel,
                                                           smoothing_length,
                                                           viscosity=viscosity)
        boundary_model_bernoulli = BoundaryModelDummyParticles(boundary.density,
                                                               boundary.mass,
                                                               state_equation=state_equation,
                                                               BernoulliPressureExtrapolation(),
                                                               smoothing_kernel,
                                                               smoothing_length,
                                                               viscosity=viscosity)

        boundary_systems = [
            WallBoundarySystem(boundary, boundary_model_adami),
            WallBoundarySystem(boundary, boundary_model_bernoulli),
            TotalLagrangianSPHSystem(boundary, smoothing_kernel,
                                     smoothing_length, 1e6, 0.3;
                                     boundary_model=boundary_model_adami),
            TotalLagrangianSPHSystem(boundary, smoothing_kernel,
                                     smoothing_length, 1e6, 0.3;
                                     boundary_model=boundary_model_bernoulli)
        ]

        # Create fluid system
        fluid_system = WeaklyCompressibleSPHSystem(fluid,
                                                   SummationDensity(),
                                                   state_equation,
                                                   smoothing_kernel,
                                                   smoothing_length)

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

        tank1 = RectangularTank(particle_spacing, (width, height), (width, height),
                                density, n_layers=n_layers,
                                faces=(true, true, true, false))

        boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                     tank1.boundary.mass,
                                                     state_equation=state_equation,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)

        boundary_system = WallBoundarySystem(tank1.boundary, boundary_model)
        viscosity = boundary_system.boundary_model.viscosity

        semi = DummySemidiscretization()

        # In this testset, we verify that pressures in a static tank are extrapolated correctly.
        # Use constant density equal to the reference density of the state equation,
        # so that the pressure is constant zero. Then we test that the extrapolation also yields zero.
        @testset "Constant Zero Pressure" begin
            fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                        state_equation,
                                                        smoothing_kernel, smoothing_length)
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
                                    density, n_layers=n_layers,
                                    faces=(true, true, true, false))

            fluid_system2 = WeaklyCompressibleSPHSystem(tank2.fluid, SummationDensity(),
                                                        state_equation,
                                                        smoothing_kernel, smoothing_length)

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

        # In this test, we initialize a fluid with a hydrostatic pressure gradient
        # and check that this gradient is extrapolated correctly.
        @testset "Hydrostatic Pressure Gradient" begin
            for flipped_condition in (Val(true), Val(false))
                tank3 = RectangularTank(particle_spacing, (width, height), (width, height),
                                        density, acceleration=[0.0, -9.81],
                                        state_equation=state_equation, n_layers=n_layers,
                                        faces=(true, true, true, false))

                fluid_system3 = WeaklyCompressibleSPHSystem(tank3.fluid, SummationDensity(),
                                                            state_equation,
                                                            smoothing_kernel,
                                                            smoothing_length,
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
                                                 density, acceleration=[0.0, -9.81],
                                                 state_equation=state_equation, n_layers=0,
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
end

include("rhs.jl")
