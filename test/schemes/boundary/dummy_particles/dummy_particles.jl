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

    @testset "Viscosity Adami/Bernoulli: Wall Velocity" begin
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
        pressure_extrapolations = [
            AdamiPressureExtrapolation(),
            BernoulliPressureExtrapolation()
        ]

        for pressure_extrapolation in pressure_extrapolations
            @testset "Pressure Extrapolation: $(typeof(pressure_extrapolation))" begin
                # Create boundary and fluid systems
                boundary_model = BoundaryModelDummyParticles(boundary.density,
                                                             boundary.mass,
                                                             state_equation=state_equation,
                                                             pressure_extrapolation,
                                                             smoothing_kernel,
                                                             smoothing_length,
                                                             viscosity=viscosity)
                boundary_system = BoundarySPHSystem(boundary, boundary_model)
                fluid_system = WeaklyCompressibleSPHSystem(fluid,
                                                           SummationDensity(),
                                                           state_equation,
                                                           smoothing_kernel,
                                                           smoothing_length)

                neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=1.0,
                                                                                  eachpoint=TrixiParticles.eachparticle(fluid_system))

                velocities = [
                    [0.0; -1.0],
                    [1.0; 1.0],
                    [-1.0; 0.0],
                    [0.7; 0.2],
                    [0.3; 0.8]
                ]

                @testset "Wall Velocity $v_fluid" for v_fluid in velocities
                    viscosity = boundary_system.boundary_model.viscosity
                    volume = boundary_system.boundary_model.cache.volume

                    # Reset cache and perform pressure extrapolation
                    TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                                boundary_system.boundary_model.viscosity)
                    TrixiParticles.boundary_pressure_extrapolation!(boundary_model,
                                                                    boundary_system,
                                                                    fluid_system,
                                                                    boundary.coordinates,
                                                                    fluid.coordinates,
                                                                    v_fluid,
                                                                    v_fluid .*
                                                                    ones(size(fluid.coordinates)),
                                                                    neighborhood_search)

                    # Compute wall velocities
                    for particle in TrixiParticles.eachparticle(boundary_system)
                        if volume[particle] > eps()
                            TrixiParticles.compute_wall_velocity!(viscosity,
                                                                  boundary_system,
                                                                  boundary.coordinates,
                                                                  particle)
                        end
                    end

                    # Expected wall velocities
                    v_wall = zeros(size(boundary.coordinates))
                    v_wall[:, 1:particles_in_compact_support] .= -v_fluid

                    @test isapprox(boundary_system.boundary_model.cache.wall_velocity,
                                   v_wall)
                end

                scale_factors = [1.0, 0.5, 0.7, 1.8, 67.5]

                # For a constant velocity profile (each fluid particle has the same velocity),
                # the wall velocity is `v_wall = -v_fluid` (see eq. 22 in Adami_2012).
                # With a staggered velocity profile, we can test the smoothed velocity field
                # for a variable velocity profile.
                @testset "Wall Velocity Staggered: Factor $scale" for scale in scale_factors
                    viscosity = boundary_system.boundary_model.viscosity
                    volume = boundary_system.boundary_model.cache.volume

                    # Create a staggered velocity profile
                    v_fluid = zeros(size(fluid.coordinates))
                    for i in TrixiParticles.eachparticle(fluid_system)
                        if mod(i, 2) == 1
                            v_fluid[:, i] .= scale
                        end
                    end

                    # Reset cache and perform pressure extrapolation
                    TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                                boundary_system.boundary_model.viscosity)
                    TrixiParticles.boundary_pressure_extrapolation!(boundary_model,
                                                                    boundary_system,
                                                                    fluid_system,
                                                                    boundary.coordinates,
                                                                    fluid.coordinates,
                                                                    v_fluid,
                                                                    v_fluid,
                                                                    neighborhood_search)

                    # Compute wall velocities
                    for particle in TrixiParticles.eachparticle(boundary_system)
                        if volume[particle] > eps()
                            TrixiParticles.compute_wall_velocity!(viscosity,
                                                                  boundary_system,
                                                                  boundary.coordinates,
                                                                  particle)
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
        n_particles = 2
        n_layers = 2
        width = particle_spacing * n_particles
        height = particle_spacing * n_particles
        density = 257

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 3 * particle_spacing
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

        boundary_system = BoundarySPHSystem(tank1.boundary, boundary_model)
        viscosity = boundary_system.boundary_model.viscosity

        # In this testset, we verify that pressures in a static tank are extrapolated correctly
        # Use constant density equal to the reference density of the state equation,
        # so that the pressure is constant zero. Then we test that the extrapolation also yields zero.
        @testset "Constant Zero Pressure" begin
            fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                        state_equation,
                                                        smoothing_kernel, smoothing_length)
            fluid_system1.cache.density .= tank1.fluid.density
            v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system1))

            TrixiParticles.compute_pressure!(fluid_system1, v_fluid)

            neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=TrixiParticles.compact_support(smoothing_kernel,
                                                                                                                           smoothing_length),
                                                                              eachpoint=TrixiParticles.eachparticle(fluid_system1))

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity)

            TrixiParticles.boundary_pressure_extrapolation!(boundary_model, boundary_system,
                                                            fluid_system1,
                                                            tank1.boundary.coordinates,
                                                            tank1.fluid.coordinates,
                                                            v_fluid,
                                                            v_fluid,
                                                            neighborhood_search)

            for particle in TrixiParticles.eachparticle(boundary_system)
                TrixiParticles.compute_adami_density!(boundary_model, boundary_system,
                                                      tank1.boundary.coordinates, particle)
            end

            @test all(boundary_system.boundary_model.pressure .== 0.0) &&
                  all(fluid_system1.pressure .== 0.0)
        end

        # Test whether the pressure is constant if the density of the state equation
        # and in the tank are not the same. Then we test that the extrapolation yields some constant value.
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
            TrixiParticles.compute_pressure!(fluid_system2, v_fluid)

            neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=TrixiParticles.compact_support(smoothing_kernel,
                                                                                                                           smoothing_length),
                                                                              eachpoint=TrixiParticles.eachparticle(fluid_system2))

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity)

            TrixiParticles.boundary_pressure_extrapolation!(boundary_model, boundary_system,
                                                            fluid_system2,
                                                            tank2.boundary.coordinates,
                                                            tank2.fluid.coordinates,
                                                            v_fluid,
                                                            v_fluid,
                                                            neighborhood_search)

            for particle in TrixiParticles.eachparticle(boundary_system)
                TrixiParticles.compute_adami_density!(boundary_model, boundary_system,
                                                      tank2.boundary.coordinates, particle)
            end

            @test all(isapprox.(boundary_system.boundary_model.pressure,
                                boundary_system.boundary_model.pressure[1])) &&
                  all(isapprox.(fluid_system2.pressure, fluid_system2.pressure[1]))

            # add smallest a_tol for smallest perturbation to trigger the test

        end

        @testset "Hydrostatic Pressure Gradient" begin
            tank3 = RectangularTank(particle_spacing, (width, height), (width, height),
                                    density, acceleration=[0.0, -9.81],
                                    state_equation=state_equation, n_layers=n_layers,
                                    faces=(true, true, true, false))

            fluid_system3 = WeaklyCompressibleSPHSystem(tank3.fluid, SummationDensity(),
                                                        state_equation,
                                                        smoothing_kernel, smoothing_length)
            fluid_system3.cache.density .= tank3.fluid.density
            v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system3))
            TrixiParticles.compute_pressure!(fluid_system3, v_fluid)

            neighborhood_search = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius=TrixiParticles.compact_support(smoothing_kernel,
                                                                                                                           smoothing_length),
                                                                              eachpoint=TrixiParticles.eachparticle(fluid_system3))

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity)

            TrixiParticles.boundary_pressure_extrapolation!(boundary_model, boundary_system,
                                                            fluid_system3,
                                                            tank3.boundary.coordinates,
                                                            tank3.fluid.coordinates,
                                                            v_fluid,
                                                            v_fluid,
                                                            neighborhood_search)

            for particle in TrixiParticles.eachparticle(boundary_system)
                TrixiParticles.compute_adami_density!(boundary_model, boundary_system,
                                                      tank3.boundary.coordinates, particle)
            end

            width_ref = particle_spacing * (n_particles + 2 * n_layers)
            height_ref = particle_spacing * (n_particles + n_layers)

            tank_ref = RectangularTank(particle_spacing, (width_ref, height_ref),
                                       (width_ref, height_ref),
                                       density, acceleration=[0.0, -9.81],
                                       state_equation=state_equation, n_layers=0,
                                       faces=(true, true, true, false))

            fluid_system_ref = WeaklyCompressibleSPHSystem(tank_ref.fluid,
                                                           SummationDensity(),
                                                           state_equation,
                                                           smoothing_kernel,
                                                           smoothing_length)

            fluid_system_ref.cache.density .= tank_ref.fluid.density
            v_fluid_ref = zeros(2, TrixiParticles.nparticles(fluid_system_ref))
            TrixiParticles.compute_pressure!(fluid_system_ref, v_fluid_ref)

            #=
            Because it is a pain to deal with the indices of the pressure arrays,
            we convert the flattened matrices back to their original shape.
            We then transform the matrices to have the same orientation, such that
            they have shape (n_rows, n_cols) and look like this (B = boundary, F = fluid):

            B B F F F B B
            B B F F F B B
            B B F F F B B
            B B B B B B B
            B B B B B B B

            We then use regular indexing on matrices in Julia,
            i.e. the top-left entry has coordinates (1, 1).

            Variables:
            - `coor`: Coordinates of the boundary system.
            - `n_cols`: Number of columns in the pressure matrix, calculated as the number of particles plus twice the number of layers.
            - `n_rows`: Number of rows in the pressure matrix, calculated as the number of particles plus the number of layers.

            Functions:
            - `coordinates_to_indices(coor)`: Maps physical coordinates to corresponding indices in an array.
              - Returns: Array of indices corresponding to the input coordinates.

            Main Operations:
            - `boundary_x_idx, boundary_y_idx`: Arrays of indices for the x and y coordinates of the boundary system.
            - `press`: Zero-initialized matrix to store pressure values.
            - Loop through the boundary system's pressure values and assign them to the `press` matrix using the boundary indices.
            - Reverse the `press` matrix along the first dimension.
            - `press_fluid`: Matrix containing the pressure of fluid3, reshaped and transposed.
            - Reverse the `press_fluid` matrix along the first dimension.
            - Assign the `press_fluid` matrix to the corresponding section of the `press` matrix.
            - `press_ref`: Matrix containing the reference fluid pressure, reshaped and transposed.
            - Reverse the `press_ref` matrix along the first dimension.
            =#

            coor = boundary_system.coordinates
            n_cols = n_particles + 2 * n_layers
            n_rows = n_particles + n_layers

            # Function that maps physical coordinates to the corresponding
            # indices in an array.
            function coordinates_to_indices(coor)
                unique_vals = sort(unique(coor))  # Sorted unique values
                index_map = Dict(val => idx for (idx, val) in enumerate(unique_vals))
                index_array = [index_map[x] for x in coor]
                return index_array
            end

            # Compute the indices for the x and y coordinates of the boundary system.
            boundary_x_idx, boundary_y_idx = [coordinates_to_indices(axis)
                                              for axis in eachrow(boundary_system.coordinates)]

            # Set up the pressure matrix to store the pressure values.
            press = zeros(n_rows, n_cols)

            # Assign the boundary pressure values to the pressure matrix using the boundary indices.
            for i in 1:length(boundary_system.boundary_model.pressure)
                press[boundary_y_idx[i], boundary_x_idx[i]] = boundary_system.boundary_model.pressure[i]
            end
            # To keep the orientation consistent, reverse the pressure matrix along the first dimension.
            press = reverse(press, dims=1)

            # Extract the fluid pressure matrix. while keeping the orientation consistent.
            press_fluid = transpose(reshape(fluid_system3.pressure,
                                            (n_particles, n_particles))) # A matrix containing the pressure of fluid3
            press_fluid = reverse(press_fluid, dims=1)

            # Assign the fluid pressure values to the corresponding section of the pressure matrix.
            press[begin:n_particles, (n_layers + 1):(n_particles + n_layers)] .= press_fluid

            # Extract the reference fluid pressure matrix. while keeping the orientation consistent.
            press_ref = transpose(reshape(fluid_system_ref.pressure, (n_cols, n_rows)))
            press_ref = reverse(press_ref, dims=1)

            # TrixiParticles.@autoinfiltrate
        end
    end
end

include("rhs.jl")
