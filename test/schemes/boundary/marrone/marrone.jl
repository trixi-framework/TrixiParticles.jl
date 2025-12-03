@testset verbose=true "Dummy Particles with `MarronePressureExtrapolation`" begin
    @testset "Compute Boundary Normal Vectors" begin
        particle_spacing = 1.0
        n_particles = 2
        n_layers = 1
        width = particle_spacing * n_particles
        height = particle_spacing * n_particles
        density = 257

        tank = RectangularTank(particle_spacing, (width, height), (width, height),
                               density, n_layers=n_layers,
                               faces=(true, true, true, false), normal=true)

        (; normals) = tank.boundary
        normals_reference = [[-0.5 -0.5 0.5 0.5 0.0 0.0 -0.5 0.5]
                             [0.0 0.0 0.0 0.0 -0.5 -0.5 -0.5 -0.5]]

        @test normals == normals_reference
    end

    @testset "`MarroneMLSKernel`" begin
        particle_spacing = 1.0
        n_particles = 2
        n_layers = 1
        width = particle_spacing * n_particles
        height = particle_spacing * n_particles
        density = 257

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = particle_spacing * 1.1
        state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                           exponent=7)

        tank1 = RectangularTank(particle_spacing, (width, height), (width, height),
                                density, n_layers=n_layers,
                                faces=(true, true, true, false), normal=true)
        n_boundary_particles = size(tank1.boundary.coordinates, 2)
        n_fluid_particles = size(tank1.fluid.coordinates, 2)

        mls_kernel = MarroneMLSKernel(smoothing_kernel, n_boundary_particles,
                                      n_fluid_particles)
        boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                     tank1.boundary.mass,
                                                     state_equation=state_equation,
                                                     MarronePressureExtrapolation(),
                                                     mls_kernel, smoothing_length)
        boundary_system = WallBoundarySystem(tank1.boundary, boundary_model)
        viscosity = boundary_system.boundary_model.viscosity
        semi = DummySemidiscretization()
        fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                    state_equation,
                                                    smoothing_kernel, smoothing_length)
        fluid_system1.cache.density .= tank1.fluid.density
        v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system1))

        TrixiParticles.compute_pressure!(fluid_system1, v_fluid, semi)

        TrixiParticles.set_zero!(boundary_model.pressure)
        TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                    viscosity)

        boundary_coords = tank1.boundary.coordinates
        fluid_coords = tank1.fluid.coordinates

        expected_basis = zeros(n_boundary_particles, n_fluid_particles, 3)
        @testset "Compute Marrone Basis" begin
            (; basis) = mls_kernel
            expected_basis[:, 1,
                           :] = [1.0 1.0 0.0
                                 1.0 1.0 -1.0
                                 1.0 -2.0 0.0
                                 0.0 0.0 0.0
                                 1.0 0.0 1.0
                                 1.0 -1.0 1.0
                                 1.0 1.0 1.0
                                 0.0 0.0 0.0]
            expected_basis[:, 2,
                           :] = [1.0 2.0 0.0
                                 0.0 0.0 0.0
                                 1.0 -1.0 0.0
                                 1.0 -1.0 -1.0
                                 1.0 1.0 1.0
                                 1.0 0.0 1.0
                                 0.0 0.0 0.0
                                 1.0 -1.0 1.0]
            expected_basis[:, 3,
                           :] = [1.0 1.0 1.0
                                 1.0 1.0 0.0
                                 0.0 0.0 0.0
                                 1.0 -2.0 0.0
                                 1.0 0.0 2.0
                                 0.0 0.0 0.0
                                 0.0 0.0 0.0
                                 0.0 0.0 0.0]
            expected_basis[:, 4,
                           :] = [0.0 0.0 0.0
                                 1.0 2.0 0.0
                                 1.0 -1.0 1.0
                                 1.0 -1.0 0.0
                                 0.0 0.0 0.0
                                 1.0 0.0 2.0
                                 0.0 0.0 0.0
                                 0.0 0.0 0.0]

            TrixiParticles.compute_basis_marrone(mls_kernel, boundary_system, fluid_system1,
                                                 boundary_coords, fluid_coords, semi)
            @test basis == expected_basis
        end

        expected_momentum = zeros(n_boundary_particles, 3, 3)
        @testset "Compute Marrone Momentum" begin
            (; inner_kernel, momentum) = mls_kernel

            # The momentum is computed as M_i = ∑ⱼ b_ij ⊗ b_ij' ⋅ kernel_weight_j ⋅ volume_j 
            # The volume of all fluid particles is 1.0, so we leave this factor out
            # We need only consider the neighbors in the compact support of the inner kernel 
            for (i, fluid_indices) in Dict(1 => [1, 2, 3], 2 => [1, 3, 4], 5 => [1, 2, 3])
                for j in fluid_indices
                    diff = boundary_coords[:, i] - fluid_coords[:, j]
                    distance = sqrt(dot(diff, diff))
                    expected_momentum[i, :,
                                      :] += expected_basis[i, j, :] .*
                                            expected_basis[i, j, :]' *
                                            TrixiParticles.kernel(inner_kernel, distance,
                                                                  smoothing_length)
                end

                if abs(det(expected_momentum[i, :, :])) < 1.0f-9
                    expected_momentum[i, :,
                                      :] = Matrix{Float64}(I, 3, 3)
                else
                    expected_momentum[i, :, :] = inv(expected_momentum[i, :, :])
                end
            end

            TrixiParticles.compute_momentum_marrone(mls_kernel, boundary_system,
                                                    fluid_system1, boundary_coords,
                                                    fluid_coords, v_fluid, semi,
                                                    smoothing_length)

            @test expected_momentum[1, :, :] == momentum[1, :, :]
            @test expected_momentum[2, :, :] == momentum[2, :, :]
            @test expected_momentum[5, :, :] == momentum[5, :, :]
            @test Matrix{Float64}(I, 3, 3) == momentum[7, :, :]
        end

        @testset "Pressure Extrapolation" begin
            particle_spacing = 1.0
            n_particles = 10
            n_layers = 4
            width = particle_spacing * n_particles
            height = particle_spacing * n_particles
            density = 257

            smoothing_kernel = SchoenbergCubicSplineKernel{2}()
            smoothing_length = 3 * particle_spacing
            state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                               exponent=7)

            tank1 = RectangularTank(particle_spacing, (width, height), (width, height),
                                    density, n_layers=n_layers,
                                    faces=(true, true, true, false), normal=true)
            n_boundary_particles = size(tank1.boundary.coordinates, 2)
            n_fluid_particles = size(tank1.fluid.coordinates, 2)

            mls_kernel = MarroneMLSKernel(smoothing_kernel, n_boundary_particles,
                                          n_fluid_particles)

            boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                         tank1.boundary.mass,
                                                         state_equation=state_equation,
                                                         MarronePressureExtrapolation(),
                                                         mls_kernel, smoothing_length)

            boundary_system = WallBoundarySystem(tank1.boundary, boundary_model)
            viscosity = boundary_system.boundary_model.viscosity

            semi = DummySemidiscretization()

            # In this testset, we verify that pressures in a static tank are extrapolated correctly.
            # Use constant density equal to the reference density of the state equation,
            # so that the pressure is constant zero. Then we test that the extrapolation also yields zero.
            @testset "Constant Zero Pressure" begin
                fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                            state_equation,
                                                            smoothing_kernel,
                                                            smoothing_length)
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
                                                                v_fluid,
                                                                semi)

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
                                        faces=(true, true, true, false), normal=true)

                fluid_system2 = WeaklyCompressibleSPHSystem(tank2.fluid, SummationDensity(),
                                                            state_equation,
                                                            smoothing_kernel,
                                                            smoothing_length)

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
                                                                v_fluid,
                                                                semi)

                # Test that pressure of the fluid is indeed constant
                @test all(isapprox.(fluid_system2.pressure, fluid_system2.pressure[1]))
                # Test that boundary pressure equals fluid pressure
                @test all(isapprox.(boundary_system.boundary_model.pressure,
                                    fluid_system2.pressure[1], atol=1.0e-10))
            end

            # In this test, we initialize a fluid with a hydrostatic pressure gradient
            # and check that this gradient is extrapolated correctly.
            @testset "Hydrostatic Pressure Gradient" begin
                particle_spacing = 0.1
                n_particles = 10
                n_layers = 2
                width = particle_spacing * n_particles
                height = particle_spacing * n_particles
                density = 257

                smoothing_kernel = SchoenbergCubicSplineKernel{2}()
                smoothing_length = 2 * particle_spacing
                state_equation = StateEquationCole(sound_speed=10, reference_density=257,
                                                   exponent=7)

                tank1 = RectangularTank(particle_spacing, (width, height), (width, height),
                                        density, n_layers=n_layers,
                                        faces=(true, true, true, false), normal=true)
                n_boundary_particles = size(tank1.boundary.coordinates, 2)
                n_fluid_particles = size(tank1.fluid.coordinates, 2)

                mls_kernel = MarroneMLSKernel(smoothing_kernel, n_boundary_particles,
                                              n_fluid_particles)

                boundary_model = BoundaryModelDummyParticles(tank1.boundary.density,
                                                             tank1.boundary.mass,
                                                             state_equation=state_equation,
                                                             MarronePressureExtrapolation(),
                                                             mls_kernel, smoothing_length)

                boundary_system = WallBoundarySystem(tank1.boundary, boundary_model)
                viscosity = boundary_system.boundary_model.viscosity

                semi = DummySemidiscretization()

                tank3 = RectangularTank(particle_spacing, (width, height), (width, height),
                                        density, acceleration=[0.0, -9.81],
                                        state_equation=state_equation, n_layers=n_layers,
                                        faces=(true, true, true, false), normal=true)

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

                TrixiParticles.boundary_pressure_extrapolation!(Val(true),
                                                                boundary_model,
                                                                boundary_system,
                                                                fluid_system3,
                                                                tank3.boundary.coordinates,
                                                                tank3.fluid.coordinates,
                                                                v_fluid,
                                                                v_fluid,
                                                                semi)

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

                # TODO: test failing, maximum difference between approximation and correct solution is
                # maximum(abs.(pressure - pressure-reference)) -> 813.51
                # @test all(isapprox.(pressure, pressure_reference, atol=4.0))
            end

            @testset "Numerical Consistency" begin
                mls_kernel = MarroneMLSKernel(smoothing_kernel, n_fluid_particles,
                                              n_fluid_particles)
                fluid_coords = tank1.fluid.coordinates
                fluid_system1 = WeaklyCompressibleSPHSystem(tank1.fluid, SummationDensity(),
                                                            state_equation,
                                                            smoothing_kernel,
                                                            smoothing_length)
                fluid_system1.cache.density .= tank1.fluid.density

                TrixiParticles.compute_basis_marrone(mls_kernel, fluid_system1,
                                                     fluid_system1, fluid_coords,
                                                     fluid_coords, semi)
                TrixiParticles.compute_momentum_marrone(mls_kernel, fluid_system1,
                                                        fluid_system1,
                                                        fluid_coords,
                                                        fluid_coords, v_fluid, semi,
                                                        smoothing_length)

                # We test that the `MarroneMLSKernel` correctly computes the 
                # first derivative of a constant function. 
                @testset "Zeroth Order Consistency" begin
                    zero_order_approx = zeros(n_fluid_particles)
                    constant = 3.0
                    TrixiParticles.foreach_point_neighbor(fluid_system1, fluid_system1,
                                                          fluid_coords, fluid_coords,
                                                          semi) do particle, neighbor,
                                                                   pos_diff, distance
                        neighbor_density = TrixiParticles.current_density(v_fluid,
                                                                          fluid_system1,
                                                                          neighbor)
                        neighbor_volume = neighbor_density != 0 ?
                                          TrixiParticles.hydrodynamic_mass(fluid_system1,
                                                                           neighbor) /
                                          neighbor_density : 0

                        zero_order_approx[particle] += TrixiParticles.boundary_kernel_marrone(mls_kernel,
                                                                                              particle,
                                                                                              neighbor,
                                                                                              distance,
                                                                                              smoothing_length) *
                                                       constant *
                                                       neighbor_volume
                    end
                    @test all(isapprox.(zero_order_approx, constant, atol=1.0e-10))
                end

                # We test that the `MarroneMLSKernel` correctly computes the 
                # first derivative of a linear function. 
                @testset "First Order Consistency" begin
                    first_order_approx = zeros(n_fluid_particles)
                    a = [1, 2]
                    b = 3
                    f(x) = dot(a, x) + b

                    linear_mapping = [f(fluid_coords[:, particle])
                                      for particle in 1:n_fluid_particles]
                    TrixiParticles.foreach_point_neighbor(fluid_system1, fluid_system1,
                                                          fluid_coords, fluid_coords,
                                                          semi) do particle, neighbor,
                                                                   pos_diff, distance
                        neighbor_density = TrixiParticles.current_density(v_fluid,
                                                                          fluid_system1,
                                                                          neighbor)
                        neighbor_volume = neighbor_density != 0 ?
                                          TrixiParticles.hydrodynamic_mass(fluid_system1,
                                                                           neighbor) /
                                          neighbor_density : 0
                        neighbor_val = f(fluid_coords[:, neighbor])

                        first_order_approx[particle] += TrixiParticles.boundary_kernel_marrone(mls_kernel,
                                                                                               particle,
                                                                                               neighbor,
                                                                                               distance,
                                                                                               smoothing_length) *
                                                        neighbor_val *
                                                        neighbor_volume
                    end
                    @test all(isapprox.(first_order_approx, linear_mapping, atol=1.0e-10))
                end
            end
        end
    end
end
