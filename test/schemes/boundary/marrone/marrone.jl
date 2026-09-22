using StaticArrays

@testset verbose=true "Dummy Particles with `MarronePressureExtrapolation`" begin
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

        tank = RectangularTank(particle_spacing, (width, height), (width, height),
                               density, n_layers=n_layers,
                               faces=(true, true, true, false))
        n_fluid_particles = nparticles(tank.fluid)
        n_boundary_particles = nparticles(tank.boundary)

        v_fluid = zeros(2, n_fluid_particles)
        mls_kernel = MarroneMLSKernel(smoothing_kernel, n_boundary_particles,
                                      eltype(tank.boundary))

        boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                     tank.boundary.mass,
                                                     state_equation=state_equation,
                                                     MarronePressureExtrapolation(),
                                                     mls_kernel, smoothing_length)

        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)
        (; viscosity, density_calculator) = boundary_system.boundary_model

        semi = DummySemidiscretization()

        TrixiParticles.initialize!(boundary_system, semi)

        # In this testset, we verify that pressures in a static tank are extrapolated correctly.
        # Use constant density equal to the reference density of the state equation,
        # so that the pressure is constant zero. Then we test that the extrapolation also yields zero.
        @testset "Constant Zero Pressure" begin
            fluid_system1 = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                        smoothing_length,
                                                        density_calculator=SummationDensity(),
                                                        state_equation)
            fluid_system1.cache.density .= tank.fluid.density

            TrixiParticles.compute_pressure!(fluid_system1, v_fluid, semi)

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity, density_calculator)

            TrixiParticles.boundary_pressure_extrapolation!(Val(true), boundary_model,
                                                            boundary_system,
                                                            fluid_system1,
                                                            tank.boundary.coordinates,
                                                            tank.fluid.coordinates,
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

            fluid_system2 = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                        smoothing_length,
                                                        density_calculator=SummationDensity(),
                                                        state_equation)

            fluid_system2.cache.density .= density
            v_fluid = zeros(2, TrixiParticles.nparticles(fluid_system2))
            TrixiParticles.compute_pressure!(fluid_system2, v_fluid, semi)

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity, density_calculator)

            TrixiParticles.boundary_pressure_extrapolation!(Val(true), boundary_model,
                                                            boundary_system,
                                                            fluid_system2,
                                                            tank.boundary.coordinates,
                                                            tank.fluid.coordinates,
                                                            v_fluid,
                                                            v_fluid,
                                                            semi)

            # Test that pressure of the fluid is indeed constant
            @test all(isapprox.(fluid_system2.pressure, fluid_system2.pressure[1]))
            # Test that boundary pressure equals fluid pressure
            @test all(isapprox.(boundary_system.boundary_model.pressure,
                                fluid_system2.pressure[1], atol=1.0e-10))
        end

        # In this test, we artificially set a perfectly linear pressure gradient in the fluid.
        # Because the Marrone MLS kernel is 1st-order consistent, it must extrapolate this
        # linear field to the boundary particles to machine precision.
        @testset "Hydrostatic Pressure Gradient" begin
            # Create a standard tank (no physical acceleration needed since we override it)
            tank3 = RectangularTank(particle_spacing, (width, height), (width, height),
                                    density, n_layers=n_layers,
                                    faces=(true, true, true, false))

            fluid_system3 = WeaklyCompressibleSPHSystem(tank3.fluid;
                                                        smoothing_kernel,
                                                        smoothing_length,
                                                        density_calculator=SummationDensity(),
                                                        state_equation,
                                                        acceleration=[0.0, -9.81])

            # Manually construct a perfect linear pressure field
            gravity = 9.81
            water_height = height
            fluid_coords = tank3.fluid.coordinates

            for particle in TrixiParticles.eachparticle(fluid_system3)
                y_fluid = fluid_coords[2, particle]

                # P = rho * g * h
                fluid_system3.pressure[particle] = density * gravity *
                                                   (water_height - y_fluid)
                fluid_system3.cache.density[particle] = density
            end

            TrixiParticles.set_zero!(boundary_model.pressure)
            TrixiParticles.reset_cache!(boundary_system.boundary_model.cache,
                                        viscosity, density_calculator)

            TrixiParticles.boundary_pressure_extrapolation!(Val(true),
                                                            boundary_model,
                                                            boundary_system,
                                                            fluid_system3,
                                                            boundary_system.coordinates,
                                                            fluid_coords,
                                                            v_fluid,
                                                            v_fluid,
                                                            semi)

            expected_pressure = zeros(TrixiParticles.nparticles(boundary_system))
            for particle in TrixiParticles.eachparticle(boundary_system)
                y_boundary = boundary_system.coordinates[2, particle]
                expected_pressure[particle] = density * gravity *
                                              (water_height - y_boundary)
            end

            @test all(isapprox.(boundary_system.boundary_model.pressure, expected_pressure,
                                atol=1e-8))
        end

        @testset "Numerical Consistency" begin
            mls_kernel = MarroneMLSKernel(smoothing_kernel, n_fluid_particles,
                                          eltype(tank.fluid))
            fluid_coords = tank.fluid.coordinates
            fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                                       smoothing_length,
                                                       density_calculator=SummationDensity(),
                                                       state_equation)
            fluid_system.cache.density .= tank.fluid.density

            NDIMS = ndims(fluid_system)
            ELTYPE = eltype(fluid_coords)
            (; momentum_inv) = mls_kernel

            tolerance = ELTYPE(1e-9) * smoothing_length^(2 * NDIMS)
            for particle in eachparticle(fluid_system)
                momentum_particle = TrixiParticles.compute_momentum(mls_kernel,
                                                                    fluid_system,
                                                                    fluid_system,
                                                                    fluid_coords,
                                                                    fluid_coords, v_fluid,
                                                                    semi,
                                                                    smoothing_length,
                                                                    particle)
                momentum_inv[particle] = abs(det(momentum_particle)) < tolerance ?
                                         SMatrix{NDIMS+1, NDIMS+1, ELTYPE, (NDIMS+1)^2}(I) :
                                         inv(momentum_particle)
            end

            # We test that the `MarroneMLSKernel` correctly computes the 
            # first derivative of a constant function. 
            @testset "Zeroth Order Consistency" begin
                zero_order_approx = zeros(n_fluid_particles)
                constant = 3.0
                TrixiParticles.foreach_point_neighbor(fluid_system, fluid_system,
                                                      fluid_coords, fluid_coords,
                                                      semi) do particle, neighbor,
                                                               pos_diff, distance
                    neighbor_density = TrixiParticles.current_density(v_fluid,
                                                                      fluid_system,
                                                                      neighbor)
                    neighbor_volume = neighbor_density != 0 ?
                                      TrixiParticles.hydrodynamic_mass(fluid_system,
                                                                       neighbor) /
                                      neighbor_density : 0

                    zero_order_approx[particle] += TrixiParticles.boundary_kernel_marrone(mls_kernel,
                                                                                          smoothing_length,
                                                                                          particle,
                                                                                          pos_diff,
                                                                                          distance) *
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
                TrixiParticles.foreach_point_neighbor(fluid_system, fluid_system,
                                                      fluid_coords, fluid_coords,
                                                      semi) do particle, neighbor,
                                                               pos_diff, distance
                    neighbor_density = TrixiParticles.current_density(v_fluid,
                                                                      fluid_system,
                                                                      neighbor)
                    neighbor_volume = neighbor_density != 0 ?
                                      TrixiParticles.hydrodynamic_mass(fluid_system,
                                                                       neighbor) /
                                      neighbor_density : 0
                    neighbor_val = f(fluid_coords[:, neighbor])

                    first_order_approx[particle] += TrixiParticles.boundary_kernel_marrone(mls_kernel,
                                                                                           smoothing_length,
                                                                                           particle,
                                                                                           pos_diff,
                                                                                           distance) *
                                                    neighbor_val *
                                                    neighbor_volume
                end
                @test all(isapprox.(first_order_approx, linear_mapping, atol=1.0e-10))
            end
        end
    end
end
