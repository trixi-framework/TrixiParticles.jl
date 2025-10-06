@testset verbose=true "Marrone Dummy Particles" begin
    # struct DummySemidiscretization
    #     parallelization_backend::Any

    #     function DummySemidiscretization(; parallelization_backend=SerialBackend())
    #         new(parallelization_backend)
    #     end
    # end

    # @inline function PointNeighbors.parallel_foreach(f, iterator, semi::DummySemidiscretization)
    #     PointNeighbors.parallel_foreach(f, iterator, semi.parallelization_backend)
    # end

    # @inline function TrixiParticles.get_neighborhood_search(system, neighbor_system,
    #                                                         ::DummySemidiscretization)
    #     search_radius = TrixiParticles.compact_support(system, neighbor_system)
    #     eachpoint = TrixiParticles.eachparticle(neighbor_system)
    #     return TrixiParticles.TrivialNeighborhoodSearch{ndims(system)}(; search_radius,
    #                                                                 eachpoint)
    # end

    # @inline function TrixiParticles.get_neighborhood_search(system,
    #                                                         semi::DummySemidiscretization)
    #     return get_neighborhood_search(system, system, semi)
    # end

    @testset "Boundary Normals" begin end

    @testset "MarroneMLSKernel" begin
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
        boundary_system = BoundarySPHSystem(tank1.boundary, boundary_model)
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

        #Check which boundary particle has what neighboring fluid particles. 
        #Each boundary particle should have exactly 1 neighboring particle with distance 1.
        # TrixiParticles.foreach_point_neighbor(boundary_system, fluid_system1, boundary_coords, fluid_coords, semi) do particle, neighbor, pos_diff, distance
        #    print(particle, neighbor, "\n")
        # end 

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

        @testset "Pressure Extrapolation Marrone" begin
            particle_spacing = 1.0
            n_particles = 10
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

            boundary_system = BoundarySPHSystem(tank1.boundary, boundary_model)
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

                @test all(isapprox.(pressure, pressure_reference, atol=4.0))
            end
        end
    end
end

# # Plot the tank and normal vectors
# inds_neg = findall(x->x==0.0, boundary_system.boundary_model.pressure)
# inds_pos = findall(x->x!=0.0, boundary_system.boundary_model.pressure)

# x_f, y_f = eachrow(tank2.fluid.coordinates)
# x_b, y_b = eachrow(tank2.boundary.coordinates)

# volume_boundary = zeros(n_boundary_particles)
# for particle in TrixiParticles.eachparticle(boundary_system)
#     density = TrixiParticles.current_density(v_fluid, boundary_system, particle)
#     volume_boundary[particle] = density != 0 ? TrixiParticles.hydrodynamic_mass(boundary_system, particle) / density : 0
# end

# volume_fluid = zeros(n_fluid_particles)
# for particle in TrixiParticles.eachparticle(fluid_system2)
#     density = TrixiParticles.current_density(v_fluid, fluid_system2, particle)
#     volume_fluid[particle] = density != 0 ? TrixiParticles.hydrodynamic_mass(fluid_system2, particle) / density : 0
# end
# plot(tank1.fluid, tank2.boundary, labels=["fluid" "boundary"], xlims=[-0.25, 1.25], ylims=[-0.25, 1.125])
# # plot(x_f, y_f, seriestype=:scatter, color=:red, label="Fluid")
# # plot!(x_b, y_b, seriestype=:scatter, color=:blue, label="Boundary")

# x_pos, y_pos = eachrow(boundary_system.coordinates[:,inds_pos])
# scatter!(x_pos, y_pos, color=:green)
# u_pos, v_pos = eachrow(boundary_system.initial_condition.normals[:,inds_pos])
# quiver!(x_pos, y_pos, quiver=(u_pos, v_pos), aspect_ratio=1)

# (; pressure, cache, viscosity, density_calculator, smoothing_kernel,
# smoothing_length) = boundary_model
# (; normals) = boundary_system.initial_condition
# system_coords = tank2.boundary.coordinates
# neighbor_coords = tank2.fluid.coordinates
# neighbor_system = fluid_system2
# system = boundary_system
# v = v_neighbor_system = v_fluid

# interpolation_coords = system_coords + (2 * normals) # Need only be computed once -> put into cache 

# TrixiParticles.compute_basis_marrone(smoothing_kernel, boundary_system,
#                                     fluid_system2, system_coords,
#                                     neighbor_coords, semi)
# TrixiParticles.compute_momentum_marrone(smoothing_kernel, boundary_system,
#                                         fluid_system2, system_coords,
#                                         neighbor_coords, v_fluid, semi,
#                                         smoothing_length)

# particle = 1
# force = neighbor_system.acceleration
# particle_density = isnan(TrixiParticles.current_density(v, system, particle)) ?
#                 0 : TrixiParticles.current_density(v, system, particle) # This can return NaN 
# particle_boundary_distance = norm(normals[:, particle]) # distance from boundary particle to the boundary
# particle_normal = particle_boundary_distance != 0 ?
#                 normals[:, particle] / particle_boundary_distance :
#                 zeros(size(normals[:, particle])) # normal unit vector to the boundary

# # Checked everything here for NaN's except the dot()
# pressure[particle] += 2 * particle_boundary_distance * particle_density *
#                     dot(force, particle_normal)

# # Plot the interpolation points 
# x_inter, y_inter = eachrow(interpolation_coords)
# p = plot(tank1.fluid, tank1.boundary, labels=["fluid" "boundary"], xlims=[-n_layers-1, n_particles+n_layers+1], ylims=[-n_layers-1,n_particles+1])
# scatter!(p, x_inter, y_inter, color=:red, markersize=5)

# # Plot the tank and show the index of each particle 
# p=plot(tank1.fluid, tank1.boundary, labels=["fluid" "boundary"], xlims=[-n_layers-1, n_particles+n_layers+1], ylims=[-n_layers-1,n_particles+1])
# for i in 1:n_boundary_particles
#        xi = boundary_coords[1,i]
#        yi = boundary_coords[2,i]
#        annotate!(p, xi, yi, text(string(i), :center, 10, :black))
# end
# for i in 1:n_fluid_particles
#        xi = fluid_coords[1,i]
#        yi = fluid_coords[2,i]
#        annotate!(p, xi, yi, text(string(i), :center, 10, :black))
# end
# display(p)

# for i in 1:n_boundary_particles
#   print(mls_kernel.momentum[i,:,:] == I(3), "\n")
# end
