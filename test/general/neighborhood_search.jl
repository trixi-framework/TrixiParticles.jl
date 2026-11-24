@testset verbose=true "Deactivate Out of Bounds Particles" begin
    struct MockSystemOutOfBounds <: TrixiParticles.AbstractSystem{2}
        buffer::TrixiParticles.SystemBuffer
    end

    TrixiParticles.nparticles(system::MockSystemOutOfBounds) = length(system.buffer.active_particle)
    Base.eltype(system::MockSystemOutOfBounds) = Float64

    @testset "Particles Inside Bounds" begin
        # Setup: 5 particles, all inside bounds
        buffer = TrixiParticles.SystemBuffer(5, 0)
        system = MockSystemOutOfBounds(buffer)

        u = [-0.5 0.0 0.5 -0.8 0.8
             -0.5 -0.5 0.0 0.5 0.5]

        cell_list = TrixiParticles.FullGridCellList(; min_corner=(-1.0, -1.0),
                                                    max_corner=(1.0, 1.0),
                                                    search_radius=0.1)
        dummy_nhs = (; cell_size=0.1, n_cells=0, periodic_box=nothing, cell_list)
        semi = DummySemidiscretization()

        # All particles should remain active
        initial_count = count(buffer.active_particle)
        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, dummy_nhs,
                                                           cell_list, u, u, semi)
        @test count(buffer.active_particle) == initial_count
    end

    @testset "Particles Outside Bounds" begin
        # Setup: 5 particles, some outside bounds
        buffer = TrixiParticles.SystemBuffer(5, 0)
        system = MockSystemOutOfBounds(buffer)

        # Particles 3 and 5 are outside the bounds
        u = [-0.5 0.0 2.0 -0.8 -2.0
             -0.5 -0.5 0.0 0.5 0.5]

        cell_list = TrixiParticles.FullGridCellList(; min_corner=(-1.0, -1.0),
                                                    max_corner=(1.0, 1.0),
                                                    search_radius=0.1)
        dummy_nhs = (; cell_size=0.1, n_cells=0, periodic_box=nothing, cell_list)
        semi = DummySemidiscretization()

        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, dummy_nhs,
                                                           cell_list, u, u, semi)

        # Particles 3 and 5 should be deactivated
        @test buffer.active_particle[3] == false
        @test buffer.active_particle[5] == false
        # Others should still be active
        @test buffer.active_particle[1] == true
        @test buffer.active_particle[2] == true
        @test buffer.active_particle[4] == true

        @test TrixiParticles.each_active_particle(system, buffer) == [1, 2, 4]
    end

    @testset "Edge Cases" begin
        # Test the 1001//1000 padding logic
        buffer = TrixiParticles.SystemBuffer(3, 0)
        system = MockSystemOutOfBounds(buffer)

        # Particles directly at the boundaries (should remain active)
        u = [-1.0 1.0 0.0
             -1.0 1.0 0.0]

        cell_list = TrixiParticles.FullGridCellList(; min_corner=(-1.0, -1.0),
                                                    max_corner=(1.0, 1.0),
                                                    search_radius=0.1)
        dummy_nhs = (; cell_size=0.1, n_cells=0, periodic_box=nothing, cell_list)
        semi = DummySemidiscretization()

        TrixiParticles.deactivate_out_of_bounds_particles!(system, buffer, dummy_nhs,
                                                           cell_list, u, u, semi)

        # All should still be active
        @test all(buffer.active_particle)
    end
end

@testset verbose=true "CombinedNHSHandler" begin
    # Test that the `CombinedNHSHandler` yields the same results as the `SharedNHSHandler`.
    @testset "$(NDIMS)D Fluid-Wall" for NDIMS in (2, 3)
        particle_spacing = 0.1
        fluid_size = NDIMS == 2 ? (2.0, 1.0) : (0.8, 0.9, 0.7)
        tank_size = NDIMS == 2 ? (2.0, 1.5) : (0.8, 0.9, 1.0)
        state_equation = StateEquationCole(; sound_speed=10.0, exponent=7,
                                           reference_density=1000.0)
        acceleration = NDIMS == 2 ? (0.0, -9.81) : (0.0, 0.0, -9.81)
        tank = RectangularTank(particle_spacing, fluid_size, tank_size, 1000.0;
                               n_layers=3)

        smoothing_length = 1.5 * particle_spacing
        smoothing_kernel = WendlandC2Kernel{NDIMS}()
        fluid_system = WeaklyCompressibleSPHSystem(tank.fluid; smoothing_length,
                                                   state_equation, smoothing_kernel,
                                                   acceleration,
                                                   density_calculator=ContinuityDensity(),
                                                   viscosity=ArtificialViscosityMonaghan(alpha=0.02,
                                                                                         beta=0.1),
                                                   buffer_size=5)
        boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                     tank.boundary.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel,
                                                     smoothing_length; state_equation)
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

        min_corner = minimum(tank.boundary.coordinates, dims=2) .- particle_spacing
        max_corner = maximum(tank.boundary.coordinates, dims=2) .+ particle_spacing
        cell_list = FullGridCellList(; min_corner, max_corner)
        neighborhood_search = GridNeighborhoodSearch{NDIMS}(; cell_list)

        function rhs(neighborhood_search_handler)
            semi = Semidiscretization(fluid_system, boundary_system;
                                      neighborhood_search, neighborhood_search_handler)
            ode = semidiscretize(semi, (0.0, 1.0))
            semi = ode.p.semi
            fluid_system = semi.systems[1]
            v_ode, u_ode = copy(ode.u0).x

            # Move the fluid particles to test the update of the neighborhood search.
            u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
            u .+= 0.2 * particle_spacing .* sin.(37 .* u)

            # Deactivate a few particles.
            v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
            for particle in (1, 17, 100)
                TrixiParticles.deactivate_particle!(fluid_system, particle, v, u)
            end
            TrixiParticles.update_system_buffer!(fluid_system.buffer)

            dv_ode = zero(v_ode)
            TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

            return dv_ode, semi
        end

        dv_expected, _ = rhs(SharedNHSHandler)
        dv, semi = rhs(CombinedNHSHandler)

        @test count(semi.systems[1].buffer.active_particle) ==
              nparticles(semi.systems[1]) - 8
        @test all(isfinite, dv)
        @test isapprox(dv, dv_expected, atol=1e-14)

        # Only the cell list of the fluid has been updated because the wall is not moving.
        @test semi.neighborhood_search_handler.systems_moving == [true, false]
    end
end
