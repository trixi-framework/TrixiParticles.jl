@testset verbose=true "Search Radius" begin
    smoothing_kernel = WendlandC2Kernel{2}()
    sound_speed = 10.0
    reference_density = 1000.0
    state_equation = StateEquationCole(; sound_speed, reference_density, exponent=7)
    max_spacing = 0.1
    max_mass_factor = 1.05
    refinement = TrixiParticles.ParticleRefinement(; spacing_ratio=1.2, min_spacing=0.05,
                                                   max_spacing, reference_density,
                                                   max_mass_factor)

    # The default update strategy does not support inactive particles
    grid_nhs = GridNeighborhoodSearch{2}(update_strategy=SerialUpdate())

    function create_systems(particle_spacing; smoothing_length=1.2 * particle_spacing)
        tank = RectangularTank(particle_spacing, (0.5, 0.5), (1.0, 1.0),
                               reference_density; n_layers=3)

        # The wall uses a smaller compact support than the coarsest fluid particles
        boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                                     tank.boundary.mass,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel,
                                                     1.2 * particle_spacing;
                                                     state_equation)

        fluid_system = EntropicallyDampedSPHSystem(tank.fluid; smoothing_kernel,
                                                   smoothing_length, sound_speed,
                                                   particle_refinement=refinement,
                                                   buffer_size=4)
        boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

        return fluid_system, boundary_system
    end

    # The search radius of all searches involving `fluid_system`
    function search_radii(fluid_system, boundary_system, semi)
        pairs = ((fluid_system, fluid_system), (fluid_system, boundary_system),
                 (boundary_system, fluid_system))

        return map(pairs) do (system, neighbor)
            nhs = TrixiParticles.get_neighborhood_search(system, neighbor, semi)
            return PointNeighbors.search_radius(nhs)
        end
    end

    # Smoothing length of a particle with `max_mass_factor` times the reference mass
    # at `max_spacing`. The `smoothing_length_factor` of the refinement is 1.2.
    h_max = 1.2 * max_mass_factor^(1 / 2) * max_spacing

    setups = [
        # The coarsest particles after splitting determine the search radius
        ("Initial Spacing `max_spacing`", max_spacing, 2 * h_max),
        # The initial particles are coarser than all particles after splitting,
        # so they determine the search radius.
        ("Initial Spacing Above Maximum", 0.125, 2 * 1.2 * 0.125)
    ]

    @testset verbose=true "$name, $(nameof(handler))" for (name, particle_spacing,
                                                           expected_radius) in setups,
                                                          handler in (SharedNHSHandler,
                                                           PairsNHSHandler)

        fluid_system, boundary_system = create_systems(particle_spacing)

        # A neighborhood search with a too small search radius would emit a warning
        semi = @test_logs Semidiscretization(fluid_system, boundary_system;
                                             neighborhood_search=grid_nhs,
                                             neighborhood_search_handler=handler,
                                             parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 0.1))
        v_ode, u_ode = ode.u0.x
        (; semi) = ode.p
        fluid_system, boundary_system = semi.systems

        @testset "Initialization" begin
            for radius in search_radii(fluid_system, boundary_system, semi)
                @test radius >= expected_radius || radius ≈ expected_radius
            end
        end

        # Resizing recreates the neighborhood searches and must keep their search radius
        @testset "After Resizing" begin
            TrixiParticles.resize_systems!(v_ode, u_ode, semi,
                                           [nparticles(fluid_system) + 7,
                                               nparticles(boundary_system)])

            for radius in search_radii(fluid_system, boundary_system, semi)
                @test radius >= expected_radius || radius ≈ expected_radius
            end
        end
    end

    @testset verbose=true "Warning for Too Large Smoothing Length" begin
        # The initial smoothing length is larger than the smoothing length of
        # the initial particle spacing, which determines the search radius.
        fluid_system,
        boundary_system = create_systems(max_spacing;
                                         smoothing_length=2 * max_spacing)

        @test_logs (:warn, r"smaller than the compact support") match_mode=:any Semidiscretization(fluid_system,
                                                                                                   boundary_system;
                                                                                                   neighborhood_search=grid_nhs,
                                                                                                   parallelization_backend=SerialBackend())
    end

    @testset verbose=true "Second Fluid" begin
        # The second fluid is next to the tank and has the smoothing length
        # `1.2 * 0.05 = 0.06`, which is smaller than the one of the coarsest refined
        # fluid particles.
        second_spacing = 0.05
        second_fluid = RectangularShape(second_spacing, (4, 4), (1.5, 0.1);
                                        density=reference_density)

        function second_fluid_system(; kwargs...)
            return WeaklyCompressibleSPHSystem(second_fluid; smoothing_kernel,
                                               smoothing_length=1.2 * second_spacing,
                                               state_equation,
                                               density_calculator=ContinuityDensity(),
                                               kwargs...)
        end

        second_setups = [
            ("Without Refinement", () -> second_fluid_system()),
            # Both fluids have variable smoothing lengths, which must not call
            # `compact_support` recursively.
            ("With Refinement",
             () -> second_fluid_system(particle_refinement=refinement, buffer_size=4))
        ]

        @testset verbose=true "$name" for (name, create_second_system) in second_setups
            fluid_system, boundary_system = create_systems(max_spacing)

            # The search of the second fluid would be too small without the refined
            # neighbor, which would emit a warning.
            semi = @test_logs Semidiscretization(fluid_system, boundary_system,
                                                 create_second_system();
                                                 neighborhood_search=grid_nhs,
                                                 parallelization_backend=SerialBackend())
            fluid_system, _, second_system = semi.systems

            # Pairs of both fluids interact within the larger compact support
            for (system, neighbor) in ((second_system, fluid_system),
                 (fluid_system, second_system))
                nhs = TrixiParticles.get_neighborhood_search(system, neighbor, semi)
                @test PointNeighbors.search_radius(nhs) ≈ 2 * h_max
            end
        end

        @testset "Self-Interaction Without Refinement" begin
            fluid_system, boundary_system = create_systems(max_spacing)
            semi = Semidiscretization(fluid_system, boundary_system, second_fluid_system();
                                      neighborhood_search=grid_nhs,
                                      neighborhood_search_handler=PairsNHSHandler,
                                      parallelization_backend=SerialBackend())
            second_system = semi.systems[3]

            # The searches between systems without refinement are unchanged
            nhs = TrixiParticles.get_neighborhood_search(second_system, second_system, semi)
            @test PointNeighbors.search_radius(nhs) ≈ 2 * 1.2 * second_spacing
        end
    end
end
