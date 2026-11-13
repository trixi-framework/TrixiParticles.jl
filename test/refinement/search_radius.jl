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

    @testset verbose=true "Open Boundary" begin
        # The open boundary particles are finer than the coarsest fluid particles.
        # They start with the smoothing length `1.2 * 0.05 = 0.06`.
        open_boundary_spacing = 0.05
        boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 0.5]),
                                     face_normal=[1.0, 0.0], boundary_type=InFlow(),
                                     open_boundary_layers=5, density=reference_density,
                                     particle_spacing=open_boundary_spacing)

        function wcsph_system(fluid)
            return WeaklyCompressibleSPHSystem(fluid; smoothing_kernel,
                                               smoothing_length=1.2 * max_spacing,
                                               state_equation,
                                               density_calculator=ContinuityDensity(),
                                               particle_refinement=refinement,
                                               buffer_size=4)
        end

        function edac_system(fluid)
            return EntropicallyDampedSPHSystem(fluid; smoothing_kernel,
                                               smoothing_length=1.2 * max_spacing,
                                               sound_speed,
                                               density_calculator=ContinuityDensity(),
                                               particle_refinement=refinement,
                                               buffer_size=4)
        end

        # One fluid particle next to the open boundary with the smoothing length
        # `1.2 * max_spacing = 0.12`. The closest open boundary particles are at
        # (-0.025, 0.225) and (-0.025, 0.275).
        function create_open_boundary_systems(position; create_fluid_system=wcsph_system)
            fluid = InitialCondition(; coordinates=reshape(position, 2, 1),
                                     velocity=[1.0; 0.0;;], density=reference_density,
                                     particle_spacing=max_spacing)
            fluid_system = create_fluid_system(fluid)
            open_boundary = OpenBoundarySystem(boundary_zone; fluid_system,
                                               buffer_size=4,
                                               boundary_model=BoundaryModelDynamicalPressureZhang())

            return fluid_system, open_boundary
        end

        function semidiscretize_open_boundary(fluid_system, open_boundary)
            semi = @test_logs Semidiscretization(fluid_system, open_boundary;
                                                 neighborhood_search=grid_nhs,
                                                 parallelization_backend=SerialBackend())
            ode = semidiscretize(semi, (0.0, 0.1))

            return ode.u0.x..., ode.p.semi
        end

        @testset "Search Radius" begin
            v_ode, u_ode,
            semi = semidiscretize_open_boundary(create_open_boundary_systems([0.175, 0.25])...)
            fluid_system, open_boundary = semi.systems

            nhs = TrixiParticles.get_neighborhood_search(open_boundary, fluid_system, semi)
            @test PointNeighbors.search_radius(nhs) ≈ 2 * h_max

            # The explicit search radius of the mirroring around the ghost nodes
            @test TrixiParticles.compact_support(open_boundary, fluid_system, 1) ≈
                  2 * h_max

            # The free-surface region of the open boundary still uses the support of the
            # open boundary particles
            @test TrixiParticles.compact_support(open_boundary, open_boundary, 1) ≈
                  2 * 1.2 * open_boundary_spacing
        end

        # The open boundary uses the average smoothing length of a pair. A pair can be
        # inside of the support of the fluid particle, but outside of the average support.
        function open_boundary_dv(position, create_fluid_system)
            systems = create_open_boundary_systems(position; create_fluid_system)
            v_ode, u_ode, semi = semidiscretize_open_boundary(systems...)
            fluid_system, open_boundary = semi.systems

            dv = zeros(size(TrixiParticles.wrap_v(v_ode, open_boundary, semi)))
            TrixiParticles.interact!(dv, TrixiParticles.wrap_v(v_ode, open_boundary, semi),
                                     TrixiParticles.wrap_u(u_ode, open_boundary, semi),
                                     TrixiParticles.wrap_v(v_ode, fluid_system, semi),
                                     TrixiParticles.wrap_u(u_ode, fluid_system, semi),
                                     open_boundary, fluid_system, semi)

            return dv
        end

        # EDAC also computes the pressure evolution of the open boundary particles
        @testset "Pair Outside of the Average Support With $name" for (name,
                                                                       create_fluid_system) in
                                                                      (("WCSPH",
                                                                        wcsph_system),
                                                                       ("EDAC",
                                                                        edac_system))
            # The distance to the closest open boundary particles is about 0.2016.
            # This is inside of the fluid support `2 * 0.12 = 0.24`, but outside of the
            # average support `2 * (0.06 + 0.12) / 2 = 0.18`.
            @test iszero(open_boundary_dv([0.175, 0.25], create_fluid_system))

            # Inside of the average support, the fluid particle contributes
            dv = open_boundary_dv([0.125, 0.25], create_fluid_system)
            @test !iszero(dv)
            @test all(isfinite, dv)
        end
    end
end
