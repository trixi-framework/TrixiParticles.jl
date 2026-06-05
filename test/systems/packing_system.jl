@testset verbose=true "ParticlePackingSystem" begin
    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        data_dir = pkgdir(TrixiParticles, "examples", "preprocessing", "data")
        geometry = load_geometry(joinpath(data_dir, "circle.asc"))

        initial_condition = ComplexShape(geometry; particle_spacing=0.1, density=1.0)
        system = ParticlePackingSystem(initial_condition,
                                       signed_distance_field=SignedDistanceField(geometry,
                                                                                 0.1),
                                       background_pressure=1.0)

        show_compact = "ParticlePackingSystem{2}(SchoenbergQuinticSplineKernel{2}()) with 307 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ParticlePackingSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
        │ #particles: ………………………………………………… 307                                                              │
        │ smoothing kernel: ………………………………… SchoenbergQuinticSplineKernel                                    │
        │ place_on_shell: ……………………………………… no                                                               │
        │ boundary: ……………………………………………………… no                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box

        system = ParticlePackingSystem(initial_condition,
                                       signed_distance_field=SignedDistanceField(geometry,
                                                                                 0.1),
                                       background_pressure=1.0, is_boundary=true)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ParticlePackingSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ neighborhood search: ………………………… GridNeighborhoodSearch                                           │
        │ #particles: ………………………………………………… 307                                                              │
        │ smoothing kernel: ………………………………… SchoenbergQuinticSplineKernel                                    │
        │ place_on_shell: ……………………………………… no                                                               │
        │ boundary: ……………………………………………………… yes                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box

        system = ParticlePackingSystem(initial_condition,
                                       signed_distance_field=nothing,
                                       background_pressure=1.0)
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ParticlePackingSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ neighborhood search: ………………………… Nothing                                                          │
        │ #particles: ………………………………………………… 307                                                              │
        │ smoothing kernel: ………………………………… SchoenbergQuinticSplineKernel                                    │
        │ place_on_shell: ……………………………………… no                                                               │
        │ boundary: ……………………………………………………… no                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end

    @trixi_testset "signed distance interpolation radius" begin
        initial_condition = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                             density=1.0, particle_spacing=0.1)

        # Regression setup for two different radii in `ParticlePackingSystem`:
        # - `smoothing_length` controls the packing force radius.
        # - `smoothing_length_interpolation` controls the internal neighborhood search
        #   used for interpolating signed-distance values and normals.
        #
        # The only signed-distance sample is intentionally placed at x = 2.0.
        # With the default SchoenbergQuinticSplineKernel, the packing support radius
        # from `smoothing_length = 0.1` is 3 * 0.1 = 0.3, so this sample is too far
        # away for the packing radius. It is inside the interpolation support radius
        # from `smoothing_length_interpolation = 1.0`, which is 3 * 1.0 = 3.0.
        # Therefore this test fails if the SDF neighborhood search is initialized
        # incorrectly with something else than `smoothing_length_interpolation`.
        signed_distance_field = TrixiParticles.SignedDistanceField([SVector(2.0, 0.0)],
                                                                   [SVector(1.0, 0.0)],
                                                                   [0.0], 1.0,
                                                                   false, 0.1)
        system = ParticlePackingSystem(initial_condition;
                                       signed_distance_field,
                                       smoothing_length=0.1,
                                       smoothing_length_interpolation=1.0,
                                       background_pressure=1.0)

        u = copy(initial_condition.coordinates)

        TrixiParticles.constrain_particles_onto_surface!(u, system,
                                                         DummySemidiscretization())

        # The handcrafted SDF sample has signed distance 0 and normal (1, 0).
        # For non-boundary particles with `place_on_shell = false`, `shift_length`
        # is `particle_spacing / 2`. `constrain_particle!` applies
        # `u -= (distance_signed + shift_length) * normal`, so the x-coordinate
        # moves from 0.0 to -0.05 and the y-coordinate stays unchanged.
        expected_shift = initial_condition.particle_spacing / 2
        @test isapprox(u[1, 1], -expected_shift)
        @test isapprox(u[2, 1], initial_condition.coordinates[2, 1])
    end
end
