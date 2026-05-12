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
        [ Info: No `SignedDistanceField` provided. Particles will not be constraint onto a geoemtric surface.
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ ParticlePackingSystem{2}                                                                         │
        │ ════════════════════════                                                                         │
        │ neighborhood search: ………………………… Nothing                                                          │
        │ #particles: ………………………………………………… 307                                                              │
        │ smoothing kernel: ………………………………… SchoenbergQuinticSplineKernel                                    │
        │ place_on_shell: ……………………………………… no                                                               │
        │ boundary: ……………………………………………………… no                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
    end

    @trixi_testset "signed distance interpolation radius" begin
        initial_condition = InitialCondition(; coordinates=reshape([0.0, 0.0], 2, 1),
                                             density=1.0, particle_spacing=0.1)
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

        @test u[1, 1] < initial_condition.coordinates[1, 1]
    end
end
