# Use `@trixi_testset` to isolate the mock functions in a separate namespace
@trixi_testset "Semidiscretization" begin
    # Mock systems
    system1 = Val(:system1)
    system2 = Val(:system2)

    Base.ndims(::Val{:system1}) = 2
    Base.ndims(::Val{:system2}) = 2

    TrixiParticles.u_nvariables(::Val{:system1}) = 3
    TrixiParticles.u_nvariables(::Val{:system2}) = 4
    TrixiParticles.v_nvariables(::Val{:system1}) = 3
    TrixiParticles.v_nvariables(::Val{:system2}) = 2
    TrixiParticles.nparticles(::Val{:system1}) = 2
    TrixiParticles.nparticles(::Val{:system2}) = 3
    TrixiParticles.n_moving_particles(::Val{:system1}) = 2
    TrixiParticles.n_moving_particles(::Val{:system2}) = 3

    TrixiParticles.compact_support(::Val{:system1}, neighbor) = 0.2
    TrixiParticles.compact_support(::Val{:system2}, neighbor) = 0.2

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2)

        # Verification
        @test semi.ranges_u == (1:6, 7:18)
        @test semi.ranges_v == (1:6, 7:12)

        nhs = ((TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(2)),
                TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(3))),
               (TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(2)),
                TrixiParticles.TrivialNeighborhoodSearch{2}(0.2, Base.OneTo(3))))
        @test semi.neighborhood_searches == nhs
    end

    @testset verbose=true "show" begin
        semi = Semidiscretization(system1, system2)

        show_compact = "Semidiscretization(Val{:system1}(), Val{:system2}(), neighborhood_search=TrivialNeighborhoodSearch)"
        @test repr(semi) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Semidiscretization                                                                               │
        │ ══════════════════                                                                               │
        │ #spatial dimensions: ………………………… 2                                                                │
        │ #systems: ……………………………………………………… 2                                                                │
        │ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
        │ damping coefficient: ………………………… nothing                                                          │
        │ total #particles: ………………………………… 5                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", semi) == show_box
    end
end
