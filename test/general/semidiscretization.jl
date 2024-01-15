# Use `@trixi_testset` to isolate the mock functions in a separate namespace
@trixi_testset "Semidiscretization" begin
    # Mock systems
    struct System1 <: TrixiParticles.System{3} end
    struct System2 <: TrixiParticles.System{3} end

    system1 = System1()
    system2 = System2()

    Base.ndims(::System1) = 2
    Base.ndims(::System2) = 2

    TrixiParticles.u_nvariables(::System1) = 3
    TrixiParticles.u_nvariables(::System2) = 4
    TrixiParticles.v_nvariables(::System1) = 3
    TrixiParticles.v_nvariables(::System2) = 2
    TrixiParticles.nparticles(::System1) = 2
    TrixiParticles.nparticles(::System2) = 3
    TrixiParticles.n_moving_particles(::System1) = 2
    TrixiParticles.n_moving_particles(::System2) = 3

    TrixiParticles.compact_support(::System1, neighbor) = 0.2
    TrixiParticles.compact_support(::System2, neighbor) = 0.2

    @testset verbose=true "Constructor" begin
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

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
        semi = Semidiscretization(system1, system2, neighborhood_search=nothing)

        show_compact = "Semidiscretization($System1(), $System2(), neighborhood_search=TrivialNeighborhoodSearch)"
        @test repr(semi) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ Semidiscretization                                                                               │
        │ ══════════════════                                                                               │
        │ #spatial dimensions: ………………………… 2                                                                │
        │ #systems: ……………………………………………………… 2                                                                │
        │ neighborhood search: ………………………… TrivialNeighborhoodSearch                                        │
        │ total #particles: ………………………………… 5                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", semi) == show_box
    end
end
