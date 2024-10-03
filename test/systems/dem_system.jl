include("../test_util.jl")
@testset verbose=true "DEMSystem" begin
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.25, 1.5]
        density = [990.0, 1000.0]

        initial_condition = InitialCondition(; coordinates, mass, density)
        system = DEMSystem(initial_condition, 2 * 10^5, 10e9, 0.3, acceleration=(0.0, 10.0))

        show_compact = "DEMSystem{2}(InitialCondition{Float64}(-1.0, [1.0 2.0; 1.0 2.0], [0.0 0.0; 0.0 0.0], [1.25, 1.5], [990.0, 1000.0], [0.0, 0.0]), 1.0e10, 0.3, 200000.0, 0.0001) with 2 particles"
        @test repr(system) == show_compact
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ DEMSystem{2}                                                                                     │
        │ ════════════                                                                                     │
        │ #particles: ………………………………………………… 2                                                                │
        │ elastic_modulus: …………………………………… 1.0e10                                                           │
        │ poissons_ratio: ……………………………………… 0.3                                                              │
        │ normal_stiffness: ………………………………… 200000.0                                                         │
        │ damping_coefficient: ………………………… 0.0001                                                           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end
