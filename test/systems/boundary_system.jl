@testset verbose=true "BoundarySystem" begin
    @testset verbose=true "Constructor" begin
        coordinates_ = [
            [1.0 2.0
             1.0 2.0],
            [1.0 2.0
             1.0 2.0
             1.0 2.0],
        ]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            initial_condition = InitialCondition(; coordinates, mass, density)
            model = Val(:boundary_model)

            system = BoundarySPHSystem(initial_condition, model)

            @test system isa BoundarySPHSystem
            @test ndims(system) == NDIMS
            @test system.coordinates == coordinates
            @test system.boundary_model == model
            @test system.movement === nothing
            @test system.ismoving == [false]
        end
    end

    # Use `@trixi_testset` to isolate the mock functions in a separate namespace
    @trixi_testset "show" begin
        coordinates = [1.0 2.0
                       1.0 2.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        initial_condition = InitialCondition(; coordinates, mass, density)
        model = (; hydrodynamic_mass=3)

        system = BoundarySPHSystem(initial_condition, model)

        show_compact = "BoundarySPHSystem{2}((hydrodynamic_mass = 3,), nothing) with 1 particles"
        @test repr(system) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ BoundarySPHSystem{2}                                                                             │
        │ ════════════════════                                                                             │
        │ #particles: ………………………………………………… 1                                                                │
        │ boundary model: ……………………………………… (hydrodynamic_mass = 3,)                                         │
        │ movement function: ……………………………… nothing                                                          │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end
