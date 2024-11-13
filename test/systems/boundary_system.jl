@testset verbose=true "BoundarySystem" begin
    coordinates_ = [
        [1.0 2.0
         1.0 2.0],
        [1.0 2.0
         1.0 2.0
         1.0 2.0]
    ]
    mass = [1.0, 1.0]
    density = [1000.0, 1000.0]

    @testset verbose=true "Constructor" begin
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]
            initial_condition = InitialCondition(; coordinates, mass, density)
            model = Val(:boundary_model)

            system = BoundarySPHSystem(initial_condition, model)
            TrixiParticles.update_positions!(system, 0, 0, 0, 0, 0, 0.0)

            @test system isa BoundarySPHSystem
            @test ndims(system) == NDIMS
            @test system.coordinates == coordinates
            @test system.boundary_model == model
            @test system.movement === nothing
            @test system.ismoving[] == false
        end
    end

    @testset verbose=true "Moving Boundaries" begin
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]

            initial_condition = InitialCondition(; coordinates, mass, density)
            model = (; hydrodynamic_mass=3)

            function movement_function(t)
                if NDIMS == 2
                    return SVector(0.5 * t, 0.3 * t^2)
                end

                return SVector(0.5 * t, 0.3 * t^2, 0.1 * t^3)
            end

            is_moving(t) = t < 1.0
            bm = BoundaryMovement(movement_function, is_moving)
            system = BoundarySPHSystem(initial_condition, model, movement=bm)

            # Moving
            t = 0.6
            system.movement(system, t)
            if NDIMS == 2
                new_coordinates = coordinates .+ [0.5 * t, 0.3 * t^2]
                new_velocity = [0.5, 0.6 * t] .* ones(size(new_coordinates))
                new_acceleration = [0.0, 0.6] .* ones(size(new_coordinates))
            else
                new_coordinates = coordinates .+ [0.5 * t, 0.3 * t^2, 0.1 * t^3]
                new_velocity = [0.5, 0.6 * t, 0.3 * t^2] .* ones(size(new_coordinates))
                new_acceleration = [0.0, 0.6, 0.6 * t] .* ones(size(new_coordinates))
            end

            @test isapprox(new_coordinates, system.coordinates)
            @test isapprox(new_velocity, system.cache.velocity)
            @test isapprox(new_acceleration, system.cache.acceleration)

            # Stop moving
            t = 1.0
            system.movement(system, t)

            @test isapprox(new_coordinates, system.coordinates)

            # Move only a single particle
            new_coordinates = copy(coordinates_[i])
            new_velocity = zero(new_coordinates)
            new_acceleration = zero(new_coordinates)

            initial_condition = InitialCondition(; coordinates, mass, density)

            bm = BoundaryMovement(movement_function, is_moving, moving_particles=[2])
            system = BoundarySPHSystem(initial_condition, model, movement=bm)

            t = 0.1
            system.movement(system, t)

            if NDIMS == 2
                new_coordinates[:, 2] .+= [0.5 * t, 0.3 * t^2]
                new_velocity[:, 2] .= [0.5, 0.6 * t]
                new_acceleration[:, 2] .= [0.0, 0.6]
            else
                new_coordinates[:, 2] .+= [0.5 * t, 0.3 * t^2, 0.1 * t^3]
                new_velocity[:, 2] .= [0.5, 0.6 * t, 0.3 * t^2]
                new_acceleration[:, 2] = [0.0, 0.6, 0.6 * t]
            end

            @test isapprox(new_coordinates, system.coordinates)
            @test isapprox(new_velocity, system.cache.velocity)
            @test isapprox(new_acceleration, system.cache.acceleration)
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

        show_compact = "BoundarySPHSystem{2}((hydrodynamic_mass = 3,), nothing, 0.0) with 2 particles"
        @test repr(system) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ BoundarySPHSystem{2}                                                                             │
        │ ════════════════════                                                                             │
        │ #particles: ………………………………………………… 2                                                                │
        │ boundary model: ……………………………………… (hydrodynamic_mass = 3,)                                         │
        │ movement function: ……………………………… nothing                                                          │
        │ adhesion coefficient: ……………………… 0.0                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end
