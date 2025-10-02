@testset verbose=true "WallBoundarySystem" begin
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

            system = WallBoundarySystem(initial_condition, model)
            TrixiParticles.update_positions!(system, 0, 0, 0, 0, 0, 0.0)

            @test system isa WallBoundarySystem
            @test ndims(system) == NDIMS
            @test system.coordinates == coordinates
            @test system.boundary_model == model
            @test system.prescribed_motion === nothing
            @test system.ismoving[] == false
        end
    end

    @testset verbose=true "Moving Boundaries" begin
        @testset "$(i+1)D" for i in 1:2
            NDIMS = i + 1
            coordinates = coordinates_[i]

            initial_condition = InitialCondition(; coordinates, mass, density)
            model = (; hydrodynamic_mass=3)

            function movement_function(x, t)
                if NDIMS == 2
                    return x + SVector(0.5 * t, 0.3 * t^2)
                end

                return x + SVector(0.5 * t, 0.3 * t^2, 0.1 * t^3)
            end

            is_moving(t) = t < 1.0
            bm = PrescribedMotion(movement_function, is_moving)
            system = WallBoundarySystem(initial_condition, model, prescribed_motion=bm)

            # Particles are moving at `t = 0.6`
            t = 0.6
            # `semi` is only passed to `@threaded`
            TrixiParticles.apply_prescribed_motion!(system, system.prescribed_motion,
                                                    SerialBackend(), t)
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

            # Not moving anymore, so the coordinates should not change
            t = 1.0
            # `semi` is only passed to `@threaded`
            TrixiParticles.apply_prescribed_motion!(system, system.prescribed_motion,
                                                    SerialBackend(), t)

            @test isapprox(new_coordinates, system.coordinates)

            # Move only a single particle
            new_coordinates = copy(coordinates_[i])
            new_velocity = zero(new_coordinates)
            new_acceleration = zero(new_coordinates)

            initial_condition = InitialCondition(; coordinates, mass, density)

            bm = PrescribedMotion(movement_function, is_moving, moving_particles=[2])
            system = WallBoundarySystem(initial_condition, model, prescribed_motion=bm)

            t = 0.1
            # `semi` is only passed to `@threaded`
            TrixiParticles.apply_prescribed_motion!(system, system.prescribed_motion,
                                                    SerialBackend(), t)

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

        system = WallBoundarySystem(initial_condition, model)

        show_compact = "WallBoundarySystem{2}((hydrodynamic_mass = 3,), nothing, 0.0, 0) with 2 particles"
        @test repr(system) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ WallBoundarySystem{2}                                                                            │
        │ ═════════════════════                                                                            │
        │ #particles: ………………………………………………… 2                                                                │
        │ boundary model: ……………………………………… (hydrodynamic_mass = 3,)                                         │
        │ movement function: ……………………………… nothing                                                          │
        │ adhesion coefficient: ……………………… 0.0                                                              │
        │ color: ……………………………………………………………… 0                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", system) == show_box
    end
end
