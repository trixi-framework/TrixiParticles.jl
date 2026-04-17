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

    @testset verbose=true "High-level Dummy-Particle Builder" begin
        boundary_coordinates = [1.0 2.0
                                1.0 2.0]
        fluid_coordinates = [0.0 0.5
                             0.0 0.0]

        boundary_ic = InitialCondition(; coordinates=boundary_coordinates, mass, density)
        fluid_ic = InitialCondition(; coordinates=fluid_coordinates, mass, density)

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.8
        state_equation = StateEquationCole(; sound_speed=15.0, reference_density=1000.0,
                                           exponent=1)
        viscosity = ViscosityAdami(nu=1e-6)

        fluid_system = WeaklyCompressibleSPHSystem(fluid_ic, ContinuityDensity(),
                                                   state_equation, smoothing_kernel,
                                                   smoothing_length,
                                                   correction=KernelCorrection(),
                                                   reference_particle_spacing=0.1)

        boundary_model = BoundaryModelDummyParticles(boundary_ic;
                                                     fluid_system=fluid_system,
                                                     viscosity=viscosity)
        system = WallBoundarySystem(boundary_ic, boundary_model,
                                    adhesion_coefficient=0.3,
                                    color_value=2)

        @test system isa WallBoundarySystem
        @test system.boundary_model isa BoundaryModelDummyParticles
        @test system.boundary_model.hydrodynamic_mass == boundary_ic.mass
        @test system.boundary_model.density_calculator isa AdamiPressureExtrapolation
        @test system.boundary_model.smoothing_kernel === smoothing_kernel
        @test system.boundary_model.smoothing_length == smoothing_length
        @test system.boundary_model.viscosity == viscosity
        @test system.boundary_model.state_equation == state_equation
        @test system.boundary_model.correction isa KernelCorrection
        @test system.boundary_model.cache.reference_particle_spacing == 0.1
        @test system.adhesion_coefficient == 0.3
        @test system.cache.color == 2

        edac_system = EntropicallyDampedSPHSystem(fluid_ic, smoothing_kernel,
                                                  smoothing_length, 15.0)
        edac_boundary_model = BoundaryModelDummyParticles(boundary_ic;
                                                          fluid_system=edac_system)
        @test edac_boundary_model.state_equation === nothing
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
