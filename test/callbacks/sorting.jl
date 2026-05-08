@testset verbose=true "SortingCallback" begin
    @testset verbose=true "show" begin
        # Default
        callback0 = SortingCallback()

        show_compact = "SortingCallback(interval=1)"
        @test repr(callback0) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SortingCallback                                                                                  │
        │ ═══════════════                                                                                  │
        │ interval: ……………………………………………………… 1                                                                │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback0) == show_box

        callback1 = SortingCallback(interval=11)

        show_compact = "SortingCallback(interval=11)"
        @test repr(callback1) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SortingCallback                                                                                  │
        │ ═══════════════                                                                                  │
        │ interval: ……………………………………………………… 11                                                               │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback1) == show_box

        callback2 = SortingCallback(dt=1.2)

        show_compact = "SortingCallback(dt=1.2)"
        @test repr(callback2) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SortingCallback                                                                                  │
        │ ═══════════════                                                                                  │
        │ dt: ……………………………………………………………………… 1.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback2) == show_box
    end

    @testset verbose=true "initial_sort" begin
        struct SortingCallbackMockSystem <: TrixiParticles.AbstractFluidSystem{2}
            sorted_count::Base.RefValue{Int}
        end

        TrixiParticles.v_nvariables(::SortingCallbackMockSystem) = 2
        TrixiParticles.u_nvariables(::SortingCallbackMockSystem) = 2
        TrixiParticles.n_integrated_particles(::SortingCallbackMockSystem) = 1
        TrixiParticles.u_modified!(::NamedTuple, _) = nothing

        function TrixiParticles.sort_particles!(system::SortingCallbackMockSystem, v, u,
                                                semi)
            system.sorted_count[] += 1
            return system
        end

        # Test that `initial_sort=false` doesn't trigger sorting.
        callback = SortingCallback(initial_sort=false)
        system = SortingCallbackMockSystem(Ref(0))
        semi = (; systems=(system,), ranges_v=(1:2,), ranges_u=(1:2,))
        integrator = (; p=semi, u=(; x=(zeros(2), zeros(2))), t=0.0)

        callback.initialize(callback, nothing, 0.0, integrator)

        @test system.sorted_count[] == 0

        # Test that `initial_sort=true` triggers sorting.
        callback = SortingCallback(initial_sort=true)
        callback.initialize(callback, nothing, 0.0, integrator)
        @test system.sorted_count[] == 1
    end

    @testset verbose=true "condition" begin
        TrixiParticles.isfinished(integrator::NamedTuple) = integrator.isfinished

        callback = SortingCallback(interval=3)

        # Don't trigger after only 2 accepted steps.
        integrator = (; t=0.1, stats=(; naccept=2), isfinished=false)
        @test callback.condition(nothing, 0.1, integrator) == false

        # Trigger after 3 accepted steps.
        integrator = (; t=0.1, stats=(; naccept=3), isfinished=false)
        @test callback.condition(nothing, 0.1, integrator) == true

        # Don't trigger if the integrator is finished.
        integrator = (; t=0.1, stats=(; naccept=3), isfinished=true)
        @test callback.condition(nothing, 0.1, integrator) == false

        callback = SortingCallback(dt=0.02)
        @test callback.condition(nothing, 0.019, nothing) == false
        @test callback.condition(nothing, 0.021, nothing) == true

        # Don't trigger again until the time has advanced by another 0.02.
        callback.affect!.last_t = 0.021
        @test callback.condition(nothing, 0.022, nothing) == false
    end

    @testset verbose=true "sort_system!" begin
        coordinates = [3.0 1.0 4.0 2.0
                       30.0 10.0 40.0 20.0]
        velocity = [13.0 11.0 14.0 12.0
                    23.0 21.0 24.0 22.0]
        density = [103.0, 101.0, 104.0, 102.0]
        mass = ones(4)
        state_equation = Val(:state_equation)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.1

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel, smoothing_length,
                                             density_calculator=SummationDensity(),
                                             state_equation)

        v = copy(velocity)
        u = copy(coordinates)
        system.cache.density .= density
        system.pressure .= [203.0, 201.0, 204.0, 202.0]

        perm = [2, 4, 1, 3]
        TrixiParticles.sort_system!(system, v, u, perm, nothing)

        @test u == coordinates[:, perm]
        @test v == velocity[:, perm]
        @test system.cache.density == density[perm]
        @test system.pressure == [203.0, 201.0, 204.0, 202.0][perm]
    end

    @testset "Illegal Input" begin
        error_str = "Setting both interval and dt is not supported!"
        @test_throws ArgumentError(error_str) SortingCallback(dt=0.1, interval=1)
    end
end
