@testset verbose=true "SortingCallback" begin
    @testset verbose=true "show" begin
        # Default
        @test_throws "either `interval` or `dt` must be set to" SortingCallback()

        callback1 = SortingCallback(interval=11)

        show_compact = "SortingCallback(interval=11, initial_sort=true)"
        @test repr(callback1) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SortingCallback                                                                                  │
        │ ═══════════════                                                                                  │
        │ interval: ……………………………………………………… 11                                                               │
        │ initial_sort: …………………………………………… yes                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback1) == show_box

        callback2 = SortingCallback(dt=1.2)

        show_compact = "SortingCallback(dt=1.2, initial_sort=true)"
        @test repr(callback2) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SortingCallback                                                                                  │
        │ ═══════════════                                                                                  │
        │ dt: ……………………………………………………………………… 1.2                                                              │
        │ initial_sort: …………………………………………… yes                                                              │
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
        TrixiParticles.derivative_discontinuity!(::NamedTuple, _) = nothing

        function TrixiParticles.sort_particles!(system::SortingCallbackMockSystem, v, u,
                                                semi)
            system.sorted_count[] += 1
            return system
        end

        # Test that `initial_sort=false` doesn't trigger sorting.
        callback = SortingCallback(interval=1, initial_sort=false)
        system = SortingCallbackMockSystem(Ref(0))
        semi = (; systems=(system,), ranges_v=(1:2,), ranges_u=(1:2,))
        integrator = (; p=(; semi), u=(; x=(zeros(2), zeros(2))), t=0.0)

        callback.initialize(callback, nothing, 0.0, integrator)

        @test system.sorted_count[] == 0

        # Test that `initial_sort=true` triggers sorting.
        callback = SortingCallback(interval=1, initial_sort=true)
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
        integrator = (; isfinished=false)
        @test callback.condition(nothing, 0.019, integrator) == false
        @test callback.condition(nothing, 0.021, integrator) == true

        # Don't trigger again until the time has advanced by another 0.02.
        callback.affect!.last_t = 0.021
        @test callback.condition(nothing, 0.022, integrator) == false

        # Don't trigger if the integrator is finished.
        integrator = (; isfinished=true)
        @test callback.condition(nothing, 0.05, integrator) == false
    end

    @testset verbose=true "sort_system!" begin
        coordinates = [3.0 1.0 4.0 2.0
                       30.0 10.0 40.0 20.0]
        velocity = [13.0 11.0 14.0 12.0
                    23.0 21.0 24.0 22.0]
        density = [103.0, 101.0, 104.0, 102.0]
        # Non-uniform masses, as they result from e.g. a hydrostatic initial density.
        mass = [1.3, 1.1, 1.4, 1.2]
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
        @test system.mass == mass[perm]
    end

    @testset verbose=true "`sort_system!` permutes caches only updated by the `UpdateCallback`" begin
        coordinates = [3.0 1.0 4.0 2.0
                       30.0 10.0 40.0 20.0]
        velocity = zeros(2, 4)
        density = [103.0, 101.0, 104.0, 102.0]
        mass = ones(4)
        state_equation = Val(:state_equation)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.1

        # Both features are only updated by the `UpdateCallback` with
        # `update_everystage=false`, so their caches are kept between time steps.
        density_diffusion = DensityDiffusionAntuono(delta=0.1, update_everystage=false)
        shifting_technique = ParticleShiftingTechnique(update_everystage=false)

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        system = WeaklyCompressibleSPHSystem(initial_condition;
                                             smoothing_kernel, smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation, density_diffusion,
                                             shifting_technique)

        # With `ContinuityDensity`, the density is stored in the last row of `v`.
        v = vcat(velocity, density')
        u = copy(coordinates)

        # Fill the caches with values that identify the particle they belong to.
        gradient = [1.0 2.0 3.0 4.0
                    5.0 6.0 7.0 8.0]
        delta_v = 10 * gradient
        system.cache.density_diffusion_normalized_density_gradient .= gradient
        system.cache.delta_v .= delta_v

        perm = [2, 4, 1, 3]
        TrixiParticles.sort_system!(system, v, u, perm, nothing)

        @test system.cache.density_diffusion_normalized_density_gradient ==
              gradient[:, perm]
        @test system.cache.delta_v == delta_v[:, perm]
    end

    @testset "Illegal Input" begin
        error_str = "setting both `interval` and `dt` is not supported"
        @test_throws ArgumentError(error_str) SortingCallback(dt=0.1, interval=1)
    end
end
