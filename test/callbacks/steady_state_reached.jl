@testset verbose=true "SteadyStateReachedCallback" begin
    @testset verbose=true "show" begin
        callback0 = SteadyStateReachedCallback(interval=1)

        show_compact = "SteadyStateReachedCallback(interval=1, abstol=1.0e-8, reltol=1.0e-6)"
        @test repr(callback0) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SteadyStateReachedCallback                                                                       │
        │ ══════════════════════════                                                                       │
        │ interval: ……………………………………………………… 1.0                                                              │
        │ interval size: ………………………………………… 10.0                                                             │
        │ absolute tolerance: …………………………… 1.0e-8                                                           │
        │ relative tolerance: …………………………… 1.0e-6                                                           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback0) == show_box

        callback1 = SteadyStateReachedCallback(interval=11)

        show_compact = "SteadyStateReachedCallback(interval=11, abstol=1.0e-8, reltol=1.0e-6)"
        @test repr(callback1) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SteadyStateReachedCallback                                                                       │
        │ ══════════════════════════                                                                       │
        │ interval: ……………………………………………………… 11.0                                                             │
        │ interval size: ………………………………………… 10.0                                                             │
        │ absolute tolerance: …………………………… 1.0e-8                                                           │
        │ relative tolerance: …………………………… 1.0e-6                                                           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback1) == show_box

        callback2 = SteadyStateReachedCallback(dt=1.2)

        show_compact = "SteadyStateReachedCallback(dt=1.2, abstol=1.0e-8, reltol=1.0e-6)"
        @test repr(callback2) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SteadyStateReachedCallback                                                                       │
        │ ══════════════════════════                                                                       │
        │ dt: ……………………………………………………………………… 1.2                                                              │
        │ interval size: ………………………………………… 10.0                                                             │
        │ absolute tolerance: …………………………… 1.0e-8                                                           │
        │ relative tolerance: …………………………… 1.0e-6                                                           │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback2) == show_box
    end

    @testset "Illegal Input" begin
        error_str = "either `interval` or `dt` must be set to a positive value"
        @test_throws ArgumentError(error_str) SteadyStateReachedCallback()

        error_str = "setting both `interval` and `dt` is not supported"
        @test_throws ArgumentError(error_str) SteadyStateReachedCallback(dt=0.1, interval=1)

        error_str = "`interval_size` must be positive"
        @test_throws ArgumentError(error_str) SteadyStateReachedCallback(interval=1,
                                                                         interval_size=0)

        error_str = "`interval` must be non-negative"
        @test_throws ArgumentError(error_str) SteadyStateReachedCallback(interval=-1)
        @test_throws ArgumentError(error_str) SteadyStateReachedCallback(interval=-2)
    end

    @testset "constructor" begin
        callback = SteadyStateReachedCallback(interval=1, abstol=1, reltol=1)
        steady_state_cb = callback.affect!

        @test steady_state_cb.abstol === 1.0
        @test steady_state_cb.reltol === 1.0
        @test steady_state_cb.previous_ekin == [Inf]

        callback = SteadyStateReachedCallback(interval=Int32(2))
        steady_state_cb = callback.affect!
        @test steady_state_cb.interval === 2

        push!(steady_state_cb.previous_ekin, 1.0)
        semi = (; integrate_tlsph=Ref(true), update_callback_used=Ref(false))
        integrator = (; p=(; semi), opts=(; callback=(; discrete_callbacks=[callback])))
        callback.initialize(callback, nothing, 0.0, integrator)

        @test steady_state_cb.previous_ekin == [Inf]

        callback = SteadyStateReachedCallback(dt=0.1)
        steady_state_cb = callback.affect!.affect!
        push!(steady_state_cb.previous_ekin, 1.0)
        integrator = (; p=(; semi), opts=(; callback=(; discrete_callbacks=[callback])))
        TrixiParticles.initialize_steady_state_callback!(callback.affect!, nothing, 0.0,
                                                         integrator)

        @test steady_state_cb.previous_ekin == [Inf]

        callback = SteadyStateReachedCallback(dt=0.1, interval=0)
        @test callback.affect!.affect!.interval === 0.1
    end

    @testset "condition interval" begin
        struct MockSteadyStateIntegrator
            stats::NamedTuple{(:naccept,), Tuple{Int}}
        end

        TrixiParticles.steady_state_condition!(::TrixiParticles.SteadyStateReachedCallback{Int},
                                               ::MockSteadyStateIntegrator) = true

        function mock_integrator(naccept)
            return MockSteadyStateIntegrator((; naccept))
        end

        callback = SteadyStateReachedCallback(interval=1).affect!
        @test callback(nothing, nothing, mock_integrator(1))

        callback = SteadyStateReachedCallback(interval=10).affect!
        @test !callback(nothing, nothing, mock_integrator(9))
        @test callback(nothing, nothing, mock_integrator(10))
    end
end
