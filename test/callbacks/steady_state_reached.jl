@testset verbose=true "SteadyStateReachedCallback" begin
    @testset verbose=true "every `dt`" begin
        steady_state_reached = SteadyStateReachedCallback(; dt=0.002, interval_size=10)

        trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "pipe_flow_2d.jl"),
                      extra_callback=steady_state_reached, tspan=(0.0, 1.5))

        @test sol.t[end] < 1.0
    end

    @testset verbose=true "every `interval`" begin
        steady_state_reached = SteadyStateReachedCallback(; interval=1, interval_size=10,
                                                          abstol=1.0e-5, reltol=1.0e-4)
        trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "pipe_flow_2d.jl"),
                      extra_callback=steady_state_reached, tspan=(0.0, 1.5))

        @test sol.t[end] < 1.0
    end

    @testset verbose=true "show" begin
        # Default
        callback0 = SteadyStateReachedCallback()

        show_compact = "SteadyStateReachedCallback(abstol=1.0e-8, reltol=1.0e-6)"
        @test repr(callback0) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SteadyStateReachedCallback                                                                       │
        │ ══════════════════════════                                                                       │
        │ absolute tolerance: …………………………… 1.0e-8                                                           │
        │ relative tolerance: …………………………… 1.0e-6                                                           │
        │ interval: ……………………………………………………… 0.0                                                              │
        │ interval size: ………………………………………… 10.0                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback0) == show_box

        callback1 = SteadyStateReachedCallback(interval=11)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SteadyStateReachedCallback                                                                       │
        │ ══════════════════════════                                                                       │
        │ absolute tolerance: …………………………… 1.0e-8                                                           │
        │ relative tolerance: …………………………… 1.0e-6                                                           │
        │ interval: ……………………………………………………… 11.0                                                             │
        │ interval size: ………………………………………… 10.0                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback1) == show_box

        callback2 = SteadyStateReachedCallback(dt=1.2)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ SteadyStateReachedCallback                                                                       │
        │ ══════════════════════════                                                                       │
        │ absolute tolerance: …………………………… 1.0e-8                                                           │
        │ relative tolerance: …………………………… 1.0e-6                                                           │
        │ interval: ……………………………………………………… 1.2                                                              │
        │ interval_size: ………………………………………… 10.0                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback2) == show_box
    end

    @testset "Illegal Input" begin
        error_str = "setting both `interval` and `dt` is not supported"
        @test_throws ArgumentError(error_str) SteadyStateReachedCallback(dt=0.1, interval=1)
    end
end
