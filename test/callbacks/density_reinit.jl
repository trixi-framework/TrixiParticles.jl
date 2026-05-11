@testset verbose=true "DensityReinitializationCallback" begin
    struct MockDensityReinitSystem
        density_calculator
        name::Symbol
    end

    density_reinit_calls = Symbol[]

    TrixiParticles.wrap_v(v_ode, system::MockDensityReinitSystem, semi) = (:v, system.name)
    TrixiParticles.wrap_u(u_ode, system::MockDensityReinitSystem, semi) = (:u, system.name)

    function TrixiParticles.reinit_density!(system::MockDensityReinitSystem, v, u,
                                            v_ode, u_ode, semi)
        push!(density_reinit_calls, system.name)
        return system
    end

    @testset verbose=true "show" begin
        system = MockDensityReinitSystem(nothing, :fluid)
        callback = DensityReinitializationCallback(system; interval=10)

        show_compact = "DensityReinitializationCallback(interval=10, reinit_initial_solution=true)"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ DensityReinitializationCallback                                                                  │
        │ ═══════════════════════════════                                                                  │
        │ interval: ……………………………………………………… 10                                                               │
        │ reinit_initial_solution: ……………… true                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box
    end

    @testset verbose=true "dt condition" begin
        system = MockDensityReinitSystem(nothing, :fluid)
        callback = DensityReinitializationCallback(system; dt=0.1).affect!
        callback.last_t = 0.0

        @test !callback(nothing, 0.099, nothing)
        @test callback(nothing, 0.1, nothing)
    end

    @testset verbose=true "selected system" begin
        empty!(density_reinit_calls)

        system1 = MockDensityReinitSystem(nothing, :fluid1)
        system2 = MockDensityReinitSystem(nothing, :fluid2)
        callback = DensityReinitializationCallback(system1; interval=1).affect!
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(system1, system2))

        TrixiParticles.reinitialize_density!(callback, vu_ode, semi)

        @test density_reinit_calls == [:fluid1]
    end
end
