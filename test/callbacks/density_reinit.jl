@testset verbose=true "DensityReinitializationCallback" begin
    struct MockDensityReinitSystem
        density_calculator::Any
        name::Symbol
    end

    struct MockNoDensityReinitSystem
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

    @testset verbose=true "selected system after semidiscretization replacement" begin
        empty!(density_reinit_calls)

        replacement_system1 = MockDensityReinitSystem(Ref(:replacement1), :replacement1)
        replacement_system2 = MockDensityReinitSystem(Ref(:replacement2), :replacement2)

        callback = DensityReinitializationCallback(system_index=1, interval=1).affect!
        vu_ode = (; x=(:v_ode, :u_ode))
        semi_replaced = (; systems=(replacement_system1, replacement_system2))

        TrixiParticles.reinitialize_density!(callback, vu_ode, semi_replaced)

        @test density_reinit_calls == [:replacement1]
    end

    @testset verbose=true "selected systems after semidiscretization replacement" begin
        empty!(density_reinit_calls)

        replacement_system1 = MockDensityReinitSystem(Ref(:replacement1), :replacement1)
        replacement_system2 = MockDensityReinitSystem(Ref(:replacement2), :replacement2)
        replacement_system3 = MockDensityReinitSystem(Ref(:replacement3), :replacement3)

        callback = DensityReinitializationCallback(system_indices=(3, 1),
                                                   interval=1).affect!
        vu_ode = (; x=(:v_ode, :u_ode))
        semi_replaced = (; systems=(replacement_system1, replacement_system2,
                                    replacement_system3))

        TrixiParticles.reinitialize_density!(callback, vu_ode, semi_replaced)

        @test density_reinit_calls == [:replacement3, :replacement1]
    end

    @testset verbose=true "system index validation" begin
        @test_throws ArgumentError DensityReinitializationCallback(system_index=0,
                                                                   interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(system_indices=(),
                                                                   interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(system_indices=(1, 1),
                                                                   interval=1)

        callback = DensityReinitializationCallback(system_index=3, interval=1).affect!
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(MockDensityReinitSystem(nothing, :fluid),))

        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        semi)

        callback = DensityReinitializationCallback(system_index=1, interval=1).affect!
        semi = (; systems=(MockDensityReinitSystem(SummationDensity(), :fluid),))

        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        semi)

        semi = (; systems=(MockNoDensityReinitSystem(:boundary),))

        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        semi)
    end
end
