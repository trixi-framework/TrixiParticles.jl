@testset verbose=true "DensityReinitializationCallback" begin
    struct MockDensityReinitSystem
        density_calculator::Any
        name::Symbol
    end

    struct MockNoDensityReinitSystem
        name::Symbol
    end

    struct MockDensityReinitIntegrator
        p::Any
        u::Any
        t::Float64
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
        semi = (; systems=(system,))
        callback = DensityReinitializationCallback(system, semi; interval=10)

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
        semi = (; systems=(system,))
        callback = DensityReinitializationCallback(system, semi; dt=0.1).affect!
        callback.last_t = 0.0

        @test !callback(nothing, 0.099, nothing)
        @test !callback(nothing, 0.1, nothing)
        @test callback(nothing, 0.101, nothing)
    end

    @testset verbose=true "reinit initial solution" begin
        system = MockDensityReinitSystem(nothing, :fluid)
        semi = (; systems=(system,))
        callback = DensityReinitializationCallback(system, semi; interval=1,
                                                   reinit_initial_solution=false).affect!

        @test !callback.reinit_initial_solution
    end

    @testset verbose=true "initialize respects reinit_initial_solution" begin
        empty!(density_reinit_calls)

        system = MockDensityReinitSystem(nothing, :fluid)
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(system,))
        integrator = MockDensityReinitIntegrator((; semi), vu_ode, 0.0)

        TrixiParticles.get_neighborhood_search(system::MockDensityReinitSystem,
                                               neighbor::MockDensityReinitSystem,
                                               semi) = nothing
        TrixiParticles.update_nhs!(neighborhood_search::Nothing,
                                   system::MockDensityReinitSystem,
                                   neighbor::MockDensityReinitSystem,
                                   u_system, u_neighbor, semi) = nothing
        TrixiParticles.u_modified!(integrator::MockDensityReinitIntegrator,
                                   is_modified) = nothing

        callback = DensityReinitializationCallback(system, semi; interval=1,
                                                   reinit_initial_solution=false).affect!
        TrixiParticles.initialize_reinit_cb!(callback, vu_ode, 0.0, integrator)
        @test isempty(density_reinit_calls)

        callback = DensityReinitializationCallback(system, semi; interval=1,
                                                   reinit_initial_solution=true).affect!
        TrixiParticles.initialize_reinit_cb!(callback, vu_ode, 0.0, integrator)
        @test density_reinit_calls == [:fluid]
    end

    @testset verbose=true "selected system" begin
        empty!(density_reinit_calls)

        system1 = MockDensityReinitSystem(nothing, :fluid1)
        system2 = MockDensityReinitSystem(nothing, :fluid2)
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(system1, system2))
        callback = DensityReinitializationCallback(system1, semi; interval=1).affect!

        @test callback.system_index == 1

        TrixiParticles.reinitialize_density!(callback, vu_ode, semi)

        @test density_reinit_calls == [:fluid1]
    end

    @testset verbose=true "selected semidiscretized system" begin
        empty!(density_reinit_calls)

        original_system1 = MockDensityReinitSystem(nothing, :original1)
        original_system2 = MockDensityReinitSystem(nothing, :original2)
        semidiscretized_system1 = MockDensityReinitSystem(nothing, :semidiscretized1)
        semidiscretized_system2 = MockDensityReinitSystem(nothing, :semidiscretized2)
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(original_system1, original_system2))
        semi_replaced = (; systems=(semidiscretized_system1, semidiscretized_system2))

        # Simulate the case where `semidiscretize` creates a copy of the system.
        callback = DensityReinitializationCallback(original_system2, semi;
                                                   interval=1).affect!
        @test callback.system_index == 2

        TrixiParticles.reinitialize_density!(callback, vu_ode, semi_replaced)
        @test density_reinit_calls == [:semidiscretized2]

        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        (; systems=()))
    end

    @testset verbose=true "system validation" begin
        empty!(density_reinit_calls)

        system = MockDensityReinitSystem(nothing, :fluid)
        semi = (; systems=(system,))
        other_semi = (; systems=(MockDensityReinitSystem(nothing, :other_fluid),))

        @test_throws MethodError DensityReinitializationCallback(; interval=1)
        @test_throws MethodError DensityReinitializationCallback(system; interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(MockDensityReinitSystem(SummationDensity(),
                                                                                           :fluid),
                                                                   semi;
                                                                   interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(MockNoDensityReinitSystem(:boundary),
                                                                   semi;
                                                                   interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(system, other_semi;
                                                                   interval=1)

        callback = DensityReinitializationCallback(system, semi; interval=1).affect!
        vu_ode = (; x=(:v_ode, :u_ode))

        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        (;
                                                                         systems=(MockNoDensityReinitSystem(:boundary),)))
    end
end
