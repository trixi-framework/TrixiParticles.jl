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

    @testset verbose=true "reinit initial solution" begin
        system = MockDensityReinitSystem(nothing, :fluid)
        callback = DensityReinitializationCallback(system; interval=1,
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

        callback = DensityReinitializationCallback(system; interval=1,
                                                   reinit_initial_solution=false).affect!
        TrixiParticles.initialize_reinit_cb!(callback, vu_ode, 0.0, integrator)
        @test isempty(density_reinit_calls)

        callback = DensityReinitializationCallback(system; interval=1,
                                                   reinit_initial_solution=true).affect!
        TrixiParticles.initialize_reinit_cb!(callback, vu_ode, 0.0, integrator)
        @test density_reinit_calls == [:fluid]
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

    @testset verbose=true "selected semidiscretized system" begin
        empty!(density_reinit_calls)

        # Simulate the case where `semidiscretize` creates a copy of the system.
        original_system = MockDensityReinitSystem(nothing, :original)
        semidiscretized_system = MockDensityReinitSystem(nothing, :semidiscretized)
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(semidiscretized_system,))

        callback = DensityReinitializationCallback(semidiscretized_system;
                                                   interval=1).affect!
        TrixiParticles.reinitialize_density!(callback, vu_ode, semi)
        @test density_reinit_calls == [:semidiscretized]

        callback = DensityReinitializationCallback(original_system; interval=1).affect!
        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        semi)
    end

    @testset verbose=true "system validation" begin
        empty!(density_reinit_calls)

        system = MockDensityReinitSystem(nothing, :fluid)
        @test_throws MethodError DensityReinitializationCallback(; interval=1)
        @test_throws MethodError DensityReinitializationCallback(system, system; interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(MockDensityReinitSystem(SummationDensity(),
                                                                                           :fluid);
                                                                   interval=1)
        @test_throws ArgumentError DensityReinitializationCallback(MockNoDensityReinitSystem(:boundary);
                                                                   interval=1)

        callback = DensityReinitializationCallback(system; interval=1).affect!
        vu_ode = (; x=(:v_ode, :u_ode))
        semi = (; systems=(MockDensityReinitSystem(nothing, :other_fluid),))

        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        semi)
    end
end
