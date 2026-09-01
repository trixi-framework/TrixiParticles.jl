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
        opts::Any
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
        integrator = MockDensityReinitIntegrator((; semi), vu_ode, 0.0, nothing)

        TrixiParticles.get_neighborhood_search(system::MockDensityReinitSystem,
                                               neighbor::MockDensityReinitSystem,
                                               semi) = nothing
        TrixiParticles.update_nhs!(neighborhood_search::Nothing,
                                   system::MockDensityReinitSystem,
                                   neighbor::MockDensityReinitSystem,
                                   u_system, u_neighbor, semi) = nothing
        TrixiParticles.derivative_discontinuity!(integrator::MockDensityReinitIntegrator,
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

    @testset verbose=true "only selected systems are affected" begin
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

    @testset verbose=true "semidiscretization index lookup" begin
        coordinates1 = [0.0 0.1
                        0.0 0.0]
        coordinates2 = [0.3 0.4
                        0.0 0.0]
        mass = [1.0, 1.0]
        density = [1000.0, 1000.0]
        smoothing_kernel = WendlandC2Kernel{2}()
        smoothing_length = 0.2
        state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                           exponent=1)

        system1 = WeaklyCompressibleSPHSystem(InitialCondition(; coordinates=coordinates1,
                                                               mass, density);
                                              smoothing_kernel, smoothing_length,
                                              density_calculator=ContinuityDensity(),
                                              state_equation)
        system2 = WeaklyCompressibleSPHSystem(InitialCondition(; coordinates=coordinates2,
                                                               mass, density);
                                              smoothing_kernel, smoothing_length,
                                              density_calculator=ContinuityDensity(),
                                              state_equation)
        semi = Semidiscretization(system1, system2; neighborhood_search=nothing)
        ode = semidiscretize(semi, (0.0, 1.0))

        # `semidiscretize` can replace systems with runtime copies. The callback must
        # therefore resolve its stored index against the integrator semidiscretization
        # instead of closing over the original system object.
        semi_runtime = ode.p.semi
        replacement_systems = deepcopy(semi_runtime.systems)
        semi_replaced = TrixiParticles.Semidiscretization(replacement_systems,
                                                          semi_runtime.ranges_u,
                                                          semi_runtime.ranges_v,
                                                          semi_runtime.neighborhood_search_handler,
                                                          semi_runtime.interaction_matrix,
                                                          semi_runtime.parallelization_backend,
                                                          semi_runtime.update_callback_used,
                                                          semi_runtime.integrate_tlsph)
        vu_ode = deepcopy(ode.u0)
        v_ode, u_ode = vu_ode.x
        TrixiParticles.update_nhs!(semi_replaced, u_ode)

        # Use distinct densities to verify that the callback updates only
        # the selected runtime system at the stored index.
        v1 = TrixiParticles.wrap_v(v_ode, replacement_systems[1], semi_replaced)
        v2 = TrixiParticles.wrap_v(v_ode, replacement_systems[2], semi_replaced)
        v1[end, :] .= -1.0
        v2[end, :] .= -2.0

        callback = DensityReinitializationCallback(system2, semi; interval=1).affect!
        @test callback.system_index == 2

        TrixiParticles.reinitialize_density!(callback, vu_ode, semi_replaced)

        @test all(==(-1.0), v1[end, :])
        @test all(isfinite, v2[end, :])
        @test !all(==(-2.0), v2[end, :])

        empty!(density_reinit_calls)

        system = MockDensityReinitSystem(nothing, :fluid)
        semi = (; systems=(system,))
        other_semi = (; systems=(MockDensityReinitSystem(nothing, :other_fluid),))

        # Constructor validation catches invalid systems before an index is stored.
        @test_throws MethodError DensityReinitializationCallback(; interval=1)
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

        # Runtime validation catches a callback whose stored index points at an
        # incompatible system in the integrator semidiscretization.
        @test_throws ArgumentError TrixiParticles.reinitialize_density!(callback, vu_ode,
                                                                        (;
                                                                         systems=(MockNoDensityReinitSystem(:boundary),)))
    end

    @testset "nonuniform multi-system reinitialization with buffer" begin
        # Reinitialize one buffered system from a second interacting system without changing it.
        spacing = 0.1
        kernel = WendlandC6Kernel{2}()
        smoothing_length = 2spacing
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0, exponent=1)
        initial1 = RectangularShape(spacing, (3, 3), (0.0, 0.0); density=1000.0)
        initial2 = RectangularShape(spacing, (3, 3), (0.35, 0.0); density=1000.0)
        system1 = WeaklyCompressibleSPHSystem(initial1; smoothing_kernel=kernel,
                                              smoothing_length,
                                              density_calculator=ContinuityDensity(),
                                              state_equation, buffer_size=2)
        system2 = WeaklyCompressibleSPHSystem(initial2; smoothing_kernel=kernel,
                                              smoothing_length,
                                              density_calculator=ContinuityDensity(),
                                              state_equation)
        semi = Semidiscretization(system1, system2; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        callback = DensityReinitializationCallback(system1, semi; interval=1,
                                                   reinit_initial_solution=false)
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        semi = ode.p.semi
        system1, system2 = semi.systems
        vu_ode = deepcopy(ode.u0)
        v_ode, u_ode = vu_ode.x
        v1 = TrixiParticles.wrap_v(v_ode, system1, semi)
        v2 = TrixiParticles.wrap_v(v_ode, system2, semi)
        u1 = TrixiParticles.wrap_u(u_ode, system1, semi)
        u2 = TrixiParticles.wrap_u(u_ode, system2, semi)
        active1 = collect(TrixiParticles.eachparticle(system1))
        active2 = collect(TrixiParticles.eachparticle(system2))
        # Use nonuniform densities to distinguish a true summation update from stale values.
        v1[end, active1] .= range(800.0, 1200.0; length=length(active1))
        v2[end, active2] .= range(900.0, 1100.0; length=length(active2))
        density2_before = copy(v2[end, :])

        # Compute the expected Shepard-normalized density independently of the callback.
        TrixiParticles.update_nhs!(semi, u_ode)
        coefficient = zeros(size(v1, 2))
        TrixiParticles.compute_shepard_coeff!(system1,
                                              TrixiParticles.current_coordinates(u1,
                                                                                 system1),
                                              v_ode, u_ode, semi, coefficient)
        summation = zeros(size(v1, 2))
        TrixiParticles.summation_density!(system1, semi, u1, u_ode, summation)
        expected = summation[active1] ./ coefficient[active1]

        # Confirm that the expected density includes particles from the other system.
        cross_contribution = zeros(size(v1, 2))
        coords1 = TrixiParticles.current_coordinates(u1, system1)
        coords2 = TrixiParticles.current_coordinates(u2, system2)
        TrixiParticles.foreach_point_neighbor(system1, system2, coords1, coords2,
                                              semi) do particle, neighbor, pos_diff,
                                                       distance
            volume = TrixiParticles.hydrodynamic_mass(system2, neighbor) /
                     TrixiParticles.current_density(v2, system2, neighbor)
            cross_contribution[particle] += volume *
                                            TrixiParticles.smoothing_kernel(system1,
                                                                            distance,
                                                                            particle)
        end

        # The callback updates active particles only and leaves the other system untouched.
        integrator = MockDensityReinitIntegrator((; semi), vu_ode, 0.05, nothing)
        callback.affect!(integrator)
        inactive1 = setdiff(axes(v1, 2), active1)

        @test v1[end, active1]≈expected rtol=2e-14 atol=2e-14
        @test all(iszero, v1[end, inactive1])
        @test v2[end, :] == density2_before
        @test any(>(0), cross_contribution[active1])
    end

    @testset "simultaneous interacting reinitialization" begin
        # Two callbacks must use the same pre-reinitialization state, independent of callback order.
        spacing = 0.1
        kernel = WendlandC6Kernel{2}()
        state_equation = StateEquationCole(; sound_speed=10.0,
                                           reference_density=1000.0, exponent=1)
        initial1 = RectangularShape(spacing, (2, 2), (0.0, 0.0); density=1000.0)
        initial2 = RectangularShape(spacing, (2, 2), (0.1, 0.0); density=1000.0)
        system1 = WeaklyCompressibleSPHSystem(initial1; smoothing_kernel=kernel,
                                              smoothing_length=2spacing,
                                              density_calculator=ContinuityDensity(),
                                              state_equation)
        system2 = WeaklyCompressibleSPHSystem(initial2; smoothing_kernel=kernel,
                                              smoothing_length=2spacing,
                                              density_calculator=ContinuityDensity(),
                                              state_equation)
        semi = Semidiscretization(system1, system2; neighborhood_search=nothing,
                                  parallelization_backend=SerialBackend())
        ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
        semi = ode.p.semi
        system1, system2 = semi.systems
        vu_ode = deepcopy(ode.u0)
        v_ode, u_ode = vu_ode.x
        v1 = TrixiParticles.wrap_v(v_ode, system1, semi)
        v2 = TrixiParticles.wrap_v(v_ode, system2, semi)
        u1 = TrixiParticles.wrap_u(u_ode, system1, semi)
        u2 = TrixiParticles.wrap_u(u_ode, system2, semi)
        v1[end, :] .= range(800.0, 1200.0; length=size(v1, 2))
        v2[end, :] .= range(1600.0, 2400.0; length=size(v2, 2))
        TrixiParticles.update_nhs!(semi, u_ode)

        # Construct both expected densities before either callback writes to the shared state.
        coefficient1 = zeros(size(v1, 2))
        coefficient2 = zeros(size(v2, 2))
        TrixiParticles.compute_shepard_coeff!(system1,
                                              TrixiParticles.current_coordinates(u1,
                                                                                 system1),
                                              v_ode, u_ode, semi, coefficient1)
        TrixiParticles.compute_shepard_coeff!(system2,
                                              TrixiParticles.current_coordinates(u2,
                                                                                 system2),
                                              v_ode, u_ode, semi, coefficient2)
        expected1 = zeros(size(v1, 2))
        expected2 = zeros(size(v2, 2))
        TrixiParticles.summation_density!(system1, semi, u1, u_ode, expected1)
        TrixiParticles.summation_density!(system2, semi, u2, u_ode, expected2)
        expected1 ./= coefficient1
        expected2 ./= coefficient2

        # Calling one callback triggers the coordinated reinitialization of both systems once.
        density_callback1 = DensityReinitializationCallback(system1, semi; dt=1.0,
                                                            reinit_initial_solution=false)
        density_callback2 = DensityReinitializationCallback(system2, semi; dt=1.0,
                                                            reinit_initial_solution=false)
        callback1 = density_callback1.affect!
        callback2 = density_callback2.affect!
        callbacks = CallbackSet(density_callback1, density_callback2)
        integrator = MockDensityReinitIntegrator((; semi), vu_ode, 0.05,
                                                 (; callback=callbacks))
        callback1(integrator)

        @test v1[end, :]≈expected1 rtol=2e-14 atol=2e-14
        @test v2[end, :]≈expected2 rtol=2e-14 atol=2e-14
        @test callback2.last_t == integrator.t
        density_after = copy(v2[end, :])
        callback2(integrator)
        @test v2[end, :] == density_after
    end
end
