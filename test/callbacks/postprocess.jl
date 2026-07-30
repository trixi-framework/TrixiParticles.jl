@testset verbose=true "PostprocessCallback" begin
    @testset verbose=true "errors" begin
        error_str1 = "`funcs` cannot be empty"
        @test_throws ArgumentError(error_str1) PostprocessCallback(interval=10,
                                                                   write_file_interval=0)

        error_str2 = "setting both `interval` and `dt` is not supported"
        @test_throws ArgumentError(error_str2) PostprocessCallback(interval=10,
                                                                   write_file_interval=0,
                                                                   dt=0.1,
                                                                   another_function=(system,
                                                                                     v_ode,
                                                                                     u_ode,
                                                                                     semi,
                                                                                     t) -> 1)
    end

    @testset verbose=true "show" begin
        function example_function(system, v_ode, u_ode, semi, t)
            return 0
        end

        callback = PostprocessCallback(another_function=(system, v_ode, u_ode, semi,
                                                         t) -> 1;
                                       interval=10,
                                       example_function, write_file_interval=0)

        show_compact = "PostprocessCallback(interval=10, functions=[another_function, example_function])"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 10                                                               │
        │ write file: ………………………………………………… no                                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… another_function                                                 │
        │ function2: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; dt=0.1, example_function, write_file_interval=0)

        show_compact = "PostprocessCallback(dt=0.1, functions=[example_function])"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ dt: ……………………………………………………………………… 0.1                                                              │
        │ write file: ………………………………………………… no                                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; dt=0.1, example_function, write_file_interval=3)
        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ dt: ……………………………………………………………………… 0.1                                                              │
        │ write file: ………………………………………………… every 3 * dt                                                     │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; interval=23, example_function,
                                       write_file_interval=4)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 23                                                               │
        │ write file: ………………………………………………… every 4 * interval                                               │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; interval=23, example_function)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ interval: ……………………………………………………… 23                                                               │
        │ write file: ………………………………………………… always                                                           │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        callback = PostprocessCallback(; dt=0.2, example_function)

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ PostprocessCallback                                                                              │
        │ ═══════════════════                                                                              │
        │ dt: ……………………………………………………………………… 0.2                                                              │
        │ write file: ………………………………………………… always                                                           │
        │ exclude boundary: ………………………………… yes                                                              │
        │ filename: ……………………………………………………… values                                                           │
        │ output directory: ………………………………… out                                                              │
        │ append timestamp: ………………………………… no                                                               │
        │ write json file: …………………………………… yes                                                              │
        │ write csv file: ……………………………………… yes                                                              │
        │ function1: …………………………………………………… example_function                                                 │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box
    end

    @testset verbose=true "backup_condition" begin
        function example_function(system, data, t)
            return t
        end

        # Helper to create a mock integrator for backup_condition tests
        function make_integrator(; naccept, t, tspan)
            return (; stats=(; naccept), t, sol=(; prob=(; tspan)))
        end

        # Integer interval
        @testset "interval-based (Int)" begin
            cb_interval = PostprocessCallback(; interval=10, write_file_interval=3,
                                              example_function)
            pp = cb_interval.affect!

            # No write at naccept=0 (initialization)
            integrator = make_integrator(naccept=0, t=0.0, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            # Write at 1st, 2nd trigger: no write (write_file_interval=3)
            integrator = make_integrator(naccept=10, t=0.1, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            integrator = make_integrator(naccept=20, t=0.2, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            # Write at 3rd trigger (3 * interval = 30)
            integrator = make_integrator(naccept=30, t=0.3, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == true

            integrator = make_integrator(naccept=40, t=0.4, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            # Write every write_file_interval triggers, regardless of t_0
            # (naccept resets after restart, so counting is relative to new start)
            integrator = make_integrator(naccept=30, t=10.3, tspan=(10.0, 11.0))
            @test TrixiParticles.backup_condition(pp, integrator) == true
        end

        # Float (dt-based)
        @testset "dt-based (Float)" begin
            cb_dt = PostprocessCallback(; dt=0.1, write_file_interval=3, example_function)
            pp = cb_dt.affect!.affect!

            # No write at naccept=0 (initialization)
            integrator = make_integrator(naccept=0, t=0.0, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            # 1st and 2nd trigger: no write
            integrator = make_integrator(naccept=1, t=0.1, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            integrator = make_integrator(naccept=2, t=0.2, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            # 3rd trigger: write
            integrator = make_integrator(naccept=3, t=0.3, tspan=(0.0, 1.0))
            @test TrixiParticles.backup_condition(pp, integrator) == true

            # After restart at t_0=1.0: counting restarts from tspan[1]
            integrator = make_integrator(naccept=0, t=1.0, tspan=(1.0, 2.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            integrator = make_integrator(naccept=1, t=1.1, tspan=(1.0, 2.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            integrator = make_integrator(naccept=2, t=1.2, tspan=(1.0, 2.0))
            @test TrixiParticles.backup_condition(pp, integrator) == false

            # 3rd trigger after restart: write (not at t=1.2 as the old code would give)
            integrator = make_integrator(naccept=3, t=1.3, tspan=(1.0, 2.0))
            @test TrixiParticles.backup_condition(pp, integrator) == true
        end
    end
end
