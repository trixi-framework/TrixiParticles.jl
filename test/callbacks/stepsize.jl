@testset verbose=true "StepsizeCallback" begin
    @testset verbose=true "show" begin
        callback = StepsizeCallback(cfl=1.2)

        show_compact = "StepsizeCallback(is_constant=true, cfl_number=1.2)"
        @test repr(callback) == show_compact

        show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ StepsizeCallback                                                                                 │
        │ ════════════════                                                                                 │
        │ is constant: ……………………………………………… true                                                             │
        │ CFL number: ………………………………………………… 1.2                                                              │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", callback) == show_box

        iisph_callback = IISPHTimeStepCallback()
        @test repr(iisph_callback) ==
              "IISPHTimeStepCallback(require_fixed_step=true, warm_start_pressure=true, project_at_step_end=false, pressure_projection=stage)"

        iisph_show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ IISPHTimeStepCallback                                                                            │
        │ ═════════════════════                                                                            │
        │ require fixed step: …………………………… true                                                             │
        │ warm-start pressure: ………………………… true                                                             │
        │ project at step end: ………………………… false                                                            │
        │ pressure projection: ………………………… stage                                                            │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", iisph_callback) == iisph_show_box

        @test repr(IISPHTimeStepLimiter()) ==
              "IISPHTimeStepLimiter(require_fixed_step=true, warm_start_pressure=true)"

        adaptive_callback = IISPHPressureAdaptiveTimeStepCallback(; min_dt=0.01,
                                                                  max_dt=0.1)
        @test repr(adaptive_callback) ==
              "IISPHPressureAdaptiveTimeStepCallback(min_dt=0.01, max_dt=0.1, target_iterations=(3, 8), growth_factor=1.25, shrink_factor=0.8, cap_shrink_factor=0.5)"
    end

    @trixi_testset "IISPHTimeStepCallback and limiter" begin
        coordinates = zeros(2, 1)
        velocity = zeros(2, 1)
        mass = [1.0]
        density = [1000.0]
        reference_density = 1000.0
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 0.1

        initial_condition = InitialCondition(; coordinates, velocity, mass, density)
        system = ImplicitIncompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 reference_density,
                                                 time_step=0.1)
        semi = Semidiscretization(system; neighborhood_search=nothing)

        @test TrixiParticles.iisph_projection_dt(semi) == 0.1
        @test iisph_pressure_iteration_stats(semi) ==
              (last_iterations=0, total_iterations=0, max_iterations=0,
               solve_count=0, average_iterations=0.0, last_solve_time=0.0,
               total_solve_time=0.0, max_solve_time=0.0, average_solve_time=0.0)

        callback = IISPHTimeStepCallback()
        integrator = (; p=(; semi), dt=0.05, opts=(; adaptive=false))
        TrixiParticles.initialize_iisph_time_step_callback(callback, nothing, 0.0,
                                                           integrator)
        @test TrixiParticles.iisph_projection_dt(semi) == 0.05
        @test !TrixiParticles.iisph_step_end_projection_enabled(semi)
        @test !TrixiParticles.iisph_pressure_projection_only_enabled(semi)

        projection_callback = IISPHTimeStepCallback(project_at_step_end=true)
        TrixiParticles.initialize_iisph_time_step_callback(projection_callback, nothing,
                                                           0.0, integrator)
        @test TrixiParticles.iisph_step_end_projection_enabled(semi)
        @test !TrixiParticles.iisph_strang_projection_enabled(semi)

        strang_callback = IISPHTimeStepCallback(pressure_projection=:strang)
        TrixiParticles.initialize_iisph_time_step_callback(strang_callback, nothing,
                                                           0.0, integrator)
        @test TrixiParticles.iisph_step_end_projection_enabled(semi)
        @test TrixiParticles.iisph_strang_projection_enabled(semi)
        @test !TrixiParticles.iisph_pressure_projection_only_enabled(semi)

        TrixiParticles.set_iisph_pressure_projection_only!(semi, true)
        @test TrixiParticles.iisph_pressure_projection_only_enabled(semi)
        TrixiParticles.set_iisph_pressure_projection_only!(semi, false)
        @test !TrixiParticles.iisph_pressure_projection_only_enabled(semi)

        @test_throws ArgumentError IISPHTimeStepCallback(pressure_projection=:invalid)

        limiter = IISPHTimeStepLimiter()
        integrator_limiter = (; dt=0.025, opts=(; adaptive=false))
        limiter(nothing, integrator_limiter, (; semi), 0.0)
        @test TrixiParticles.iisph_projection_dt(semi) == 0.025

        adaptive_integrator = (; p=(; semi), dt=0.0125, opts=(; adaptive=true))
        @test_throws ArgumentError TrixiParticles.initialize_iisph_time_step_callback(callback,
                                                                                      nothing,
                                                                                      0.0,
                                                                                      adaptive_integrator)

        system.pressure .= 8.0
        TrixiParticles.disable_iisph_pressure_warm_start!(semi)
        TrixiParticles.reset_iisph_pressure_initialization!(semi)
        TrixiParticles.initialize_iisph_pressure!(semi)
        TrixiParticles.initialize_iisph_pressure!(semi)
        @test system.pressure[1] == 2.0

        system.pressure .= 8.0
        TrixiParticles.enable_iisph_pressure_warm_start!(semi)
        TrixiParticles.reset_iisph_pressure_initialization!(semi)
        TrixiParticles.initialize_iisph_pressure!(semi)
        TrixiParticles.initialize_iisph_pressure!(semi)
        @test system.pressure[1] == 4.0

        TrixiParticles.reset_iisph_pressure_initialization!(semi)
        TrixiParticles.initialize_iisph_pressure!(semi)
        @test system.pressure[1] == 2.0

        pressure = [1.0]
        a_ii = [0.0]
        sum_term = [0.0]
        density_error = [42.0]
        TrixiParticles.pressure_update(system, pressure, reference_density, a_ii,
                                       sum_term, system.omega, density_error, semi)
        @test pressure[1] == 0.0
        @test density_error[1] == 0.0

        TrixiParticles.record_iisph_pressure_iterations!(semi, 2, 0.25)
        TrixiParticles.record_iisph_pressure_iterations!(semi, 5, 0.75)
        @test iisph_pressure_iteration_stats(semi) ==
              (last_iterations=5, total_iterations=7, max_iterations=5,
               solve_count=2, average_iterations=3.5, last_solve_time=0.75,
               total_solve_time=1.0, max_solve_time=0.75, average_solve_time=0.5)
        @test iisph_pressure_step_stats(semi) ==
              (total_iterations=7, max_iterations=5, solve_count=2,
               average_iterations=3.5, total_solve_time=1.0, max_solve_time=0.75,
               average_solve_time=0.5)

        reset_iisph_pressure_step_stats!(semi)
        @test iisph_pressure_step_stats(semi) ==
              (total_iterations=0, max_iterations=0, solve_count=0,
               average_iterations=0.0, total_solve_time=0.0, max_solve_time=0.0,
               average_solve_time=0.0)
        @test iisph_pressure_iteration_stats(semi) ==
              (last_iterations=5, total_iterations=7, max_iterations=5,
               solve_count=2, average_iterations=3.5, last_solve_time=0.75,
               total_solve_time=1.0, max_solve_time=0.75, average_solve_time=0.5)

        reset_iisph_pressure_iteration_stats!(semi)
        @test iisph_pressure_iteration_stats(semi) ==
              (last_iterations=0, total_iterations=0, max_iterations=0,
               solve_count=0, average_iterations=0.0, last_solve_time=0.0,
               total_solve_time=0.0, max_solve_time=0.0, average_solve_time=0.0)
        @test iisph_pressure_step_stats(semi) ==
              (total_iterations=0, max_iterations=0, solve_count=0,
               average_iterations=0.0, total_solve_time=0.0, max_solve_time=0.0,
               average_solve_time=0.0)

        low_work = (; average_iterations=2.0, max_iterations=3, solve_count=1)
        in_band = (; average_iterations=5.0, max_iterations=6, solve_count=1)
        high_work = (; average_iterations=32.0, max_iterations=40, solve_count=1)
        capped = (; average_iterations=5.0, max_iterations=20, solve_count=1)
        @test TrixiParticles.iisph_pressure_adaptive_dt(0.1, low_work;
                                                        target_min_iterations=3,
                                                        target_max_iterations=8) >
              0.1
        @test TrixiParticles.iisph_pressure_adaptive_dt(0.1, in_band;
                                                        target_min_iterations=3,
                                                        target_max_iterations=8) == 0.1
        @test TrixiParticles.iisph_pressure_adaptive_dt(0.1, high_work;
                                                        target_min_iterations=3,
                                                        target_max_iterations=8) < 0.1
        @test TrixiParticles.iisph_pressure_adaptive_dt(0.1, capped;
                                                        pressure_max_iterations=20) == 0.05
    end
end
