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
              "IISPHTimeStepCallback(require_fixed_step=true, warm_start_pressure=true)"

        iisph_show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ IISPHTimeStepCallback                                                                            │
        │ ═════════════════════                                                                            │
        │ require fixed step: …………………………… true                                                             │
        │ warm-start pressure: ………………………… true                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", iisph_callback) == iisph_show_box

        @test repr(IISPHTimeStepLimiter()) ==
              "IISPHTimeStepLimiter(require_fixed_step=true, warm_start_pressure=true)"
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

        callback = IISPHTimeStepCallback()
        integrator = (; p=(; semi), dt=0.05, opts=(; adaptive=false))
        TrixiParticles.initialize_iisph_time_step_callback(callback, nothing, 0.0,
                                                           integrator)
        @test TrixiParticles.iisph_projection_dt(semi) == 0.05

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
    end
end
