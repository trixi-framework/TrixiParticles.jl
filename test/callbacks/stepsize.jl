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
        @test repr(iisph_callback) == "IISPHTimeStepCallback(require_fixed_step=true)"

        iisph_show_box = """
        ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
        │ IISPHTimeStepCallback                                                                            │
        │ ═════════════════════                                                                            │
        │ require fixed step: …………………………… true                                                             │
        └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""
        @test repr("text/plain", iisph_callback) == iisph_show_box

        @test repr(IISPHTimeStepLimiter()) == "IISPHTimeStepLimiter(require_fixed_step=true)"
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
    end
end
