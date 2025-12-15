# Minimal objects used to mock an OrdinaryDiffEq integrator. They only store the fields
# touched by the stepsize callback so we can assert on them in the unit tests.

mutable struct StepsizeTestOpts
    adaptive::Bool
    dtmax::Float64
end

mutable struct StepsizeTestIntegrator
    u
    p
    opts::StepsizeTestOpts
    dtcache::Float64
    proposed_dt::Float64
    u_modified_flag::Bool
end

function StepsizeTestIntegrator(v_ode, u_ode, semi; adaptive=false)
    state = (; x=(v_ode, u_ode))
    opts = StepsizeTestOpts(adaptive, 0.0)
    return StepsizeTestIntegrator(state, semi, opts, 0.0, 0.0, true)
end

function TrixiParticles.set_proposed_dt!(integrator::StepsizeTestIntegrator, dt)
    integrator.proposed_dt = dt
    return integrator
end

function TrixiParticles.u_modified!(integrator::StepsizeTestIntegrator, flag)
    integrator.u_modified_flag = flag
    return integrator
end

function TrixiParticles.calculate_dt(v_ode, u_ode, cfl_number,
                                     semi::NamedTuple{(:factor, :offset)})
    return semi.factor * cfl_number + semi.offset
end

@testset verbose=true "StepsizeCallback" begin
    @testset verbose=true "show" begin
        # Check both compact and pretty-printed repr outputs
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
    end

    @testset verbose=true "condition" begin
        # Ensure the discrete callback only triggers for non-constant stepsize on
        # non-adaptive integrators
        constant_callback = StepsizeCallback{true, Float64}(0.5)
        variable_callback = StepsizeCallback{false, Float64}(0.5)

        integrator_nonadaptive = (; opts=(; adaptive=false))
        integrator_adaptive = (; opts=(; adaptive=true))

        @test !constant_callback(nothing, 0.0, integrator_nonadaptive)
        @test variable_callback(nothing, 0.0, integrator_nonadaptive)
        @test !variable_callback(nothing, 0.0, integrator_adaptive)
    end

    @testset verbose=true "affect!" begin
        # Verify that `affect!` computes the dt via `calculate_dt` and forwards it
        # to `set_proposed_dt!`, `dtmax`, and `dtcache`
        semi = (; factor=2.0, offset=0.05)
        cfl_number = 0.3
        expected_dt = semi.factor * cfl_number + semi.offset

        integrator = StepsizeTestIntegrator([1.0, 2.0], [3.0, 4.0], semi; adaptive=false)
        stepsize = StepsizeCallback{true, Float64}(cfl_number)

        returned = stepsize(integrator)

        @test returned === stepsize
        @test integrator.proposed_dt == expected_dt
        @test integrator.opts.dtmax == expected_dt
        @test integrator.dtcache == expected_dt
        @test integrator.u_modified_flag == false
    end

    @testset verbose=true "initialize" begin
        # The initialize hook should delegate to the callback and thus set the dt once
        semi = (; factor=1.5, offset=0.1)
        cfl_number = 0.25
        expected_dt = semi.factor * cfl_number + semi.offset

        integrator = StepsizeTestIntegrator([0.0], [0.0], semi; adaptive=false)
        callback = StepsizeCallback(cfl=cfl_number)

        TrixiParticles.initialize_stepsize_callback(callback, nothing, 0.0, integrator)

        @test integrator.proposed_dt == expected_dt
        @test integrator.opts.dtmax == expected_dt
        @test integrator.dtcache == expected_dt
    end
end
