"""
    DensityReinitializationCallback(; interval::Integer=0, dt=0.0)
    
Callback to reinitialize the density field when using [ContinuityDensity](@ref).

# Keywords
- `interval=0`: Reinitialize the density every `interval` time steps.
- `dt`:         Reinitialize the density in regular intervals of `dt` in terms
                of integration time.
"""

using DiffEqCallbacks

mutable struct DensityReinitializationCallback{I}
    interval :: I
    last_t   :: I
end

function DensityReinitializationCallback(particle_system; interval::Integer=0, dt=0.0)
    if dt > 0 && interval > 0
        error("Setting both interval and dt is not supported!")
    end

    if dt > 0
        interval = Float64(dt)
    end

    if particle_system.density_calculator isa SummationDensity
        throw(ArgumentError("density reinitialization doesn't provide any advantage for summation density"))
    end

    last_t = 0.0

    reinit_cb = DensityReinitializationCallback(interval, last_t)

    return FunctionCallingCallback(reinit_cb, func_everystep=true, func_start=false)
end

# affect!
function (densreinit_callback::DensityReinitializationCallback)(u, t, integrator)
    semi = integrator.p
    @unpack systems = semi
    @unpack interval = densreinit_callback

    function condition(interval::Integer)
        interval > 0 && (((integrator.stats.naccept % interval == 0) &&
          !(integrator.stats.naccept == 0 && integrator.iter > 0)))
    end
    function condition(interval::Float64)
        interval > 0 && (t - densreinit_callback.last_t) > interval
    end

    if condition(interval)
        densreinit_callback.last_t = t

        v_ode = u.x[1]
        u_ode = u.x[2]

        for (system_index, system) in pairs(systems)
            v = wrap_v(v_ode, system_index, system, semi)
            u = wrap_u(u_ode, system_index, system, semi)
            reinit_density!(v, u, system, system_index, u_ode, v_ode, semi, t)
        end
    end
end
