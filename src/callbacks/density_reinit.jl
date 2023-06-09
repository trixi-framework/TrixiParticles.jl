"""
DensityReinitializationCallback(; interval::Integer=0, dt=0.0)

Callback to reinitialize the density field when using continuity density.

# Keywords
- `interval=0`:                 Reinitialize the density every `interval` time steps.
- `dt`:                         Reinitialize the density in regular intervals of `dt` in terms
                                of integration time
"""

using DiffEqCallbacks

mutable struct DensityReinitializationCallback{I}
    interval :: I
    last_t   :: I
end

function DensityReinitializationCallback(particle_container; interval::Integer=0, dt=0.0)
    if dt > 0 && interval > 0
        error("Setting both interval and dt is not supported!")
    end

    if dt > 0
        interval = Float64(dt)
    end

    if particle_container.density_calculator isa SummationDensity
        error("Density reinitialization doesn't provide any advantage for summation density")
    end

    last_t = 0.0

    reinit_cb = DensityReinitializationCallback(interval, last_t)

    return FunctionCallingCallback(reinit_cb, func_everystep=true, func_start=false)
end

# affect!
function (densreinit_callback::DensityReinitializationCallback)(u, t, integrator)
    semi = integrator.p
    @unpack particle_containers = semi
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

        for (container_index, container) in pairs(particle_containers)
            v = wrap_v(v_ode, container_index, container, semi)
            u = wrap_u(u_ode, container_index, container, semi)
            perform_corrections_after_timestep!(v, u, container, container_index, u_ode,
                                                semi, t)
        end
    end
end
