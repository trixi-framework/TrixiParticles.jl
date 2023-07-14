"""
    DensityReinitializationCallback(; interval::Integer=0, dt=0.0)

Callback to reinitialize the density field when using [ContinuityDensity](@ref).

# Keywords
- `interval=0`:              Reinitialize the density every `interval` time steps.
- `dt`:                      Reinitialize the density in regular intervals of `dt` in terms
                             of integration time.
- `reinit_initial_solution`: Reinitialize the initial solution (default=false)
## References:
- Panizzo, Andrea, Giovanni Cuomo, and Robert A. Dalrymple. "3D-SPH simulation of landslide generated waves."
    In: Coastal Engineering 2006: (In 5 Volumes). 2007. 1503-1515.
    [doi:10.1142/9789812709554_0128](https://doi.org/10.1142/9789812709554_0128)
"""

mutable struct DensityReinitializationCallback{I}
    interval::I
    last_t::Float64
    reinit_initial_solution::Bool
end

function DensityReinitializationCallback(particle_system; interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=false)
    if dt > 0 && interval > 0
        error("Setting both interval and dt is not supported!")
    end

    if dt > 0
        interval = Float64(dt)
    end

    if particle_system.density_calculator isa SummationDensity
        throw(ArgumentError("density reinitialization doesn't provide any advantage for summation density"))
    end

    last_t = -Inf

    reinit_cb = DensityReinitializationCallback(interval, last_t, reinit_initial_solution)

    return DiscreteCallback(reinit_cb, reinit_cb, save_positions=(false, false),
                            initialize=initialize_reinit_cb!)
end

function initialize_reinit_cb!(cb, u, t, integrator)
    initialize_reinit_cb!(cb.affect!, u, t, integrator)
end

function initialize_reinit_cb!(cb::DensityReinitializationCallback, u, t, integrator)
    # Save initial solution
    if cb.reinit_initial_solution
        # Update systems to compute quantities like density and pressure.
        semi = integrator.p
        v_ode, u_ode = u.x
        update_systems_and_nhs(v_ode, u_ode, semi, t)

        # Apply the callback.
        cb(integrator)
    end

    cb.last_t = t

    return nothing
end

# condition with interval
function (reinit_callback::DensityReinitializationCallback{Int})(u, t, integrator)
    @unpack interval = reinit_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && ((integrator.stats.naccept % interval == 0) &&
            !(integrator.stats.naccept == 0 && integrator.iter > 0))
end

# condition with dt
function (reinit_callback::DensityReinitializationCallback)(u, t, integrator)
    @unpack interval, last_t = reinit_callback

    return (t - last_t) > interval
end

# affect!
function (reinit_callback::DensityReinitializationCallback)(integrator)
    vu_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "reinit density" reinit_density!(vu_ode, semi)

    reinit_callback.last_t = integrator.t
end
