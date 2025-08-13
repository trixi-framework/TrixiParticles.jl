"""
    DensityReinitializationCallback(; interval::Integer=0, dt=0.0)

Callback to reinitialize the density field when using [`ContinuityDensity`](@ref) [Panizzo2007](@cite).

# Keywords
- `interval=0`:              Reinitialize the density every `interval` time steps.
- `dt`:                      Reinitialize the density in regular intervals of `dt` in terms
                             of integration time.
- `reinit_initial_solution`: Reinitialize the initial solution (default=false)
"""
mutable struct DensityReinitializationCallback{I}
    interval::I
    last_t::Float64
    reinit_initial_solution::Bool
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:DensityReinitializationCallback})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!
    print(io, "DensityReinitializationCallback(interval=", callback.interval,
          ", reinit_initial_solution=", callback.reinit_initial_solution, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:DensityReinitializationCallback})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!
        setup = [
            "interval" => callback.interval,
            "reinit_initial_solution" => callback.reinit_initial_solution
        ]
        summary_box(io, "DensityReinitializationCallback", setup)
    end
end

function DensityReinitializationCallback(particle_system; interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=true)
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
                            initialize=(initialize_reinit_cb!))
end

function initialize_reinit_cb!(cb, u, t, integrator)
    initialize_reinit_cb!(cb.affect!, u, t, integrator)
end

function initialize_reinit_cb!(cb::DensityReinitializationCallback, u, t, integrator)
    # Reinitialize initial solution
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
    (; interval) = reinit_callback

    return condition_integrator_interval(integrator, interval, save_final_solution=false)
end

# condition with dt
function (reinit_callback::DensityReinitializationCallback)(u, t, integrator)
    (; interval, last_t) = reinit_callback

    return (t - last_t) > interval
end

# affect!
function (reinit_callback::DensityReinitializationCallback)(integrator)
    vu_ode = integrator.u
    semi = integrator.p

    @trixi_timeit timer() "reinit density" reinit_density!(vu_ode, semi)

    reinit_callback.last_t = integrator.t
end
