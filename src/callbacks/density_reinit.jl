mutable struct DensityReinitializationCallbackAffect{I, PS}
    interval::I
    system::PS
    last_t::Float64
    reinit_initial_solution::Bool
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:DensityReinitializationCallbackAffect})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!
    print(io, "DensityReinitializationCallback(interval=", callback.interval,
          ", reinit_initial_solution=", callback.reinit_initial_solution, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:DensityReinitializationCallbackAffect})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!
        setup = Pair{String, Any}["interval" => callback.interval,
                                  "reinit_initial_solution" => callback.reinit_initial_solution]
        summary_box(io, "DensityReinitializationCallback", setup)
    end
end

"""
    DensityReinitializationCallback(system; interval::Integer=0, dt=0.0,
                                    reinit_initial_solution=true)

Callback to reinitialize the density field when using [`ContinuityDensity`](@ref) [Panizzo2007](@cite).

Pass a system to reinitialize its density field. The system must be the system stored
in the semidiscretization used by the integrator. If [`semidiscretize`](@ref) creates
a copy of the system, pass the corresponding system from `ode.p.semi.systems`.

# Keywords
- `interval=0`: Reinitialize the density every `interval` time steps.
- `dt`:         Reinitialize the density in regular intervals of `dt` in terms
                of integration time.
- `reinit_initial_solution`: Reinitialize the initial solution.
"""
function DensityReinitializationCallback(system; interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=true)
    if dt > 0 && interval > 0
        error("Setting both interval and dt is not supported!")
    end

    if dt > 0
        interval = Float64(dt)
    end

    check_density_reinit_system(system)

    last_t = -Inf

    reinit_cb = DensityReinitializationCallbackAffect(interval, system, last_t,
                                                      reinit_initial_solution)

    return DiscreteCallback(reinit_cb, reinit_cb, save_positions=(false, false),
                            initialize=(initialize_reinit_cb!))
end

function initialize_reinit_cb!(cb, u, t, integrator)
    initialize_reinit_cb!(cb.affect!, u, t, integrator)
end

function initialize_reinit_cb!(cb::DensityReinitializationCallbackAffect, u, t, integrator)
    semi = integrator.p.semi
    foreach_reinit_system(cb, semi) do system
        check_density_reinit_system(system)
    end

    if cb.reinit_initial_solution
        # Update systems to compute quantities like density and pressure.
        v_ode, u_ode = u.x
        update_systems_and_nhs(v_ode, u_ode, semi, t)

        # Apply the callback.
        cb(integrator)
    end

    cb.last_t = t

    return nothing
end

# condition with interval
function (reinit_callback::DensityReinitializationCallbackAffect{<:Integer})(u, t,
                                                                             integrator)
    (; interval) = reinit_callback

    return condition_integrator_interval(integrator, interval, save_final_solution=false)
end

# condition with dt
function (reinit_callback::DensityReinitializationCallbackAffect)(u, t, integrator)
    (; interval, last_t) = reinit_callback

    return (t - last_t) >= interval
end

# affect!
function (reinit_callback::DensityReinitializationCallbackAffect)(integrator)
    vu_ode = integrator.u
    semi = integrator.p.semi

    @trixi_timeit timer() "reinit density" reinitialize_density!(reinit_callback, vu_ode,
                                                                 semi)

    reinit_callback.last_t = integrator.t

    # Tell OrdinaryDiffEq that `integrator.u` has been modified
    u_modified!(integrator, true)

    return integrator
end

function reinitialize_density!(reinit_callback::DensityReinitializationCallbackAffect,
                               vu_ode, semi)
    v_ode, u_ode = vu_ode.x

    foreach_reinit_system(reinit_callback, semi) do particle_system
        check_density_reinit_system(particle_system)
        v = wrap_v(v_ode, particle_system, semi)
        u = wrap_u(u_ode, particle_system, semi)

        reinit_density!(particle_system, v, u, v_ode, u_ode, semi)
    end

    return reinit_callback
end

function foreach_reinit_system(f, reinit_callback::DensityReinitializationCallbackAffect,
                               semi)
    system_index = findfirst(s -> s === reinit_callback.system, semi.systems)

    if isnothing(system_index)
        throw(ArgumentError("system is not in the semidiscretization. This can " *
                            "happen when `semidiscretize` creates a copy of the " *
                            "system. Create the callback with the corresponding " *
                            "system from `ode.p.semi.systems`."))
    end

    f(semi.systems[system_index])

    return nothing
end

function check_density_reinit_system(particle_system)
    if !hasproperty(particle_system, :density_calculator)
        throw(ArgumentError("density reinitialization requires a system with a density calculator"))
    end

    if particle_system.density_calculator isa SummationDensity
        throw(ArgumentError("density reinitialization doesn't provide any advantage for summation density"))
    end

    return particle_system
end
