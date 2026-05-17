"""
    DensityReinitializationCallback(particle_system; interval::Integer=0, dt=0.0)
    DensityReinitializationCallback(system_index::Integer; interval::Integer=0, dt=0.0)
    DensityReinitializationCallback(system_indices; interval::Integer=0, dt=0.0)
    DensityReinitializationCallback(; system_index::Integer, interval::Integer=0, dt=0.0)
    DensityReinitializationCallback(; system_indices, interval::Integer=0, dt=0.0)

Callback to reinitialize the density field when using [`ContinuityDensity`](@ref) [Panizzo2007](@cite).

Pass `system_index` or `system_indices` to select systems by their position in the
semidiscretization. This is robust when [`semidiscretize`](@ref) replaces systems internally.

# Keywords
- `interval=0`:              Reinitialize the density every `interval` time steps.
- `dt`:                      Reinitialize the density in regular intervals of `dt` in terms
                             of integration time.
- `reinit_initial_solution`: Reinitialize the initial solution (default=false)
"""
mutable struct DensityReinitializationCallback{I, PS, SI}
    interval::I
    particle_system::PS
    system_indices::SI
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
        setup = Pair{String, Any}["interval" => callback.interval,
                                  "reinit_initial_solution" => callback.reinit_initial_solution]
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

    check_density_reinit_system(particle_system)

    last_t = -Inf

    reinit_cb = DensityReinitializationCallback(interval, particle_system, nothing, last_t,
                                                reinit_initial_solution)

    return DiscreteCallback(reinit_cb, reinit_cb, save_positions=(false, false),
                            initialize=(initialize_reinit_cb!))
end

function DensityReinitializationCallback(system_index::Integer; interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=true)
    return reinit_callback_from_indices((system_index,); interval, dt,
                                        reinit_initial_solution)
end

function DensityReinitializationCallback(system_indices::Tuple{Vararg{Integer}};
                                         interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=true)
    return reinit_callback_from_indices(system_indices; interval, dt, reinit_initial_solution)
end

function DensityReinitializationCallback(system_indices::AbstractVector{<:Integer};
                                         interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=true)
    return reinit_callback_from_indices(system_indices; interval, dt, reinit_initial_solution)
end

function DensityReinitializationCallback(system_indices::AbstractRange{<:Integer};
                                         interval::Integer=0, dt=0.0,
                                         reinit_initial_solution=true)
    return reinit_callback_from_indices(system_indices; interval, dt, reinit_initial_solution)
end

function reinit_callback_from_indices(system_indices; interval::Integer=0, dt=0.0,
                                      reinit_initial_solution=true)
    if dt > 0 && interval > 0
        error("Setting both interval and dt is not supported!")
    end

    if dt > 0
        interval = Float64(dt)
    end

    last_t = -Inf
    system_indices_ = normalize_reinit_system_indices(system_indices)

    reinit_cb = DensityReinitializationCallback(interval, nothing, system_indices_,
                                                last_t, reinit_initial_solution)

    return DiscreteCallback(reinit_cb, reinit_cb, save_positions=(false, false),
                            initialize=(initialize_reinit_cb!))
end

function DensityReinitializationCallback(; system_index=nothing, system_indices=nothing,
                                         interval::Integer=0,
                                         dt=0.0, reinit_initial_solution=true)
    if !isnothing(system_index) && !isnothing(system_indices)
        throw(ArgumentError("pass either `system_index` or `system_indices`, not both"))
    end

    if isnothing(system_index) && isnothing(system_indices)
        throw(ArgumentError("pass `system_index` or `system_indices`"))
    end

    indices = isnothing(system_index) ? system_indices : system_index

    return reinit_callback_from_indices(indices; interval, dt, reinit_initial_solution)
end

function initialize_reinit_cb!(cb, u, t, integrator)
    initialize_reinit_cb!(cb.affect!, u, t, integrator)
end

function initialize_reinit_cb!(cb::DensityReinitializationCallback, u, t, integrator)
    semi = integrator.p.semi
    foreach_reinit_system(cb, semi) do system
        check_density_reinit_system(system)
    end

    # Reinitialize initial solution
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
function (reinit_callback::DensityReinitializationCallback{Int})(u, t, integrator)
    (; interval) = reinit_callback

    return condition_integrator_interval(integrator, interval, save_final_solution=false)
end

# condition with dt
function (reinit_callback::DensityReinitializationCallback)(u, t, integrator)
    (; interval, last_t) = reinit_callback

    return (t - last_t) >= interval
end

# affect!
function (reinit_callback::DensityReinitializationCallback)(integrator)
    vu_ode = integrator.u
    semi = integrator.p.semi

    @trixi_timeit timer() "reinit density" reinitialize_density!(reinit_callback, vu_ode,
                                                                 semi)

    reinit_callback.last_t = integrator.t

    # Tell OrdinaryDiffEq that `integrator.u` has been modified
    u_modified!(integrator, true)

    return integrator
end

function reinitialize_density!(reinit_callback::DensityReinitializationCallback,
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

function foreach_reinit_system(f, reinit_callback::DensityReinitializationCallback, semi)
    (; particle_system, system_indices) = reinit_callback

    if !isnothing(system_indices)
        foreach(system_indices) do system_index
            f(current_reinit_system(system_index, semi))
        end

        return nothing
    end

    system_index = findfirst(system -> system === particle_system, semi.systems)

    if isnothing(system_index)
        throw(ArgumentError("system is not in the semidiscretization. This can happen " *
                            "when `semidiscretize` replaced the system. Pass a system " *
                            "index to `DensityReinitializationCallback` to select the " *
                            "system independently of replacement."))
    end

    f(semi.systems[system_index])

    return nothing
end

function current_reinit_system(system_index, semi)
    if system_index > length(semi.systems)
        throw(ArgumentError("system index $system_index is out of bounds for a " *
                            "semidiscretization with $(length(semi.systems)) systems"))
    end

    return semi.systems[system_index]
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

function normalize_reinit_system_indices(system_index::Integer)
    return (normalize_reinit_system_index(system_index),)
end

function normalize_reinit_system_indices(system_indices)
    system_indices_ = Tuple(normalize_reinit_system_index(system_index)
                            for system_index in system_indices)

    if isempty(system_indices_)
        throw(ArgumentError("`system_indices` must not be empty"))
    end

    if !allunique(system_indices_)
        throw(ArgumentError("`system_indices` must be unique"))
    end

    return system_indices_
end

function normalize_reinit_system_index(system_index::Integer)
    if system_index < 1
        throw(ArgumentError("system indices must be positive"))
    end

    return Int(system_index)
end
