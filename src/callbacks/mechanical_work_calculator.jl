"""
    MechanicalWorkCalculatorCallback(system::TotalLagrangianSPHSystem, semi; interval=1,
                                     save_interval=interval,
                                     eachparticle=(n_integrated_particles(system) + 1):nparticles(system),
                                     only_compute_force_on_fluid=false)

Callback that accumulates the work done by a set of particles in a
[`TotalLagrangianSPHSystem`](@ref) by integrating the instantaneous power over time.

With the default arguments it tracks the work done by the clamped particles
that follow a [`PrescribedMotion`](@ref). By selecting a different particle set, it can also
be used to measure the work done by the structure on the surrounding fluid.

- **Prescribed/clamped motion work** (default) -- monitor only the clamped particles by
    leaving `eachparticle` at its default range
    `(n_integrated_particles(system) + 1):nparticles(system)`.
- **Fluid load measurement** -- set `eachparticle=eachparticle(system)` together with
    `only_compute_force_on_fluid=true` to accumulate the work that the entire structure
    exerts on the surrounding fluid (useful for drag or lift estimates).

Internally the callback integrates the instantaneous power, i.e. the dot product between
the force exerted by the particle and its prescribed velocity, using an explicit Euler
time integration scheme.

The accumulated value can be retrieved via [`calculated_mechanical_work`](@ref).
The callback also stores the cumulative work history at a configurable frequency.
This makes it possible to query the work done in an arbitrary time window after
the simulation has completed.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `system`: The [`TotalLagrangianSPHSystem`](@ref) whose particles should be monitored.
- `semi`: The [`Semidiscretization`](@ref) that contains `system`.

# Keywords
- `interval=1`: Interval (in number of time steps) at which to compute the instantaneous power.
                It is recommended to keep this at `1` (every time step) or small (≤ 5)
                to limit time integration errors in the integral.
- `save_interval=interval`: Interval (in number of time steps) at which to store the
                cumulative mechanical work. This must be a positive multiple of `interval`.
                The initial and final cumulative values are always stored.
- `eachparticle=(n_integrated_particles(system) + 1):nparticles(system)`: Iterator
                selecting which particles contribute. The default includes all clamped
                particles in the system; pass `eachparticle(system)` to include every particle.
- `only_compute_force_on_fluid=false`: When `true`, only interactions with
                fluid systems are accounted for. Combined with
                `eachparticle=eachparticle(system)`, this accumulates the work that the
                entire structure exerts on the fluid, which is useful for drag or lift
                estimates.

# Examples
```jldoctest; output = false, setup = :(system = TotalLagrangianSPHSystem(RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); smoothing_kernel=WendlandC2Kernel{2}(), smoothing_length=1.0, young_modulus=1.0, poisson_ratio=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend())), filter = r"Stacktrace:.*"s
semi = Semidiscretization(system)
ode = semidiscretize(semi, (0.0, 1.0))

# Note that `Semidiscretization` might create a deep copy of the system,
# which means we have to extract the new system from `semi`.
# When working with GPUs, `semidiscretize` also creates a deep copy of `semi` and another
# copy of the system, so the clean way to get the correct new system is this:
semi_new = ode.p.semi
system_new = semi_new.systems[1]

# Create a mechanical work calculator callback that is called every 2 time steps
mechanical_work_cb = MechanicalWorkCalculatorCallback(system_new, semi_new; interval=2,
                                                      save_interval=10)

# After the simulation, retrieve the accumulated mechanical work
mechanical_work = calculated_mechanical_work(mechanical_work_cb)

# Or retrieve the mechanical work accumulated in a time window
mechanical_work_in_window = calculated_mechanical_work(mechanical_work_cb;
                                                       time_window=(0.2, 0.8))

# output
[ Info: To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`, a deep copy of the system is created inside the `Semidiscretization`. Use `system = semi.systems[i]` to access simulation data.
ERROR: ArgumentError: no mechanical work history has been stored
Stacktrace:
```
"""
struct MechanicalWorkCalculatorCallback{ELTYPE, T, DV, EP}
    interval                    :: Int
    save_interval               :: Int
    t                           :: T # Time of last call
    work                        :: T
    work_history_times          :: Vector{ELTYPE}
    work_history                :: Vector{ELTYPE}
    system_index                :: Int
    dv                          :: DV
    eachparticle                :: EP
    only_compute_force_on_fluid :: Bool
end

# This should dispatch on `TotalLagrangianSPHSystem`, but this name is not yet defined
# due to the include order.
function MechanicalWorkCalculatorCallback(system::AbstractStructureSystem, semi; interval=1,
                                          save_interval=interval,
                                          eachparticle=(n_integrated_particles(system) + 1):nparticles(system),
                                          only_compute_force_on_fluid=false)
    ELTYPE = eltype(system)
    system_index = system_indices(system, semi)

    interval > 0 || throw(ArgumentError("`interval` must be positive"))
    save_interval > 0 || throw(ArgumentError("`save_interval` must be positive"))
    if save_interval % interval != 0
        throw(ArgumentError("`save_interval` must be a positive multiple of `interval`"))
    end

    # Allocate buffer to write accelerations for all particles (including clamped ones)
    dv = allocate(semi.parallelization_backend, ELTYPE,
                  (v_nvariables(system), nparticles(system)))

    # Note that time and work are initialized in
    # `initialize_mechanical_work_calculator_callback`.
    cb = MechanicalWorkCalculatorCallback(interval, save_interval, Ref(zero(ELTYPE)),
                                          Ref(zero(ELTYPE)), ELTYPE[], ELTYPE[],
                                          system_index, dv, eachparticle,
                                          only_compute_force_on_fluid)

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(cb, cb, save_positions=(false, false),
                            initialize=initialize_mechanical_work_calculator_callback)
end

function initialize_mechanical_work_calculator_callback(discrete_callback, u, t, integrator)
    work_callback = discrete_callback.affect!

    # Reset time and mechanical work
    work_callback.t[] = t
    work_callback.work[] = zero(eltype(work_callback.work))

    empty!(work_callback.work_history_times)
    empty!(work_callback.work_history)
    save_mechanical_work!(work_callback, t)
end

"""
    calculated_mechanical_work(cb::DiscreteCallback{<:Any, <:MechanicalWorkCalculatorCallback};
                               time_window=nothing)

Get the accumulated mechanical work from a [`MechanicalWorkCalculatorCallback`](@ref).
Without a `time_window`, this returns the work accumulated from the beginning of the
simulation. Pass `time_window=(t_start, t_end)` to retrieve only the work done in
that time window. The cumulative work at `t_start` and `t_end` is determined by
linear interpolation between the stored history values, which are written by
[`MechanicalWorkCalculatorCallback`](@ref) at its `save_interval`.

# Arguments
- `cb`: The `DiscreteCallback` returned by [`MechanicalWorkCalculatorCallback`](@ref).
- `time_window`: Optional tuple `(t_start, t_end)` selecting a time window.

# Examples
```jldoctest; output = false, setup = :(system = TotalLagrangianSPHSystem(RectangularShape(0.1, (3, 4), (0.1, 0.0), density=1.0); smoothing_kernel=WendlandC2Kernel{2}(), smoothing_length=1.0, young_modulus=1.0, poisson_ratio=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend())), filter = r"Stacktrace:.*"s
# Create a mechanical work calculator callback
mechanical_work_cb = MechanicalWorkCalculatorCallback(system, semi)

# After the simulation, retrieve the accumulated mechanical work
mechanical_work = calculated_mechanical_work(mechanical_work_cb)

# Or retrieve the mechanical work accumulated in a time window
mechanical_work_in_window = calculated_mechanical_work(mechanical_work_cb;
                                                       time_window=(0.2, 0.8))

# output
ERROR: ArgumentError: no mechanical work history has been stored
Stacktrace:
```
"""
function calculated_mechanical_work(cb::DiscreteCallback{<:Any,
                                                         <:MechanicalWorkCalculatorCallback};
                                    time_window=nothing)
    work_callback = cb.affect!

    if isnothing(time_window)
        return work_callback.work[]
    end

    t_start, t_end = time_window
    if t_end < t_start
        throw(ArgumentError("`time_window` must satisfy `t_start <= t_end`"))
    end

    work_start = interpolate_cumulative_mechanical_work(work_callback, t_start)
    work_end = interpolate_cumulative_mechanical_work(work_callback, t_end)

    return work_end - work_start
end

function interpolate_cumulative_mechanical_work(callback::MechanicalWorkCalculatorCallback,
                                                t)
    times = callback.work_history_times
    work_history = callback.work_history

    if isempty(times)
        throw(ArgumentError("no mechanical work history has been stored"))
    end

    if t < first(times) || t > last(times)
        throw(ArgumentError("`time_window` must lie within the stored mechanical work " *
                            "history ($(first(times)), $(last(times)))"))
    end

    index = searchsortedlast(times, t)

    if index > 0 && times[index] == t
        return work_history[index]
    end

    # Linear interpolation between the two nearest stored values.
    t_left = times[index]
    t_right = times[index + 1]
    work_left = work_history[index]
    work_right = work_history[index + 1]

    return work_left + (work_right - work_left) * (t - t_left) / (t_right - t_left)
end

# `condition`
function (callback::MechanicalWorkCalculatorCallback)(u, t, integrator)
    (; interval) = callback

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (callback::MechanicalWorkCalculatorCallback)(integrator)
    # Determine time step size as difference to last time this callback was called
    t = integrator.t
    dt = t - callback.t[]

    # Update time of last call
    callback.t[] = t

    semi = integrator.p.semi
    v_ode, u_ode = integrator.u.x
    work = callback.work

    system = semi.systems[callback.system_index]
    update_mechanical_work_calculator!(work, system, callback.eachparticle,
                                       callback.only_compute_force_on_fluid, callback.dv,
                                       v_ode, u_ode, semi, t, dt)

    if condition_integrator_interval(integrator, callback.save_interval)
        save_mechanical_work!(callback, t)
    end

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return integrator
end


function save_mechanical_work!(callback::MechanicalWorkCalculatorCallback, t)
    push!(callback.work_history_times, t)
    push!(callback.work_history, callback.work[])

    return callback
end

function update_mechanical_work_calculator!(work, system, eachparticle,
                                            only_compute_force_on_fluid, dv,
                                            v_ode, u_ode, semi, t, dt)
    @trixi_timeit timer() "calculate mechanical work" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
        end

        set_zero!(dv)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        foreach_system(semi) do neighbor_system
            if only_compute_force_on_fluid && !(neighbor_system isa AbstractFluidSystem)
                # Not a fluid system, ignore this system
                return
            end

            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)

            interact!(dv, v, u, v_neighbor, u_neighbor,
                      system, neighbor_system, semi,
                      integrate_tlsph=true, # Required when using split integration
                      eachparticle=eachparticle)
        end

        if !only_compute_force_on_fluid
            @threaded semi for particle in eachparticle
                add_acceleration!(dv, system, particle)
                add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t)
            end
        end

        for particle in eachparticle
            velocity = current_velocity(v, system, particle)
            dv_particle = extract_svector(dv, system, particle)

            # The force on the clamped particle is mass times acceleration
            F_particle = system.mass[particle] * dv_particle

            # To obtain mechanical work, we need to integrate the instantaneous power.
            # Instantaneous power is force applied BY the particle times its velocity.
            # The force applied BY the particle is the negative of the force applied ON it.
            work[] += dot(-F_particle, velocity) * dt
        end
    end

    return work
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:MechanicalWorkCalculatorCallback})
    @nospecialize cb # reduce precompilation time

    ELTYPE = eltype(cb.affect!.work)
    print(io, "MechanicalWorkCalculatorCallback{$ELTYPE}(interval=", cb.affect!.interval,
          ", save_interval=", cb.affect!.save_interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:MechanicalWorkCalculatorCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_cb = cb.affect!
        ELTYPE = eltype(update_cb.work)
        setup = [
            "interval" => update_cb.interval,
            "save interval" => update_cb.save_interval
        ]
        summary_box(io, "MechanicalWorkCalculatorCallback{$ELTYPE}", setup)
    end
end
