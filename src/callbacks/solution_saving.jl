@doc raw"""
    SolutionSavingCallback(; interval::Integer=0, dt=0.0, save_times=Float64[],
                           save_initial_solution=true, save_final_solution=true,
                           output_directory="out", append_timestamp=false, prefix="",
                           verbose=false, overwrite=false, max_coordinates=2^15,
                           custom_quantities...)


Callback to save the current numerical solution in VTK format.
Use at most one of `interval`, `dt`, and `save_times`: pass `interval` to save
every `interval` accepted time steps, `dt` to save in intervals of `dt` in terms
of integration time by adding additional `tstops` (note that this may change the
solution), or `save_times` to save at specific times. The initial and final
solution can be added independently with `save_initial_solution` and
`save_final_solution`.

Additional user-defined quantities can be saved by passing keyword arguments.
A custom quantity can be an array or a function. Functions are called as
`(system, dv_ode, du_ode, v_ode, u_ode, semi, t)` when that method exists,
otherwise as `(system, data, t)`. In the latter case, `data` is a named tuple
with fields depending on the system type. To ignore a custom quantity for a
specific system, return `nothing`.

# Keywords
- `interval=0`:                 Save the solution every `interval` accepted time steps.
                                A value of `0` disables step-interval saves.
- `dt`:                         Save the solution in regular intervals of `dt` in terms
                                of integration time by adding additional `tstops`
                                (note that this may change the solution).
- `save_times=Float64[]`:       Specific times at which to save a solution. These times
                                are mutually exclusive with `interval` and `dt`.
- `save_initial_solution=true`: Save the initial solution. Setting this to `false`
                                does not suppress an initial time explicitly listed in
                                `save_times`.
- `save_final_solution=true`:   Save the final solution. Setting this to `false`
                                does not suppress a final time explicitly listed in
                                `save_times`.
- `overwrite=false`:            If `true`, previously written VTK files are overwritten
                                instead of creating a new file set at each save interval.
                                In this case, filenames receive the postfix `_current`.
                                This option is useful for memory efficiency in large
                                simulations where only the final results matter. It
                                provides a rolling checkpoint at each save interval.
                                If `false` (default), files are not overwritten and an
                                iteration postfix is appended for each interval.
- `output_directory="out"`:     Directory to save the VTK files.
- `append_timestamp=false`:     Append current timestamp to the output directory.
- `prefix=""`:                  Prefix added to the filename.
- `verbose=false`:              Print to standard IO when a file is written.
- `max_coordinates=2^15`:       The coordinates of particles will be clipped if their
                                absolute values exceed this threshold.
- `custom_quantities...`:       Additional custom quantities to include in the VTK output.
                                Check the available data for each system with
                                `available_data(system)`.
                                See [Custom Quantities](@ref custom_quantities) for a list
                                of pre-defined custom quantities that can be used here.

# Examples
```jldoctest; output = false, filter = [r"output directory:.*", r"\s+│"]
# Save every 100 time steps
saving_callback = SolutionSavingCallback(interval=100)

# Save in intervals of 0.1 in terms of simulation time
saving_callback = SolutionSavingCallback(dt=0.1)

# Additionally store the kinetic energy of each system as "my_custom_quantity"
saving_callback = SolutionSavingCallback(dt=0.1, my_custom_quantity=kinetic_energy)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SolutionSavingCallback                                                                           │
│ ══════════════════════                                                                           │
│ dt: ……………………………………………………………………… 0.1                                                              │
│ custom quantities: ……………………………… [:my_custom_quantity => TrixiParticles.kinetic_energy]           │
│ save initial solution: …………………… yes                                                              │
│ save final solution: ………………………… yes                                                              │
│ overwrite solution: …………………………… no                                                               │
│ output directory: ………………………………… *path ignored with filter regex above*                           │
│ prefix: ……………………………………………………………                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
mutable struct SolutionSavingCallback{I, CQ}
    interval              :: I
    save_times            :: Vector{Float64}
    save_initial_solution :: Bool
    save_final_solution   :: Bool
    verbose               :: Bool
    output_directory      :: String
    prefix                :: String
    overwrite             :: Bool
    max_coordinates       :: Float64
    custom_quantities     :: CQ
    latest_saved_iter     :: Int
    git_hash              :: Ref{String}
end

function SolutionSavingCallback(; interval::Integer=0, dt=0.0,
                                save_times=Float64[],
                                save_initial_solution=true, save_final_solution=true,
                                output_directory="out", append_timestamp=false,
                                prefix="", verbose=false, overwrite=false,
                                max_coordinates=Float64(2^15),
                                custom_quantities...)
    save_times = sort!(collect(Float64.(save_times)))

    if (dt > 0 && interval > 0) || (length(save_times) > 0 && (dt > 0 || interval > 0))
        throw(ArgumentError("Setting multiple save times for the same solution " *
                            "callback is not possible. Use either `dt`, `interval` or `save_times`."))
    end

    if dt > 0
        interval = Float64(dt)
    end

    if append_timestamp
        output_directory *= string("_", Dates.format(now(), "YY-mm-ddTHHMMSS"))
    end

    solution_callback = SolutionSavingCallback(interval, save_times,
                                               save_initial_solution, save_final_solution,
                                               verbose, output_directory, prefix, overwrite,
                                               max_coordinates, custom_quantities,
                                               -1, Ref("UnknownVersion"))

    if length(save_times) > 0
        return PresetTimeCallback(save_times, solution_callback;
                                  initialize=(initialize_save_times_cb!),
                                  save_positions=(false, false))
    elseif dt > 0
        # Add a `tstop` every `dt`, and save the final solution
        return PeriodicCallback(solution_callback, dt,
                                initialize=(initialize_save_cb!),
                                save_positions=(false, false),
                                final_affect=save_final_solution)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(solution_callback, solution_callback,
                                save_positions=(false, false),
                                initialize=(initialize_save_cb!))
    end
end

function initialize_save_cb!(cb, u, t, integrator)
    # The `SolutionSavingCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initialize_save_cb!(cb.affect!, u, t, integrator)
end

function initialize_save_cb!(solution_callback::SolutionSavingCallback, u, t, integrator)
    initialize_save_cb!(solution_callback, u, t, integrator,
                        solution_callback.save_initial_solution)
end

function initialize_save_cb!(solution_callback::SolutionSavingCallback, u, t, integrator,
                             save_initial_solution)
    semi = integrator.p
    set_callbacks_used!(semi, integrator)

    # Reset `latest_saved_iter`
    solution_callback.latest_saved_iter = -1
    solution_callback.git_hash[] = compute_git_hash()

    write_meta_data(solution_callback, integrator)

    # Save initial solution
    if save_initial_solution
        solution_callback(integrator)
    end

    return nothing
end

function initialize_save_times_cb!(cb, u, t, integrator)
    solution_callback = cb.affect!
    add_final_save_time!(cb.condition.tstops, solution_callback, integrator)

    # `PresetTimeCallback` calls `affect!` after this initializer when `t` is in
    # `tstops`. Avoid writing the initial solution twice when it is also a save time.
    save_initial_solution = solution_callback.save_initial_solution &&
                            !insorted(t, cb.condition.tstops)
    initialize_save_cb!(solution_callback, u, t, integrator, save_initial_solution)
end

function add_final_save_time!(save_times, solution_callback, integrator)
    if solution_callback.save_final_solution
        push!(save_times, last(integrator.sol.prob.tspan))
        sort!(unique!(save_times))
    end

    return nothing
end

# `condition`
function (solution_callback::SolutionSavingCallback)(u, t, integrator)
    (; interval, save_final_solution) = solution_callback

    return condition_integrator_interval(integrator, interval; save_final_solution)
end

# `affect!`
function (solution_callback::SolutionSavingCallback)(integrator)
    (; interval, output_directory, custom_quantities, git_hash, verbose, overwrite,
     prefix, max_coordinates) = solution_callback

    @trixi_timeit timer() "save solution" begin
        @trixi_timeit timer() "update dvdu" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() dvdu_ode = get_dvdu(integrator)
        end

        vu_ode = integrator.u
        semi = integrator.p
        iter = get_iter(interval, integrator)
        append_collection = solution_callback.latest_saved_iter >= 0

        if iter == solution_callback.latest_saved_iter
            # This should only happen at the end of the simulation when using `dt` and the
            # final time is not a multiple of the saving interval.
            @assert isfinished(integrator)

            # Avoid overwriting the previous file
            iter += 1
        end

        if verbose
            println("Writing solution to $output_directory at t = $(integrator.t)")
        end

        _trixi2vtk(dvdu_ode, vu_ode, semi, integrator.t;
                   iter, overwrite, append_collection, output_directory, prefix,
                   git_hash=git_hash[], max_coordinates, custom_quantities...)

        solution_callback.latest_saved_iter = iter
    end

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return nothing
end

# The type of the `DiscreteCallback` returned by the constructor is
# `DiscreteCallback{typeof(condition), typeof(affect!), typeof(initialize), typeof(finalize)}`.
#
# When `interval` is used, this is
# `DiscreteCallback{<:SolutionSavingCallback,
#                   <:SolutionSavingCallback,
#                   typeof(TrixiParticles.initialize_save_cb!)}`.
#
# When `dt` is used, this is
# `DiscreteCallback{DiffEqCallbacks.var"#99#103"{...},
#                   DiffEqCallbacks.PeriodicCallbackAffect{<:SolutionSavingCallback},
#                   DiffEqCallbacks.var"#100#104"{...}}`.
#
# When `save_times` is used, this is
# `DiscreteCallback{<:DiffEqCallbacks.PresetTimeFunction,
#                   <:SolutionSavingCallback,
#                   <:DiffEqCallbacks.PresetTimeFunction}`.
#
# So we can unambiguously dispatch on
# - `DiscreteCallback{<:SolutionSavingCallback, <:SolutionSavingCallback}`,
# - `DiscreteCallback{<:Any, <:PeriodicCallbackAffect{<:SolutionSavingCallback}}`,
# - `DiscreteCallback{<:Any, <:SolutionSavingCallback}`.
function (finalize::SolutionSavingCallback)(c, u, t, integrator)
    return nothing
end

# With `interval`
function Base.show(io::IO,
                   cb::DiscreteCallback{<:SolutionSavingCallback, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    solution_saving = cb.affect!
    print(io, "SolutionSavingCallback(interval=", solution_saving.interval, ")")
end

# With `dt`
function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SolutionSavingCallback}})
    @nospecialize cb # reduce precompilation time

    solution_saving = cb.affect!.affect!
    print(io, "SolutionSavingCallback(dt=", solution_saving.interval, ")")
end

# With `save_times`
function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    solution_saving = cb.affect!
    print(io, "SolutionSavingCallback(save_times=", solution_saving.save_times, ")")
end

# With `interval`
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:SolutionSavingCallback, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        solution_saving = cb.affect!
        cq = collect(solution_saving.custom_quantities)

        setup = [
            "interval" => solution_saving.interval,
            "custom quantities" => isempty(cq) ? nothing : cq,
            "save initial solution" => solution_saving.save_initial_solution ?
                                       "yes" : "no",
            "save final solution" => solution_saving.save_final_solution ? "yes" :
                                     "no",
            "overwrite solution" => solution_saving.overwrite ? "yes" : "no",
            "output directory" => abspath(solution_saving.output_directory),
            "prefix" => solution_saving.prefix
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end

# With `dt`
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SolutionSavingCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        solution_saving = cb.affect!.affect!
        cq = collect(solution_saving.custom_quantities)

        setup = [
            "dt" => solution_saving.interval,
            "custom quantities" => isempty(cq) ? nothing : cq,
            "save initial solution" => solution_saving.save_initial_solution ?
                                       "yes" : "no",
            "save final solution" => solution_saving.save_final_solution ? "yes" :
                                     "no",
            "overwrite solution" => solution_saving.overwrite ? "yes" : "no",
            "output directory" => abspath(solution_saving.output_directory),
            "prefix" => solution_saving.prefix
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end

# With `save_times`. See comments above.
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        solution_saving = cb.affect!
        cq = collect(solution_saving.custom_quantities)

        setup = [
            "save_times" => solution_saving.save_times,
            "custom quantities" => isempty(cq) ? nothing : cq,
            "save initial solution" => solution_saving.save_initial_solution ?
                                       "yes" : "no",
            "save final solution" => solution_saving.save_final_solution ? "yes" :
                                     "no",
            "overwrite solution" => solution_saving.overwrite ? "yes" : "no",
            "output directory" => abspath(solution_saving.output_directory),
            "prefix" => solution_saving.prefix
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end
