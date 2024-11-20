@doc raw"""
    SolutionSavingCallback(; interval::Integer=0, dt=0.0, save_times=Array{Float64, 1}([]),
                           save_initial_solution=true, save_final_solution=true,
                           output_directory="out", append_timestamp=false, prefix="",
                           verbose=false, write_meta_data=true, max_coordinates=2^15,
                           custom_quantities...)


Callback to save the current numerical solution in VTK format in regular intervals.
Either pass `interval` to save every `interval` time steps,
or pass `dt` to save in intervals of `dt` in terms of integration time by adding
additional `tstops` (note that this may change the solution).

Additional user-defined quantities can be saved by passing functions
as keyword arguments, which map `(v, u, t, system)` to an `Array` where
the columns represent the particles in the same order as in `u`.
To ignore a custom quantity for a specific system, return `nothing`.

# Keywords
- `interval=0`:                 Save the solution every `interval` time steps.
- `dt`:                         Save the solution in regular intervals of `dt` in terms
                                of integration time by adding additional `tstops`
                                (note that this may change the solution).
- `save_times=[]`               List of times at which to save a solution.
- `save_initial_solution=true`: Save the initial solution.
- `save_final_solution=true`:   Save the final solution.
- `output_directory="out"`:     Directory to save the VTK files.
- `append_timestamp=false`:     Append current timestamp to the output directory.
- 'prefix=""':                  Prefix added to the filename.
- `custom_quantities...`:       Additional user-defined quantities.
- `write_meta_data=true`:       Write meta data.
- `verbose=false`:              Print to standard IO when a file is written.
- `max_coordinates=2^15`:       The coordinates of particles will be clipped if their
                                absolute values exceed this threshold.
- `custom_quantities...`:   Additional custom quantities to include in the VTK output.
                            Each custom quantity must be a function of `(v, u, t, system)`,
                            which will be called for every system, where `v` and `u` are the
                            wrapped solution arrays for the corresponding system and `t` is
                            the current simulation time. Note that working with these `v`
                            and `u` arrays requires undocumented internal functions of
                            TrixiParticles. See [Custom Quantities](@ref custom_quantities)
                            for a list of pre-defined custom quantities that can be used here.

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
│ output directory: ………………………………… *path ignored with filter regex above*                           │
│ prefix: ……………………………………………………………                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
mutable struct SolutionSavingCallback{I, CQ}
    interval              :: I
    save_times            :: Array{Float64, 1}
    save_initial_solution :: Bool
    save_final_solution   :: Bool
    write_meta_data       :: Bool
    verbose               :: Bool
    output_directory      :: String
    prefix                :: String
    max_coordinates       :: Float64
    custom_quantities     :: CQ
    latest_saved_iter     :: Int
    git_hash              :: Ref{String}
end

function SolutionSavingCallback(; interval::Integer=0, dt=0.0,
                                save_times=Array{Float64, 1}([]),
                                save_initial_solution=true, save_final_solution=true,
                                output_directory="out", append_timestamp=false,
                                prefix="", verbose=false, write_meta_data=true,
                                max_coordinates=Float64(2^15), custom_quantities...)
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
                                               write_meta_data, verbose, output_directory,
                                               prefix, max_coordinates, custom_quantities,
                                               -1, Ref("UnknownVersion"))

    if length(save_times) > 0
        # See the large comment below for an explanation why we use `finalize` here.
        # When support for Julia 1.9 is dropped, the `finalize` argument can be removed.
        return PresetTimeCallback(save_times, solution_callback, finalize=solution_callback)
    elseif dt > 0
        # Add a `tstop` every `dt`, and save the final solution
        return PeriodicCallback(solution_callback, dt,
                                initialize=initialize_save_cb!,
                                save_positions=(false, false),
                                final_affect=save_final_solution)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(solution_callback, solution_callback,
                                save_positions=(false, false),
                                initialize=initialize_save_cb!)
    end
end

function initialize_save_cb!(cb, u, t, integrator)
    # The `SolutionSavingCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initialize_save_cb!(cb.affect!, u, t, integrator)
end

function initialize_save_cb!(solution_callback::SolutionSavingCallback, u, t, integrator)
    # Reset `latest_saved_iter`
    solution_callback.latest_saved_iter = -1
    solution_callback.git_hash[] = compute_git_hash()

    # Save initial solution
    if solution_callback.save_initial_solution
        # Update systems to compute quantities like density and pressure
        semi = integrator.p
        v_ode, u_ode = u.x
        update_systems_and_nhs(v_ode, u_ode, semi, t; update_from_callback=true)

        # Apply the callback
        solution_callback(integrator)
    end

    return nothing
end

# `condition`
function (solution_callback::SolutionSavingCallback)(u, t, integrator)
    (; interval, save_final_solution) = solution_callback

    return condition_integrator_interval(integrator, interval,
                                         save_final_solution=save_final_solution)
end

# `affect!`
function (solution_callback::SolutionSavingCallback)(integrator)
    (; interval, output_directory, custom_quantities, write_meta_data, git_hash,
    verbose, prefix, latest_saved_iter, max_coordinates) = solution_callback

    vu_ode = integrator.u
    semi = integrator.p
    iter = get_iter(interval, integrator)

    if iter == latest_saved_iter
        # This should only happen at the end of the simulation when using `dt` and the
        # final time is not a multiple of the saving interval.
        @assert isfinished(integrator)

        # Avoid overwriting the previous file
        iter += 1
    end

    latest_saved_iter = iter

    if verbose
        println("Writing solution to $output_directory at t = $(integrator.t)")
    end

    @trixi_timeit timer() "save solution" trixi2vtk(vu_ode, semi, integrator.t;
                                                    iter, output_directory, prefix,
                                                    write_meta_data, git_hash=git_hash[],
                                                    max_coordinates, custom_quantities...)

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
#                   typeof(TrixiParticles.initialize_save_cb!),
#                   typeof(SciMLBase.FINALIZE_DEFAULT)}`.
#
# When `dt` is used, this is
# `DiscreteCallback{DiffEqCallbacks.var"#99#103"{...},
#                   DiffEqCallbacks.PeriodicCallbackAffect{<:SolutionSavingCallback},
#                   DiffEqCallbacks.var"#100#104"{...}
#                   typeof(SciMLBase.FINALIZE_DEFAULT)}`.
#
# When `save_times` is used, this is
# `DiscreteCallback{DiffEqCallbacks.var"#115#117"{...},
#                   <:SolutionSavingCallback,
#                   DiffEqCallbacks.var"#116#118"{...},
#                   typeof(SciMLBase.FINALIZE_DEFAULT)}}`.
#
# So we can unambiguously dispatch on
# - `DiscreteCallback{<:SolutionSavingCallback, <:SolutionSavingCallback}`,
# - `DiscreteCallback{<:Any, <:PeriodicCallbackAffect{<:SolutionSavingCallback}}`,
# - `DiscreteCallback{<:Any, <:SolutionSavingCallback}`.
#
# WORKAROUND FOR JULIA 1.9:
# When `save_times` is used, the `affect!` is also wrapped in an anonymous function:
# `DiscreteCallback{DiffEqCallbacks.var"#110#113"{...},
#                   DiffEqCallbacks.var"#111#114"{<:SolutionSavingCallback},
#                   DiffEqCallbacks.var"#116#118"{...},
#                   typeof(SciMLBase.FINALIZE_DEFAULT)}}`.
#
# To dispatch here, we set `finalize` to the callback itself, so that the fourth parameter
# becomes `<:SolutionSavingCallback`. This is only used in Julia 1.9. 1.10 and later uses
# a newer version of DiffEqCallbacks.jl that does not have this issue.
#
# To use the callback as `finalize`, we have to define the following function.
# `finalize` is set to `FINALIZE_DEFAULT` by default in the `PresetTimeCallback`,
# which is a function that just returns `nothing`.
# We define the `SolutionSavingCallback` to do the same when called with these arguments.
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

# With `save_times`, also working in Julia 1.9.
# When support for Julia 1.9 is dropped, this can be changed to
# `DiscreteCallback{<:Any, <:SolutionSavingCallback}`, and the `finalize` argument
# in the constructor of `SolutionSavingCallback` can be removed.
function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:Any, <:Any, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    # This has to be changed to `cb.affect!` when support for Julia 1.9 is dropped
    # and finalize is removed from the constructor of `SolutionSavingCallback`.
    solution_saving = cb.finalize
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
            "output directory" => abspath(solution_saving.output_directory),
            "prefix" => solution_saving.prefix
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end

# With `save_times`. See comments above.
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:Any, <:Any, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        # This has to be changed to `cb.affect!` when support for Julia 1.9 is dropped
        # and finalize is removed from the constructor of `SolutionSavingCallback`.
        solution_saving = cb.finalize
        cq = collect(solution_saving.custom_quantities)

        setup = [
            "save_times" => solution_saving.save_times,
            "custom quantities" => isempty(cq) ? nothing : cq,
            "save initial solution" => solution_saving.save_initial_solution ?
                                       "yes" : "no",
            "save final solution" => solution_saving.save_final_solution ? "yes" :
                                     "no",
            "output directory" => abspath(solution_saving.output_directory),
            "prefix" => solution_saving.prefix
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end
