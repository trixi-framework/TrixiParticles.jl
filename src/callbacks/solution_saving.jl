"""
    SolutionSavingCallback(; interval::Integer=0, dt=0.0,
                           save_initial_solution=true,
                           save_final_solution=true,
                           output_directory="out", append_timestamp=false, max_coordinates=2^15,
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
- `save_initial_solution=true`: Save the initial solution.
- `save_final_solution=true`:   Save the final solution.
- `output_directory="out"`:     Directory to save the VTK files.
- `append_timestamp=false`:     Append current timestamp to the output directory.
- 'prefix':                     Prefix added to the filename.
- `custom_quantities...`:       Additional user-defined quantities.
- `write_meta_data`:            Write meta data.
- `verbose=false`:              Print to standard IO when a file is written.
- `max_coordinates=2^15`        The coordinates of particles will be clipped if their absolute values exceed this threshold.

# Examples
```julia
# Save every 100 time steps.
saving_callback = SolutionSavingCallback(interval=100)

# Save in intervals of 0.1 in terms of simulation time.
saving_callback = SolutionSavingCallback(dt=0.1)

# Additionally store the norm of the particle velocity for fluid systems as "v_mag".
using LinearAlgebra
function v_mag(v, u, t, system)
    # Ignore for other systems.
    return nothing
end
function v_mag(v, u, t, system::WeaklyCompressibleSPHSystem)
    return [norm(v[1:ndims(system), i]) for i in axes(v, 2)]
end
saving_callback = SolutionSavingCallback(dt=0.1, v_mag=v_mag)
```
"""
struct SolutionSavingCallback{I, CQ}
    interval              :: I
    save_initial_solution :: Bool
    save_final_solution   :: Bool
    write_meta_data       :: Bool
    verbose               :: Bool
    output_directory      :: String
    prefix                :: String
    max_coordinates       :: Float64
    custom_quantities     :: CQ
    latest_saved_iter     :: Vector{Int}
end

function SolutionSavingCallback(; interval::Integer=0, dt=0.0,
                                save_initial_solution=true,
                                save_final_solution=true,
                                output_directory="out", append_timestamp=false,
                                prefix="", verbose=false, write_meta_data=true,
                                max_coordinates=Float64(2^15), custom_quantities...)
    if dt > 0 && interval > 0
        throw(ArgumentError("Setting both interval and dt is not supported!"))
    end

    if dt > 0
        interval = Float64(dt)
    end

    if append_timestamp
        output_directory *= string("_", Dates.format(now(), "YY-mm-ddTHHMMSS"))
    end

    solution_callback = SolutionSavingCallback(interval,
                                               save_initial_solution, save_final_solution,
                                               write_meta_data, verbose, output_directory,
                                               prefix, max_coordinates, custom_quantities,
                                               [-1])

    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(solution_callback, dt,
                                initialize=initialize_save_cb!,
                                save_positions=(false, false),
                                final_affect=save_final_solution)
    else
        # The first one is the condition, the second the affect!
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
    # Save initial solution
    if solution_callback.save_initial_solution
        # Update systems to compute quantities like density and pressure.
        semi = integrator.p
        v_ode, u_ode = u.x
        update_systems_and_nhs(v_ode, u_ode, semi, t)

        # Apply the callback.
        solution_callback(integrator)
    end

    return nothing
end

# condition
function (solution_callback::SolutionSavingCallback)(u, t, integrator)
    (; interval, save_final_solution) = solution_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (((integrator.stats.naccept % interval == 0) &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
            (save_final_solution && isfinished(integrator)))
end

# affect!
function (solution_callback::SolutionSavingCallback)(integrator)
    (; interval, output_directory, custom_quantities, write_meta_data,
    verbose, prefix, latest_saved_iter, max_coordinates) = solution_callback

    vu_ode = integrator.u
    semi = integrator.p
    iter = get_iter(interval, integrator)

    if iter == latest_saved_iter[1]
        # This should only happen at the end of the simulation when using `dt` and the
        # final time is not a multiple of the saving interval.
        @assert isfinished(integrator)

        # Avoid overwriting the previous file
        iter += 1
    end

    latest_saved_iter[1] = iter

    if verbose
        println("Writing solution to $output_directory at t = $(integrator.t)")
    end

    @trixi_timeit timer() "save solution" trixi2vtk(vu_ode, semi, integrator.t; iter=iter,
                                                    output_directory=output_directory,
                                                    prefix=prefix,
                                                    write_meta_data=write_meta_data,
                                                    max_coordinates=max_coordinates,
                                                    custom_quantities...)

    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)

    return nothing
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SolutionSavingCallback})
    @nospecialize cb # reduce precompilation time

    solution_saving = cb.affect!
    print(io, "SolutionSavingCallback(interval=", solution_saving.interval, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:SolutionSavingCallback}})
    @nospecialize cb # reduce precompilation time

    solution_saving = cb.affect!.affect!
    print(io, "SolutionSavingCallback(dt=", solution_saving.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SolutionSavingCallback})
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
            "prefix" => solution_saving.prefix,
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end

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
            "prefix" => solution_saving.prefix,
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end
