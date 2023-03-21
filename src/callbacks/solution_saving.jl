"""
    SolutionSavingCallback(; interval::Integer=0, dt=0.0,
                           save_initial_solution=true,
                           save_final_solution=true,
                           output_directory="out", append_timestamp=false,
                           custom_quantities...)
Callback to save the current numerical solution in VTK format in regular intervals.
Either pass `interval` to save every `interval` time steps,
or pass `dt` to save in intervals of `dt` in terms of integration time by adding
additional `tstops` (note that this may change the solution).
Additional user-defined quantities can be saved by passing functions
as keyword arguments, which map `(v, u, t, container)` to an `Array` where
the columns represent the particles in the same order as in `u`.
To ignore a custom quantity for a specific container, return `nothing`.
# Keywords
- `interval=0`:                 Save the solution every `interval` time steps.
- `dt`:                         Save the solution in regular intervals of `dt` in terms
                                of integration time by adding additional `tstops`
                                (note that this may change the solution).
- `save_initial_solution=true`: Save the initial solution.
- `save_final_solution=true`:   Save the final solution.
- `output_directory="out"`:     Directory to save the VTK files.
- `append_timestamp=false`:     Append current timestamp to the output directory.
- `custom_quantities...`:       Additional user-defined quantities.
# Examples
```julia
# Save every 100 time steps.
saving_callback = SolutionSavingCallback(interval=100)
# Save in intervals of 0.1 in terms of simulation time.
saving_callback = SolutionSavingCallback(dt=0.1)
# Additionally store the norm of the particle velocity for fluid containers as "v_mag".
using LinearAlgebra
function v_mag(v, u, t, container)
    # Ignore for other containers.
    return nothing
end
function v_mag(v, u, t, container::FluidParticleContainer)
    return [norm(v[1:ndims(container), i]) for i in axes(v, 2)]
end
saving_callback = SolutionSavingCallback(dt=0.1, v_mag=v_mag)
```
"""
struct SolutionSavingCallback{I, CQ}
    interval::I
    save_initial_solution::Bool
    save_final_solution::Bool
    output_directory::String
    custom_quantities::CQ
end

function SolutionSavingCallback(; interval::Integer=0, dt=0.0,
                                save_initial_solution=true,
                                save_final_solution=true,
                                output_directory="out", append_timestamp=false,
                                custom_quantities...)
    if dt > 0
        interval = Float64(dt)
    end

    if append_timestamp
        output_directory *= string("_", Dates.format(now(), "YY-mm-ddTHHMMSS"))
    end

    solution_callback = SolutionSavingCallback(interval,
                                               save_initial_solution, save_final_solution,
                                               output_directory, custom_quantities)

    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(solution_callback, dt,
                                initialize=initialize_save_cb!,
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
        # Update containers to compute quantities like density and pressure.
        semi = integrator.p
        v_ode, u_ode = u.x
        update_containers_and_nhs(v_ode, u_ode, semi, t)

        # Apply the callback.
        solution_callback(integrator)
    end

    return nothing
end

# condition
function (solution_callback::SolutionSavingCallback)(u, t, integrator)
    @unpack interval, save_final_solution = solution_callback

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
    @unpack interval, output_directory, custom_quantities = solution_callback

    vu_ode = integrator.u
    semi = integrator.p
    iter = get_iter(interval, integrator)

    @trixi_timeit timer() "save solution" trixi2vtk(vu_ode, semi, integrator.t; iter=iter,
                                                    output_directory=output_directory,
                                                    custom_quantities...)

    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)

    return nothing
end

get_iter(::Integer, integrator) = integrator.stats.naccept
function get_iter(dt::AbstractFloat, integrator)
    # Basically `(t - tspan[1]) / dt` as `Int`.
    Int(div(integrator.t - first(integrator.sol.prob.tspan), dt, RoundNearest))
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
            "output directory" => abspath(normpath(solution_saving.output_directory)),
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
            "output directory" => abspath(normpath(solution_saving.output_directory)),
        ]
        summary_box(io, "SolutionSavingCallback", setup)
    end
end
