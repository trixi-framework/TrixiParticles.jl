struct CheckpointCallback{I}
    interval         :: I
    output_directory :: String
    filename         :: String
    overwrite        :: Bool
end

function CheckpointCallback(; interval::Integer=-1, dt=0.0,
                            output_directory=joinpath("out", "checkpoints"),
                            filename="checkpoint", overwrite=false)
    if dt > 0 && interval !== -1
        throw(ArgumentError("Setting both interval and dt is not supported!"))
    end

    if dt > 0
        interval = Float64(dt)
    end

    checkpoint_callback = CheckpointCallback(interval, output_directory, filename,
                                             overwrite)

    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(checkpoint_callback, dt,
                                initialize=(initialize_checkpoint_callback!),
                                save_positions=(false, false),
                                final_affect=true)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(checkpoint_callback, checkpoint_callback,
                                initialize=(initialize_checkpoint_callback!),
                                save_positions=(false, false))
    end
end

# `initialize`
function initialize_checkpoint_callback!(cb, u, t, integrator)
    # The `CheckpointCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.

    initialize_checkpoint_callback!(cb.affect!, u, t, integrator)
end

function initialize_checkpoint_callback!(cb::CheckpointCallback, u, t, integrator)
    return nothing
end

# `condition`
function (checkpoint_callback::CheckpointCallback)(u, t, integrator)
    (; interval) = checkpoint_callback

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (checkpoint_callback::CheckpointCallback)(integrator)
    (; interval, output_directory, filename, overwrite) = checkpoint_callback

    @trixi_timeit timer() "save checkpoint" begin
        vu_ode = integrator.u
        v_ode, u_ode = vu_ode.x
        semi = integrator.p

        if overwrite
            filename_ = filename
        else
            iter = get_iter(interval, integrator)
            filename_ = (filename * add_underscore_to_optional_postfix(iter))
        end

        save_checkpoint(v_ode, u_ode, semi, integrator.t; output_directory,
                        filename=filename_)
    end

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return nothing
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:CheckpointCallback})
    @nospecialize cb # reduce precompilation time
    print(io, "CheckpointCallback(interval=", cb.affect!.interval, ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:CheckpointCallback}})
    @nospecialize cb # reduce precompilation time
    print(io, "CheckpointCallback(dt=", cb.affect!.affect!.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:CheckpointCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        checkpoint_cb = cb.affect!
        setup = [
            "interval" => checkpoint_cb.interval,
            "output_directory" => checkpoint_cb.output_directory,
            "overwrite file" => checkpoint_cb.overwrite ? "yes" : "no"
        ]
        summary_box(io, "CheckpointCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:CheckpointCallback}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        checkpoint_cb = cb.affect!.affect!
        setup = [
            "dt" => checkpoint_cb.interval,
            "output_directory" => checkpoint_cb.output_directory,
            "overwrite file" => checkpoint_cb.overwrite ? "yes" : "no"
        ]
        summary_box(io, "CheckpointCallback", setup)
    end
end
