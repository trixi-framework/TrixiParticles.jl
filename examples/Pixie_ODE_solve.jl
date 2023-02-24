using Pixie
using OrdinaryDiffEq

function pixie_ode_solve(ode, method, tspan; abstol=1e-6, reltol=1e-3, dtmax=1e-2,
                         save_everystep=false, callbacks=nothing, save_when_error=true,
                         save=false, alive_interval=100, save_interval=0.02)
    summary_callback = SummaryCallback()
    alive_callback = AliveCallback(alive_interval=alive_interval)

    saved_values = nothing
    if callbacks === nothing
        if save
            if tspan[2] < save_interval
                error("Invalid save_interval for provided time span. (save_interval > tspan)")
            end
            saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:save_interval:tspan[2],
                                                                   index=(v, u, t, container) -> Pixie.eachparticle(container))
            callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)
        else
            callbacks = CallbackSet(summary_callback, alive_callback)
        end
    end

    sol = nothing
    try
        sol = solve(ode, method, abstol=abstol, reltol=reltol, dtmax=dtmax,
                    save_everystep=save_everystep, callback=callbacks)
    catch e
        if save_when_error
            pixie2vtk(sol, prefix="failed")
        end
        @error "Simulation failed:" exception=(e, catch_backtrace())
    end

    # Print the timer summary
    summary_callback()

    if save
        pixie2vtk(saved_values)
    end

    return sol
end
