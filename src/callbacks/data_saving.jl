struct TableDataSavingCallback{I, ELTYPE, F}
    interval            :: I
    write_file_interval :: Int
    start_at            :: ELTYPE
    save_interval       :: Int
    data                :: Dict{String, Vector{Vector{ELTYPE}}}
    axis_ticks          :: Dict{<:Function, Vector{Float64}}
    output_directory    :: String
    functions           :: F
    empty_vector        :: Vector{ELTYPE}
end

function TableDataSavingCallback(; interval::Integer=0, dt=0.0, save_interval::Integer=-1,
                                 output_directory="out", start_at=0.0,
                                 axis_ticks=Dict{<:Function, Vector{Float64}}(),
                                 write_file_interval::Integer=1, funcs...)
    if isempty(funcs)
        throw(ArgumentError("`funcs` cannot be empty"))
    end

    if dt > 0 && interval > 0
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    if dt > 0
        interval = Float64(dt)
    end

    table_data_cb = TableDataSavingCallback(interval, write_file_interval, start_at,
                                            save_interval,
                                            Dict{String, Vector{Vector{Float64}}}(),
                                            axis_ticks, output_directory, funcs, Float64[])
    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution
        return PeriodicCallback(table_data_cb, dt,
                                save_positions=(false, false), final_affect=true)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(table_data_cb, table_data_cb,
                                save_positions=(false, false))
    end
end

# `condition` with interval
function (cb::TableDataSavingCallback)(u, t, integrator)
    (; interval, start_at) = cb
    return t >= start_at && condition_integrator_interval(integrator, interval)
end

# `affect!`
function (cb::TableDataSavingCallback)(integrator)
    (; empty_vector, functions, data, save_interval, start_at) = cb

    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t

    t >= start_at || return nothing

    # Update systems to compute quantities like density and pressure
    update_systems_and_nhs(v_ode, u_ode, semi, t; update_from_callback=true)

    if !isempty(data) && save_interval == length(data[first(keys(data))])
        for value in values(data)
            popfirst!(value)
        end
    end

    filenames = system_names(semi.systems)

    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        for (key, f) in functions
            result = f(v, u, t, system)
            if result !== nothing
                data_key = string(key) * "_" * filenames[system_indices(system, semi)]

                push!(get!(data, data_key, empty_vector), result)
            end
        end
    end

    if isfinished(integrator) ||
       (cb.write_file_interval > 0 && backup_condition(cb, integrator))
        write_table_data(cb, integrator)
    end

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)
end

function write_table_data(cb::TableDataSavingCallback, integrator)
    (; data, axis_ticks, functions) = cb
    semi = integrator.p

    filenames = system_names(semi.systems)

    foreach_system(semi) do system
        for (function_key, f) in functions
            data_key = string(function_key) * "_" * filenames[system_indices(system, semi)]

            haskey(data, data_key) || continue

            value = stack(values(data[data_key]))

            df = DataFrame(value, [Symbol(data_key * "_$i") for i in 1:size(value, 2)])

            df[!, Symbol(data_key * "_avg")] = [sum(value, dims=2)...] ./ size(value, 2)

            if !isempty(axis_ticks) && haskey(axis_ticks, f)
                df[!, Symbol("x")] = axis_ticks[f]
            end

            # Write the DataFrame to a CSV file
            CSV.write(joinpath(cb.output_directory, data_key * ".csv"), df)
        end
    end
end
