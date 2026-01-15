"""
    PostprocessCallback(; interval::Integer=0, dt=0.0, exclude_boundary=true, filename="values",
                        output_directory="out", append_timestamp=false, write_csv=true,
                        write_json=true, write_file_interval=1, funcs...)

Create a callback to post-process simulation data at regular intervals.
This callback allows for the execution of a user-defined function `func` at specified
intervals during the simulation. The function is applied to the current state of the simulation,
and its results can be saved or used for further analysis. The provided function cannot be
anonymous as the function name will be used as part of the name of the value.

The callback can be triggered either by a fixed number of time steps (`interval`) or by
a fixed interval of simulation time (`dt`).

# Keywords
- `funcs...`:   Functions to be executed at specified intervals during the simulation.
                Each function must have the arguments `(system, data, t)`,
                which will be called for every system, where `data` is a named
                tuple with fields depending on the system type, and `t` is the
                current simulation time. Check the available data for each
                system with `available_data(system)`.
                See [Custom Quantities](@ref custom_quantities)
                for a list of pre-defined custom quantities that can be used here.
- `interval=0`: Specifies the number of time steps between each invocation of the callback.
                If set to `0`, the callback will not be triggered based on time steps.
                Either `interval` or `dt` must be set to something larger than 0.
- `dt=0.0`: Specifies the simulation time interval between each invocation of the callback.
            If set to `0.0`, the callback will not be triggered based on simulation time.
            Either `interval` or `dt` must be set to something larger than 0.
- `exclude_boundary=true`: If set to `true`, boundary particles will be excluded from the post-processing.
- `filename="values"`: The filename of the postprocessing files to be saved.
- `output_directory="out"`: The path where the results of the post-processing will be saved.
- `write_csv=true`: If set to `true`, write a csv file.
- `write_json=true`: If set to `true`, write a json file.
- `append_timestep=false`: If set to `true`, the current timestamp will be added to the filename.
- `write_file_interval=1`: Files will be written after every `write_file_interval` number of
                           postprocessing execution steps. A value of 0 indicates that files
                           are only written at the end of the simulation, eliminating I/O overhead.

# Examples
```jldoctest; output = false
# Create a callback that is triggered every 100 time steps
postprocess_callback = PostprocessCallback(interval=100, example_quantity=kinetic_energy)

# Create a callback that is triggered every 0.1 simulation time units
postprocess_callback = PostprocessCallback(dt=0.1, example_quantity=kinetic_energy)

# output
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PostprocessCallback                                                                              │
│ ═══════════════════                                                                              │
│ dt: ……………………………………………………………………… 0.1                                                              │
│ write file: ………………………………………………… always                                                           │
│ exclude boundary: ………………………………… yes                                                              │
│ filename: ……………………………………………………… values                                                           │
│ output directory: ………………………………… out                                                              │
│ append timestamp: ………………………………… no                                                               │
│ write json file: …………………………………… yes                                                              │
│ write csv file: ……………………………………… yes                                                              │
│ function1: …………………………………………………… example_quantity                                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct PostprocessCallback{I, F}
    interval            :: I
    write_file_interval :: Int
    data                :: Dict{String, Vector{Any}}
    times               :: Array{Float64, 1}
    exclude_boundary    :: Bool
    func                :: F
    filename            :: String
    output_directory    :: String
    append_timestamp    :: Bool
    write_csv           :: Bool
    write_json          :: Bool
    git_hash            :: Ref{String}
end

function PostprocessCallback(; interval::Integer=0, dt=0.0, exclude_boundary=true,
                             output_directory="out", filename="values",
                             append_timestamp=false, write_csv=true, write_json=true,
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

    post_callback = PostprocessCallback(interval, write_file_interval,
                                        Dict{String, Vector{Any}}(), Float64[],
                                        exclude_boundary, funcs, filename, output_directory,
                                        append_timestamp, write_csv, write_json,
                                        Ref("UnknownVersion"))
    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution
        return PeriodicCallback(post_callback, dt,
                                initialize=(initialize_postprocess_callback!),
                                save_positions=(false, false), final_affect=true)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(post_callback, post_callback,
                                save_positions=(false, false),
                                initialize=(initialize_postprocess_callback!))
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!
    print(io, "PostprocessCallback(interval=", callback.interval)
    print(io, ", functions=[")
    print(io, join(keys(callback.func), ", "))
    print(io, "])")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:PostprocessCallback}})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!.affect!
    print(io, "PostprocessCallback(dt=", callback.interval)
    print(io, ", functions=[")
    print(io, join(keys(callback.func), ", "))
    print(io, "])")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!

        function write_file_interval(interval)
            if interval > 1
                return "every $(interval) * interval"
            elseif interval == 1
                return "always"
            elseif interval == 0
                return "no"
            end
        end

        setup = [
            "interval" => string(callback.interval),
            "write file" => write_file_interval(callback.write_file_interval),
            "exclude boundary" => callback.exclude_boundary ? "yes" : "no",
            "filename" => callback.filename,
            "output directory" => callback.output_directory,
            "append timestamp" => callback.append_timestamp ? "yes" : "no",
            "write json file" => callback.write_csv ? "yes" : "no",
            "write csv file" => callback.write_json ? "yes" : "no"
        ]

        for (i, key) in enumerate(keys(callback.func))
            push!(setup, "function$i" => string(key))
        end
        summary_box(io, "PostprocessCallback", setup)
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:PostprocessCallback}})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!.affect!

        function write_file_interval(interval)
            if interval > 1
                return "every $(interval) * dt"
            elseif interval == 1
                return "always"
            elseif interval == 0
                return "no"
            end
        end

        setup = [
            "dt" => string(callback.interval),
            "write file" => write_file_interval(callback.write_file_interval),
            "exclude boundary" => callback.exclude_boundary ? "yes" : "no",
            "filename" => callback.filename,
            "output directory" => callback.output_directory,
            "append timestamp" => callback.append_timestamp ? "yes" : "no",
            "write json file" => callback.write_csv ? "yes" : "no",
            "write csv file" => callback.write_json ? "yes" : "no"
        ]

        for (i, key) in enumerate(keys(callback.func))
            push!(setup, "function$i" => string(key))
        end
        summary_box(io, "PostprocessCallback", setup)
    end
end

function initialize_postprocess_callback!(cb, u, t, integrator)
    # The `PostprocessCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initialize_postprocess_callback!(cb.affect!, u, t, integrator)
end

function initialize_postprocess_callback!(cb::PostprocessCallback, u, t, integrator)
    cb.git_hash[] = compute_git_hash()

    # Apply the callback
    cb(integrator; from_initialize=true)

    return cb
end

# `condition` with interval
function (pp::PostprocessCallback)(u, t, integrator)
    (; interval) = pp

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (pp::PostprocessCallback)(integrator; from_initialize=false)
    @trixi_timeit timer() "apply postprocess cb" begin
        vu_ode = integrator.u
        if from_initialize
            # Avoid calling `get_du` here, since it will call the RHS function
            # if it is called before the first time step.
            # This would cause problems with `semi.update_callback_used`,
            # which might not yet be set to `true` at this point if the `UpdateCallback`
            # comes AFTER the `PostprocessCallback` in the `CallbackSet`.
            dv_ode, du_ode = zero(vu_ode).x
        else
            # Depending on the time integration scheme, this might call the RHS function
            @trixi_timeit timer() "update du" begin
                # Don't create sub-timers here to avoid cluttering the timer output
                @notimeit timer() dv_ode, du_ode = get_du(integrator).x
            end
        end
        v_ode, u_ode = vu_ode.x
        semi = integrator.p
        t = integrator.t
        filenames = system_names(semi.systems)
        new_data = false

        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
        end

        foreach_system(semi) do system
            if system isa AbstractBoundarySystem && pp.exclude_boundary
                return
            end

            system_index = system_indices(system, semi)

            for (key, f) in pp.func
                result_ = custom_quantity(f, system, dv_ode, du_ode, v_ode, u_ode, semi, t)
                if result_ !== nothing
                    # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
                    result = transfer2cpu(result_)
                    add_entry!(pp, string(key), t, result, filenames[system_index])
                    new_data = true
                end
            end
        end

        if new_data
            push!(pp.times, t)
        end

        if isfinished(integrator) ||
           (pp.write_file_interval > 0 && backup_condition(pp, integrator))
            write_postprocess_callback(pp, integrator)
        end

        # Tell OrdinaryDiffEq that `u` has not been modified
        u_modified!(integrator, false)
    end
end

@inline function backup_condition(cb::PostprocessCallback{Int}, integrator)
    return integrator.stats.naccept > 0 &&
           round(integrator.stats.naccept / cb.interval) % cb.write_file_interval == 0
end

@inline function backup_condition(cb::PostprocessCallback, integrator)
    return integrator.stats.naccept > 0 &&
           round(Int, integrator.t / cb.interval) % cb.write_file_interval == 0
end

# After the simulation has finished, this function is called to write the data to a JSON file
function write_postprocess_callback(pp::PostprocessCallback, integrator)
    isempty(pp.data) && return

    mkpath(pp.output_directory)

    data = Dict{String, Any}()
    data["meta"] = create_meta_data_dict(pp, integrator)

    prepare_series_data!(data, pp)

    time_stamp = ""
    if pp.append_timestamp
        time_stamp = string("_", Dates.format(now(), "YY-mm-ddTHHMMSS"))
    end

    filename_json = pp.filename * time_stamp * ".json"
    filename_csv = pp.filename * time_stamp * ".csv"

    if pp.write_json
        abs_file_path = joinpath(abspath(pp.output_directory), filename_json)

        open(abs_file_path, "w") do file
            # Indent by 4 spaces
            JSON.json(file, data; pretty=4, allownan=true)
        end
    end
    if pp.write_csv
        abs_file_path = joinpath(abspath(pp.output_directory), filename_csv)

        write_csv(abs_file_path, data)
    end
end

# This function prepares the data for writing to a JSON file by creating a dictionary
# that maps each key to separate arrays of times and values, sorted by time, and includes system name.
function prepare_series_data!(data, post_callback)
    for (key, data_array) in post_callback.data
        data_values = [value for value in data_array]

        # The penultimate string in the key is the system_name
        system_name = split(key, '_')[end - 1]

        data[key] = create_series_dict(data_values, post_callback.times, system_name)
    end

    return data
end

function create_series_dict(values, times, system_name="")
    return Dict("type" => "series",
                "datatype" => eltype(values),
                "n_values" => length(values),
                "system_name" => system_name,
                "values" => values,
                "time" => times)
end

function write_csv(abs_file_path, data)
    times = Float64[]

    for val in values(data)
        if haskey(val, "time")
            times = val["time"]
            break
        end
    end

    # Initialize DataFrame with time column
    df = DataFrame(time=times)

    for (key, series) in data
        # Ensure we only process data entries, excluding any meta data or non-data entries.
        # Metadata is stored as `data["meta"]`, while data entries contain `_$(system_name)`
        if occursin("_", key)
            values = series["values"]
            # Add a new column to the DataFrame for each series of values
            df[!, Symbol(key)] = values
        end
    end

    # Write the DataFrame to a CSV file
    CSV.write(abs_file_path, df)
end

function add_entry!(pp, entry_key, t, value, system_name)
    # Get the list of data entries for the system, or initialize it if it doesn't exist
    entries = get!(pp.data, entry_key * "_" * system_name, Any[])

    # Add the new entry to the list
    push!(entries, value)
end
