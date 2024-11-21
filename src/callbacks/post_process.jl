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
                Each function must have the arguments `(v, u, t, system)`,
                and will be called for every system, where `v` and `u` are the
                wrapped solution arrays for the corresponding system and `t` is
                the current simulation time. Note that working with these `v`
                and `u` arrays requires undocumented internal functions of
                TrixiParticles. See [Custom Quantities](@ref custom_quantities)
                for a list of pre-defined functions that can be used here.
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
    config              :: OutputConfig
    write_csv           :: Bool
    write_json          :: Bool
    git_hash            :: Ref{String}
end

function PostprocessCallback(; interval::Integer=0, dt=0.0, exclude_boundary=true,
                             output_config=OutputConfig(filename="values"), write_csv=true,
                             write_json=true,
                             write_file_interval::Integer=1, funcs...)
    if isempty(funcs)
        throw(ArgumentError("`funcs` cannot be empty"))
    end

    validate_interval_and_dt(interval, dt)

    if dt > 0
        interval = Float64(dt)
    end

    post_callback = PostprocessCallback(interval, write_file_interval,
                                        Dict{String, Vector{Any}}(), Float64[],
                                        exclude_boundary, funcs, output_config, write_csv,
                                        write_json, Ref("UnknownVersion"))
    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution
        return PeriodicCallback(post_callback, dt,
                                initialize=initialize_postprocess_callback!,
                                save_positions=(false, false), final_affect=true)
    else
        # The first one is the `condition`, the second the `affect!`
        return DiscreteCallback(post_callback, post_callback,
                                save_positions=(false, false),
                                initialize=initialize_postprocess_callback!)
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

# Detailed display function for PostprocessCallback within DiscreteCallback
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!
        setup = collect_callback_setup(callback, "interval")
        summary_box(io, "PostprocessCallback", setup)
    end
end

# Detailed display function for PostprocessCallback within PeriodicCallback
function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:PostprocessCallback}})
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!.affect!
        setup = collect_callback_setup(callback, "dt")
        summary_box(io, "PostprocessCallback", setup)
    end
end

# Helper function to collect callback setup information for display
function collect_callback_setup(callback, interval_label)
    function write_file_interval_desc(interval)
        if interval > 1
            return "every $(interval) * $interval_label"
        elseif interval == 1
            return "always"
        elseif interval == 0
            return "no"
        end
    end

    setup = [
        interval_label => string(callback.interval),
        "write file" => write_file_interval_desc(callback.write_file_interval),
        "exclude boundary" => callback.exclude_boundary ? "yes" : "no",
        "filename" => callback.config.filename,
        "output directory" => callback.config.output_directory,
        "append timestamp" => callback.config.append_timestamp ? "yes" : "no",
        "write json file" => callback.write_json ? "yes" : "no",
        "write csv file" => callback.write_csv ? "yes" : "no"
    ]

    for (i, key) in enumerate(keys(callback.funcs))
        push!(setup, "function$i" => string(key))
    end
    return setup
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
    cb(integrator)

    return cb
end

# `condition` with interval
function (pp::PostprocessCallback)(u, t, integrator)
    return condition_integrator_interval(integrator, pp.interval)
end

# `affect!`
function (pp::PostprocessCallback)(integrator)
    # Extract solution arrays and parameters
    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t
    filenames = system_names(semi.systems)
    new_data_collected = false

    # Update systems to compute quantities (e.g., density, pressure)
    update_systems_and_nhs(v_ode, u_ode, semi, t; update_from_callback=true)

    # Loop over each system in the simulation
    foreach_system(semi) do system
        # Skip boundary systems if exclude_boundary is true
        if system isa BoundarySystem && pp.exclude_boundary
            return
        end

        # Get the index and name of the current system
        system_index = system_indices(system, semi)

        # Wrap solution arrays for the current system
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        # Apply each function in pp.funcs to the current system
        for (key, f) in pp.funcs
            result = f(v, u, t, system)
            if result !== nothing
                # Add the result to the data collection
                add_entry!(pp, string(key), t, result, filenames[system_index])
                new_data_collected = true
            end
        end
    end

    # Record the time if new data was collected
    if new_data_collected
        push!(pp.times, t)
    end

    # Write data to files if necessary
    if isfinished(integrator) || should_write_files(pp, integrator)
        write_postprocess_data(pp)
    end

    # Indicate that the integrator's `u` has not been modified
    u_modified!(integrator, false)
end

# Helper function to determine if it's time to write output files
function should_write_files(pp::PostprocessCallback, integrator)
    # Check if we should write files based on the write_file_interval
    if pp.interval isa Integer
        # Interval is based on step count
        return integrator.stats.naccept > 0 &&
               (integrator.stats.naccept ÷ pp.interval) % pp.write_file_interval == 0
    else
        # Interval is based on simulation time
        return integrator.stats.naccept > 0 &&
               (floor(Int, integrator.t / pp.interval) % pp.write_file_interval == 0)
    end
end

# Function to write the collected post-processing data to files
function write_postprocess_data(pp::PostprocessCallback)
    # Return early if there's no data to write
    isempty(pp.data) && return

    # Ensure the output directory exists
    mkpath(pp.config.output_directory)

    # Prepare data for output
    data = Dict{String, Any}()
    write_meta_data!(data, pp.git_hash[])
    prepare_series_data!(data, pp)

    # Write JSON file if enabled
    if pp.write_json
        filename_json = build_filepath(pp.config; extension="json")
        open(filename_json, "w") do file
            JSON.print(file, data, 4)  # Indent by 4 spaces
        end
    end

    # Write CSV file if enabled
    if pp.write_csv
        filename_csv = build_filepath(pp.config; extension="csv")
        write_csv_file(filename_csv, data)
    end
end

# Prepare series data for output by organizing it into a structured dictionary
function prepare_series_data!(data, pp::PostprocessCallback)
    for (key, data_array) in pp.data
        data_values = [value for value in data_array]

        # Extract system name from the key (assumes key format "function_systemname")
        system_name = split(key, '_')[end - 1]

        data[key] = create_series_dict(data_values, pp.times, system_name)
    end
    return data
end

# Helper function to create a series dictionary for each data series
function create_series_dict(values, times, system_name="")
    return Dict("type" => "series",
                "datatype" => eltype(values),
                "n_values" => length(values),
                "system_name" => system_name,
                "values" => values,
                "time" => times)
end

# Write meta-data to the output data dictionary
function write_meta_data!(data, git_hash)
    meta_data = Dict("solver_name" => "TrixiParticles.jl",
                     "solver_version" => git_hash,
                     "julia_version" => string(VERSION))

    data["meta"] = meta_data
    return data
end

# Function to write the data to a CSV file
function write_csv_file(file_path, data)
    times = Float64[]

    # Find the time series from the data
    for val in values(data)
        if haskey(val, "time")
            times = val["time"]
            break
        end
    end

    # Initialize DataFrame with the time column
    df = DataFrame(time=times)

    # Add data series to the DataFrame
    for (key, series) in data
        # Skip meta-data entries
        if occursin("_", key)
            values = series["values"]
            df[!, Symbol(key)] = values
        end
    end

    # Write the DataFrame to a CSV file
    CSV.write(file_path, df)
end

# Add a new entry to the data collection
function add_entry!(pp::PostprocessCallback, entry_key::String, t::Float64, value, system_name::String)
    # Construct the full key by combining the entry key and system name
    full_key = entry_key * "_" * system_name

    # Get or initialize the data vector for this key
    entries = get!(pp.data, full_key, Any[])

    # Add the new value to the entries
    push!(entries, value)
end
