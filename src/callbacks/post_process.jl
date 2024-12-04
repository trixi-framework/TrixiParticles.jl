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
function (postprocess_callback::PostprocessCallback)(integrator)
    # Extract the current solution state and parameters from the integrator
    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t
    system_names_list = system_names(semi.systems)
    data_collected = false

    # Update the systems to compute derived quantities (e.g., density, pressure)
    update_systems_and_nhs(
        v_ode,
        u_ode,
        semi_discrete_problem,
        t;
        update_from_callback = true
    )

    # Iterate over each system in the simulation
    foreach_system(semi_discrete_problem) do system
        # Skip boundary systems if 'exclude_boundary' is true
        if system isa BoundarySystem && postprocess_callback.exclude_boundary
            return
        end

        # Get the index of the current system
        system_index = system_indices(system, semi_discrete_problem)

        # Extract the solution arrays specific to the current system
        v_system = wrap_v(v_ode, system, semi_discrete_problem)
        u_system = wrap_u(u_ode, system, semi_discrete_problem)

        # Apply each user-defined function to the current system
        for (func_name, func) in postprocess_callback.funcs
            # Call the function with the current system's data
            result = func(v_system, u_system, t, system)
            if result !== nothing
                # Add the result to the data collection
                add_entry!(
                    postprocess_callback,
                    string(func_name),
                    t,
                    result,
                    system_names_list[system_index]
                )
                data_collected = true
            end
        end
    end

    # Record the time if new data was collected
    if data_collected
        push!(postprocess_callback.times, t)
    end

    # Write data to files if necessary
    if isfinished(integrator) || should_write_files(postprocess_callback, integrator)
        write_postprocess_data(postprocess_callback)
    end

    # Indicate that the integrator's `u` has not been modified
    u_modified!(integrator, false)
end

# Helper function to determine if it's time to write output files
function should_write_files(postprocess_callback::PostprocessCallback, integrator)
    # Decide based on whether the interval is defined by steps or time
    if postprocess_callback.interval isa Integer
        # Interval is based on step count
        steps_completed = integrator.stats.naccept
        intervals_completed = steps_completed ÷ postprocess_callback.interval
        # Check if it's time to write files
        return steps_completed > 0 &&
               (intervals_completed % postprocess_callback.write_file_interval == 0)
    else
        # Interval is based on simulation time
        time_elapsed = integrator.t
        intervals_completed = floor(Int, time_elapsed / postprocess_callback.interval)
        # Check if it's time to write files
        return integrator.stats.naccept > 0 &&
               (intervals_completed % postprocess_callback.write_file_interval == 0)
    end
end

# Function to write the collected post-processing data to files
function write_postprocess_data(postprocess_callback::PostprocessCallback)
    # Return early if there's no data to write
    isempty(postprocess_callback.data) && return

    # Ensure the output directory exists
    mkpath(postprocess_callback.config.output_directory)

    # Prepare the data dictionary for output
    data = Dict{String, Any}()
    # Add meta-data to the data dictionary
    write_meta_data!(data, postprocess_callback.git_hash[])
    # Organize the collected data into series format
    prepare_series_data!(data, postprocess_callback)

    # Write JSON file if enabled
    if postprocess_callback.write_json
        json_filename = build_filepath(postprocess_callback.config; extension = "json")
        open(json_filename, "w") do file
            JSON.print(file, data, 4)  # Indent by 4 spaces
        end
    end

    # Write CSV file if enabled
    if postprocess_callback.write_csv
        csv_filename = build_filepath(postprocess_callback.config; extension = "csv")
        write_csv_file(csv_filename, data)
    end
end

# Prepare series data for output by organizing it into a structured dictionary
function prepare_series_data!(data_dict::Dict{String, Any}, postprocess_callback::PostprocessCallback)
    for (key, data_array) in postprocess_callback.data
        # Extract collected values
        data_values = [value for value in data_array]

        # Extract system name from the key (assumes key format "functionname_systemname")
        key_parts = split(key, '_')
        system_name = key_parts[end - 1]  # Assuming system name is the penultimate part

        # Create a series dictionary for this data series
        data_dict[key] = create_series_dict(data_values, postprocess_callback.times, system_name)
    end
    return data_dict
end

# Helper function to create a series dictionary for each data series
function create_series_dict(values::Vector{Any}, times::Vector{Float64}, system_name::String = "")
    return Dict(
        "type" => "series",
        "datatype" => eltype(values),
        "n_values" => length(values),
        "system_name" => system_name,
        "values" => values,
        "time" => times
    )
end

# Write meta-data to the output data dictionary
function write_meta_data!(data_dict::Dict{String, Any}, git_hash::String)
    meta_data = Dict(
        "solver_name" => "TrixiParticles.jl",
        "solver_version" => git_hash,
        "julia_version" => string(VERSION)
    )

    data_dict["meta"] = meta_data
    return data_dict
end

# Function to write the data to a CSV file
function write_csv_file(file_path::String, data::Dict{String, Any})
    times = Float64[]

    # Find the time series from the data
    for series_data in values(data)
        if haskey(series_data, "time")
            times = series_data["time"]
            break
        end
    end

    # Initialize DataFrame with the time column
    df = DataFrame(time = times)

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
function add_entry!(
    postprocess_callback::PostprocessCallback,
    entry_key::String,
    time::Float64,
    value::Any,
    system_name::String
)
    # Construct the full key by combining the function name and system name
    full_key = entry_key * "_" * system_name

    # Get or initialize the data vector for this key
    data_entries = get!(postprocess_callback.data, full_key, Any[])

    # Add the new value to the data entries
    push!(data_entries, value)
end
