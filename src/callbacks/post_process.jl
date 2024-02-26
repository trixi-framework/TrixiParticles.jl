"""
    PostprocessCallback(; interval::Integer=0, dt=0.0, exclude_boundary=true, filename="values",
                        output_directory="out", append_timestamp=false, write_csv=true,
                        write_json=true, backup_period=0, funcs...)

Create a callback to post-process simulation data at regular intervals.
This callback allows for the execution of a user-defined function `func` at specified
intervals during the simulation. The function is applied to the current state of the simulation,
and its results can be saved or used for further analysis. The provided function cannot be
anonymous as the function name will be used as part of the name of the value.

The callback can be triggered either by a fixed number of time steps (`interval`) or by
a fixed interval of simulation time (`dt`).


# Keywords
- `funcs...`: Functions to be executed at specified intervals during the simulation.
              The functions must have the arguments `(v, u, t, system)`.
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
- `backup_period=0`: Specifies that a backup should be created after every `backup_period`
                     number of postprocessing execution steps. A value of 0 indicates that
                     backups should not be automatically generated during postprocessing.

# Examples
```julia
function example_function(v, u, t, system)
    println("test_func ", t)
end

# Create a callback that is triggered every 100 time steps
postprocess_callback = PostprocessCallback(example_function, interval=100)

# Create a callback that is triggered every 0.1 simulation time units
postprocess_callback = PostprocessCallback(example_function, dt=0.1)
```
"""
struct PostprocessCallback{I, F}
    interval         :: I
    backup_period    :: Int
    data             :: Dict{String, Vector{Any}}
    times            :: Array{Float64, 1}
    exclude_boundary :: Bool
    func             :: F
    filename         :: String
    output_directory :: String
    append_timestamp :: Bool
    write_csv        :: Bool
    write_json       :: Bool
end

function PostprocessCallback(; interval::Integer=0, dt=0.0, exclude_boundary=true,
                             output_directory="out", filename="values",
                             append_timestamp=false, write_csv=true, write_json=true,
                             backup_period::Integer=1, funcs...)
    if isempty(funcs)
        throw(ArgumentError("`funcs` cannot be empty"))
    end

    if dt > 0 && interval > 0
        throw(ArgumentError("setting both `interval` and `dt` is not supported"))
    end

    if dt > 0
        interval = Float64(dt)
    end

    post_callback = PostprocessCallback(interval, backup_period,
                                        Dict{String, Vector{Any}}(), Float64[],
                                        exclude_boundary, funcs, filename, output_directory,
                                        append_timestamp, write_csv, write_json)
    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(post_callback, dt,
                                initialize=initialize_postprocess_callback!,
                                save_positions=(false, false), final_affect=true)
    else
        # The first one is the condition, the second the affect!
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

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!

        function write_file_interval(interval)
            if interval  > 1
                return "every $(interval) * interval"
            elseif interval == 1
                return "every interval"
            elseif interval == 0
                return "no"
            end
        end

        setup = [
            "interval" => string(callback.interval),
            "write backup" => write_file_interval(callback.backup_period),
            "exclude boundary" => callback.exclude_boundary ? "yes" : "no",
            "filename" => callback.filename,
            "output directory" => callback.output_directory,
            "append timestamp" => callback.append_timestamp ? "yes" : "no",
            "write json file" => callback.write_csv ? "yes" : "no",
            "write csv file" => callback.write_json ? "yes" : "no",
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
            if interval  > 1
                return "every $(interval) * dt"
            elseif interval == 1
                return "every dt"
            elseif interval == 0
                return "no"
            end
        end

        setup = [
            "dt" => string(callback.interval),
            "write backup" => write_file_interval(callback.backup_period),
            "exclude boundary" => callback.exclude_boundary ? "yes" : "no",
            "filename" => callback.filename,
            "output directory" => callback.output_directory,
            "append timestamp" => callback.append_timestamp ? "yes" : "no",
            "write json file" => callback.write_csv ? "yes" : "no",
            "write csv file" => callback.write_json ? "yes" : "no",
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
    # Apply the callback
    cb(integrator)
    return nothing
end

# condition with interval
function (pp::PostprocessCallback)(u, t, integrator)
    (; interval) = pp

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (pp::PostprocessCallback)(integrator)
    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t
    filenames = system_names(semi.systems)
    new_data = false

    # Update systems to compute quantities like density and pressure
    update_systems_and_nhs(v_ode, u_ode, semi, t)

    foreach_system(semi) do system
        if system isa BoundarySystem && pp.exclude_boundary
            return
        end

        system_index = system_indices(system, semi)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        for (key, f) in pp.func
            result = f(v, u, t, system)
            if result !== nothing
                add_entry!(pp, string(key), t, result, filenames[system_index])
                new_data = true
            end
        end
    end

    if new_data
        push!(pp.times, t)
    end

    if isfinished(integrator) || (pp.backup_period > 0 && backup_condition(pp, integrator))
        write_postprocess_callback(pp)
    end

    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)
end

@inline function backup_condition(cb::PostprocessCallback{Int}, integrator)
    return integrator.stats.naccept > 0 &&
           round(integrator.stats.naccept / cb.interval) % cb.backup_period == 0
end

@inline function backup_condition(cb::PostprocessCallback, integrator)
    return integrator.stats.naccept > 0 &&
           round(Int, integrator.t / cb.interval) % cb.backup_period == 0
end

# After the simulation has finished, this function is called to write the data to a JSON file
function write_postprocess_callback(pp::PostprocessCallback)
    if isempty(pp.data)
        return
    end

    data = Dict{String, Any}()
    write_meta_data!(data)
    prepare_series_data!(data, pp)

    time_stamp = ""
    if pp.append_timestamp
        time_stamp = string("_", Dates.format(now(), "YY-mm-ddTHHMMSS"))
    end

    filename_json = pp.filename * time_stamp * ".json"
    filename_csv = pp.filename * time_stamp * ".csv"

    if pp.write_json
        abs_file_path = joinpath(abspath(pp.output_directory), filename_json)
        @info "Writing postprocessing results to $abs_file_path"

        open(abs_file_path, "w") do file
            # Indent by 4 spaces
            JSON.print(file, data, 4)
        end
    end
    if pp.write_csv
        abs_file_path = joinpath(abspath(pp.output_directory), filename_csv)
        @info "Writing postprocessing results to $abs_file_path"

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

function write_meta_data!(data)
    meta_data = Dict("solver_name" => "TrixiParticles.jl",
                     "solver_version" => get_git_hash(),
                     "julia_version" => string(VERSION))

    data["meta"] = meta_data
    return data
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

function kinetic_energy(v, u, t, system)
    return sum(each_moving_particle(system)) do particle
        velocity = current_velocity(v, system, particle)
        return 0.5 * system.mass[particle] * dot(velocity, velocity)
    end
end

function total_mass(v, u, t, system)
    return sum(each_moving_particle(system)) do particle
        return system.mass[particle]
    end
end

function max_pressure(v, u, t, system)
    return maximum(particle -> particle_pressure(v, system, particle),
                   each_moving_particle(system))
end

function min_pressure(v, u, t, system)
    return minimum(particle -> particle_pressure(v, system, particle),
                   each_moving_particle(system))
end

function avg_pressure(v, u, t, system)
    if n_moving_particles(system) == 0
        return 0.0
    end

    sum_ = sum(particle -> particle_pressure(v, system, particle),
               each_moving_particle(system))
    return sum_ / n_moving_particles(system)
end

function max_density(v, u, t, system)
    return maximum(particle -> particle_density(v, system, particle),
                   each_moving_particle(system))
end

function min_density(v, u, t, system)
    return minimum(particle -> particle_density(v, system, particle),
                   each_moving_particle(system))
end

function avg_density(v, u, t, system)
    if n_moving_particles(system) == 0
        return 0.0
    end

    sum_ = sum(particle -> particle_density(v, system, particle),
               each_moving_particle(system))
    return sum_ / n_moving_particles(system)
end
