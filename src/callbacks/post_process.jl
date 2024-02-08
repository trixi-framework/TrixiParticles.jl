using JSON
using CSV
using DataFrames

struct DataEntry
    value::Float64
    time::Float64
    system::String
end

"""
    PostprocessCallback(funcs...; interval::Integer=0, dt=0.0, exclude_boundary=true, filename="values",
                        output_directory=".", overwrite=true, write_csv=true, write_json=true)

Create a callback for post-processing simulation data at regular intervals.
This callback allows for the execution of a user-defined function `func` at specified
intervals during the simulation. The function is applied to the current state of the simulation,
and its results can be saved or used for further analysis.

The callback can be triggered either by a fixed number of time steps (`interval`) or by
a fixed interval of simulation time (`dt`).

# Keywords
- `interval=0`: Specifies the number of time steps between each invocation of the callback.
                If set to `0`, the callback will not be triggered based on time steps.
                Either interval or dt can be set to something larger than 0.
- `dt=0.0`: Specifies the simulation time interval between each invocation of the callback.
            If set to `0.0`, the callback will not be triggered based on simulation time.
            Either interval or dt can be set to something larger than 0.
- `exclude_boundary=true`: If set to `true`, boundary particles will be excluded from the post-processing.
- `filename="values"`: The filename of the postprocessing files to be saved.
- `output_directory="out"`: The path where the results of the post-processing will be saved.
- `overwrite=true`: If set to `true`, existing files with the same name as `filename` will be overwritten.
- `write_csv=true`: If set to `true`, write an csv file.
- `write_json=true`: If set to `true`, write a json file.

# Examples
```julia
example_function = function(pp, t, system, u, v, system_name) println("test_func ", t) end

# Create a callback that is triggered every 100 time steps
postprocess_callback = PostprocessCallback(example_function, interval=100)

# Create a callback that is triggered every 0.1 simulation time units
postprocess_callback = PostprocessCallback(example_function, dt=0.1)
```
"""
mutable struct PostprocessCallback{I, F}
    interval         :: I
    values           :: Dict{String, Vector{DataEntry}}
    exclude_boundary :: Bool
    func             :: F
    filename         :: String
    output_directory :: String
    overwrite        :: Bool
    write_csv        :: Bool
    write_json       :: Bool
end

function PostprocessCallback(funcs...; interval::Integer=0, dt=0.0, exclude_boundary=true,
                             output_directory="out", filename="values", overwrite=true,
                             write_csv=true, write_json=true)
    if dt > 0 && interval > 0
        throw(ArgumentError("Setting both interval and dt is not supported!"))
    end

    if dt > 0
        interval = Float64(dt)
    end

    post_callback = PostprocessCallback(interval, Dict{String, Vector{DataEntry}}(),
                                        exclude_boundary, funcs, filename, output_directory,
                                        overwrite, write_csv, write_json)
    if dt > 0
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(post_callback, dt,
                                initialize=initialize_post_callback!,
                                save_positions=(false, false), final_affect=true)
    else
        # The first one is the condition, the second the affect!
        return DiscreteCallback(post_callback, post_callback,
                                save_positions=(false, false),
                                initialize=initialize_post_callback!)
    end
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!
    print(io, "PostprocessCallback(interval=", callback.interval)
    print(io, "functions=[")
    print(io, join(callback.func, ", "))
    print(io, "])")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:PostprocessCallback}})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!.affect!
    print(io, "PostprocessCallback(interval=", callback.interval)
    print(io, "functions=[")
    print(io, join(callback.func, ", "))
    print(io, "])")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!
        setup = ["interval" => string(callback.interval)]

        for (i, f) in enumerate(callback.func)
            push!(setup, "function$i" => string(nameof(f)))
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
        setup = ["interval" => string(callback.interval)]

        for (i, f) in enumerate(callback.func)
            push!(setup, "function$i" => string(nameof(f)))
        end
        summary_box(io, "PostprocessCallback", setup)
    end
end

function initialize_post_callback!(cb, u, t, integrator)
    # The `PostprocessCallback` is either `cb.affect!` (with `DiscreteCallback`)
    # or `cb.affect!.affect!` (with `PeriodicCallback`).
    # Let recursive dispatch handle this.
    initialize_post_callback!(cb.affect!, u, t, integrator)
end

function initialize_post_callback!(cb::PostprocessCallback, u, t, integrator)

    # Update systems to compute quantities like density and pressure.
    semi = integrator.p
    v_ode, u_ode = u.x

    # Update systems to compute quantities like density and pressure
    update_systems_and_nhs(v_ode, u_ode, semi, t)

    # Apply the callback
    cb(integrator)
    return nothing
end

# condition with interval
function (pp::PostprocessCallback)(u, t, integrator)
    (; interval) = pp

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return (interval > 0 && ((integrator.stats.naccept % interval == 0) &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0))) ||
           isfinished(integrator)
end

# affect! function for an array of functions
function (pp::PostprocessCallback)(integrator)
    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t
    filenames = system_names(semi.systems)

    foreach_system(semi) do system
        if system isa BoundarySystem && pp.exclude_boundary
            return
        end

        system_index = system_indices(system, semi)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        for f in pp.func
            f(pp, t, system, u, v, filenames[system_index])
        end
    end

    if isfinished(integrator)
        pp(integrator, true)
    end

    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)
end

# This function prepares the data for writing to a JSON file by creating a dictionary
# that maps each key to separate arrays of times and values, sorted by time, and includes system name.
function prepare_series_data!(data, post_callback)
    for (key, data_array) in post_callback.values
        # Sort the data_array by time
        sorted_data_array = sort(data_array, by=data -> data.time)

        # Extracting times and values into separate arrays
        data_times = [data.time for data in sorted_data_array]
        data_values = [data.value for data in sorted_data_array]

        # Assuming each DataEntry in `sorted_data_array` has a `system` field.
        system_name = isempty(sorted_data_array) ? "" : sorted_data_array[1].system

        data[key] = create_series_dict(data_values, data_times, system_name)
    end

    return data
end

function create_series_dict(values, times, system_name="")
    return Dict("type" => "series",
                "datatype" => eltype(values),
                "novalues" => length(values),
                "system_name" => system_name,
                "values" => values,
                "time" => times)
end

function meta_data!(data)
    meta_data = Dict("solver_name" => "TrixiParticles.jl",
                     "solver_version" => get_git_hash(),
                     "julia_version" => get_julia_version())

    data["meta"] = meta_data
    return data
end

# After the simulation has finished, this function is called to write the data to a JSON file.
function (pp::PostprocessCallback)(integrator, finished::Bool)
    if isempty(pp.values)
        return nothing
    end

    data = Dict()
    data = meta_data!(data)
    data = prepare_series_data!(data, pp)

    filename_json = pp.filename * ".json"
    filename_csv = pp.filename * ".csv"
    if !pp.overwrite
        filename_json = get_unique_filename(pp.filename, ".json")
        filename_csv = get_unique_filename(pp.filename, ".csv")
    end

    if pp.write_json
        abs_file_path = joinpath(abspath(normpath(pp.output_directory)), filename_json)
        println("writing a postproccessing results to ", abs_file_path)

        open(abs_file_path, "w") do file
            JSON.print(file, data, 4)
        end
    end
    if pp.write_csv
        abs_file_path = joinpath(abspath(normpath(pp.output_directory)), filename_csv)
        println("writing a postproccessing results to ", abs_file_path)

        write_csv(abs_file_path, data)
    end
end

function write_csv(abs_file_path, data)
    times = collect(values(data))[1]["time"]
    df = DataFrame(time=times) # Initialize DataFrame with time column

    for (key, series) in data
        # Ensure we only process data entries, excluding any meta data or non-data entries
        if occursin("_", key) # A simple check to ensure we're dealing with data series
            values = series["values"]
            # Add a new column to the DataFrame for each series of values
            df[!, Symbol(key)] = values
        end
    end

    # Write the DataFrame to a CSV file
    CSV.write(abs_file_path, df)
end

function add_entry!(pp, entry_key, t, value, system_name)
    # Get the list of DataEntry for the system, or initialize it if it doesn't exist
    entries = get!(pp.values, entry_key * "_" * system_name, DataEntry[])

    # Add the new entry to the list
    push!(entries, DataEntry(value, t, system_name))
end

function calculate_ekin(pp, t, system, u, v, system_name)
    ekin = 0.0
    for particle in each_moving_particle(system)
        velocity = current_velocity(v, system, particle)
        ekin += 0.5 * hydrodynamic_mass(system, particle) * dot(velocity, velocity)
    end
    add_entry!(pp, "ekin", t, ekin, system_name)
end

function calculate_total_mass(pp, t, system, u, v, system_name)
    total_mass = 0.0
    for particle in each_moving_particle(system)
        total_mass += hydrodynamic_mass(system, particle)
    end
    add_entry!(pp, "totalm", t, total_mass, system_name)
end

function max_pressure(pp, t, system, u, v, system_name)
    max_p = 0.0
    for particle in each_moving_particle(system)
        pressure = particle_pressure(v, system, particle)
        if max_p < pressure
            max_p = pressure
        end
    end
    add_entry!(pp, "max_p", t, max_p, system_name)
end

function min_pressure(pp, t, system, u, v, system_name)
    min_p = typemax(Int64)
    for particle in each_moving_particle(system)
        pressure = particle_pressure(v, system, particle)
        if min_p > pressure
            min_p = pressure
        end
    end
    add_entry!(pp, "min_p", t, min_p, system_name)
end

function avg_pressure(pp, t, system, u, v, system_name)
    total_pressure = 0.0
    count = 0

    for particle in each_moving_particle(system)
        total_pressure += particle_pressure(v, system, particle)
        count += 1
    end

    avg_p = count > 0 ? total_pressure / count : 0.0
    add_entry!(pp, "avg_p", t, avg_p, system_name)
end

function max_density(pp, t, system, u, v, system_name)
    max_rho = 0.0
    for particle in each_moving_particle(system)
        rho = particle_density(v, system, particle)
        if max_rho < rho
            max_rho = rho
        end
    end
    add_entry!(pp, "max_rho", t, max_rho, system_name)
end

function min_density(pp, t, system, u, v, system_name)
    min_rho = typemax(Int64)
    for particle in each_moving_particle(system)
        rho = particle_density(v, system, particle)
        if min_rho > rho
            min_rho = rho
        end
    end
    add_entry!(pp, "min_rho", t, min_rho, system_name)
end

function avg_density(pp, t, system, u, v, system_name)
    total_density = 0.0
    count = 0

    for particle in each_moving_particle(system)
        total_density += particle_density(v, system, particle)
        count += 1
    end

    avg_rho = count > 0 ? total_density / count : 0.0
    add_entry!(pp, "avg_rho", t, avg_rho, system_name)
end
