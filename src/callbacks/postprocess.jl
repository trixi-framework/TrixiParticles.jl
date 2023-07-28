using JSON

struct DataEntry
    value::Float64
    time::Float64
end

mutable struct PostprocessCallback
    values::Dict{String, Vector{DataEntry}}
    dt::Vector{Float64}
end

function PostprocessCallback(; interval=0)
    post_callback = PostprocessCallback(Dict(), Float64[])

    DiscreteCallback(post_callback, post_callback,
                     save_positions=(false, false),
                     initialize=initialize_post_callback)
end

function (post_callback::PostprocessCallback)(u, t, integrator)
    push!(post_callback.dt, integrator.dt)
    for system in integrator.p.systems
        keys = pp_keys(system)
        keys !== nothing && update_values(post_callback, keys, system, t)
    end
    return isfinished(integrator)
end

function update_values(post_callback, keys, system, t)
    for key in keys
        data_available, value = pp_value(system, key)
        data_available && push!(get!(post_callback.values, key, DataEntry[]), DataEntry(value, t))
    end
end

function (post_callback::PostprocessCallback)(integrator)
    isempty(post_callback.dt) && isempty(post_callback.values) && return nothing

    series_data = Dict("dt" => create_dict(post_callback.dt))
    for (key, data_array) in post_callback.values
        series_data[key] = create_dict([data.value for data in data_array], [data.time for data in data_array])
    end

    filename = find_unique_filename("values", ".json")
    open(filename, "w") do file
        JSON.print(file, series_data, 4)
    end
end

function find_unique_filename(basename, extension)
    filename, counter = basename*extension, 1
    while isfile(filename)
        filename = basename*string(counter)*extension
        counter += 1
    end
    return filename
end

function create_dict(values, times=nothing)
    Dict("type" => "series", "datatype" => typeof(values),
         "novalues" => length(values), "values" => values, "time" => times)
end

function initialize_post_callback(discrete_callback, u, t, integrator)
    return nothing
end

function pp_keys(system)
    # skip systems that don't have support implemented
    return nothing
end

function pp_value(system, key)
    # skip systems that don't have support implemented
    return false, 0.0
end
