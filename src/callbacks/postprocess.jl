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
    update_values_for_all_systems(post_callback, integrator.p.systems, t)
    return isfinished(integrator)
end

function update_values_for_all_systems(post_callback, systems, t)
    for system in systems
        keys = pp_keys(system)
        if keys !== nothing
            update_values_for_single_system(post_callback, keys, system, t)
        end
    end
end

function update_values_for_single_system(post_callback, keys, system, t)
    for key in keys
        data_available, value = pp_value(system, key)
        if data_available
            data_entry_array = get!(post_callback.values, key, DataEntry[])
            push!(data_entry_array, DataEntry(value, t))
        end
    end
end

function (post_callback::PostprocessCallback)(integrator)
    if isempty(post_callback.dt) && isempty(post_callback.values)
        return nothing
    end

    series_data = prepare_series_data(post_callback)
    filename = get_unique_filename("values", ".json")

    open(filename, "w") do file
        JSON.print(file, series_data, 4)
    end
end

function get_unique_filename(base_name, extension)
    filename = base_name * extension
    counter = 1

    while isfile(filename)
        filename = base_name * string(counter) * extension
        counter += 1
    end

    return filename
end

function prepare_series_data(post_callback)
    series_data = Dict("dt" => create_dict(post_callback.dt))

    for (key, data_array) in post_callback.values
        data_values = [data.value for data in data_array]
        data_times = [data.time for data in data_array]
        series_data[key] = create_dict(data_values, data_times)
    end

    return series_data
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
