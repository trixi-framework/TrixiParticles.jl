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
        if keys !== nothing
            for key in keys
                data_available, value = pp_value(system, key)
                if data_available
                    value_array = get!(post_callback.values, key, DataEntry[])
                    push!(value_array, DataEntry(value, t))
                end
            end
        end
    end
    return isfinished(integrator)
end

function (post_callback::PostprocessCallback)(integrator)
    if isempty(post_callback.dt) && isempty(post_callback.values)
        return nothing
    end

    series_data = Dict("dt" => create_dict(post_callback.dt))
    for (key, data_array) in post_callback.values
        series_data[key] = create_dict([data.value for data in data_array], [data.time for data in data_array])
    end

    # Check if file exists and append ascending number if it does
    filename = "values.json"
    counter = 1
    while isfile(filename)
        filename = "values$(counter).json"
        counter += 1
    end

    open(filename, "w") do file
        JSON.print(file, series_data, 4)
    end
    return nothing
end

function create_dict(values, times=nothing)
    return Dict(
        "type" => "series",
        "datatype" => typeof(values),
        "novalues" => length(values),
        "values" => values,
        "time" => times
    )
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
