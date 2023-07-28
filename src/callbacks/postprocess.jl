mutable struct PostprocessCallback
    values::Dict{String, Any}
    dt::Vector{Float64}
end

struct data_entry
    value::Float64
    time::Float64
end

"""
PostprocessCallback()

Write a series of values for every timestep.
"""
function PostprocessCallback(; interval=0)
    post_callback = PostprocessCallback(Dict(), Float64[])

    DiscreteCallback(post_callback, post_callback,
                     save_positions=(false, false),
                     initialize=initialize_post_callback)
end

# condition
function (post_callback::PostprocessCallback)(u, t, integrator)
    vu_ode = integrator.u
    semi = integrator.p

    # save dt
    push!(post_callback.dt, integrator.dt)

    #save dp
    data_available = pressure_change(vu_ode, semi)
    if data_available[1]
        dp_array = get(post_callback.values, "dp", data_entry[])
        push!(dp_array, data_entry(data_available[2], t))
        post_callback.values["dp"] = dp_array
        # dp_t_array = get(post_callback.values, "dp_t", Float64[])
        # push!(dp_t_array, t)
        # post_callback.values["dp_t"] = dp_t_array
    end

    return isfinished(integrator)
end

# affect!
function (post_callback::PostprocessCallback)(integrator)
    # Create a dictionary to store all series data
    series_data = Dict()

    # Store the "dt" series data
    series_data["dt"] = Dict(
        "type" => "series",
        "datatype" => typeof(post_callback.dt),
        "novalues" => length(post_callback.dt),
        "values" => post_callback.dt
    )

    # Store other series data
    #for key in keys(post_callback.values)
    for (key, data_array) in post_callback.values
        # data_array = post_callback.values[key]
        values = [data.value for data in data_array]
        times = [data.time for data in data_array]
        series_data[key] = Dict(
            "type" => "series",
            "datatype" => typeof(post_callback.values[key]),
            "novalues" => length(post_callback.values[key]),
            "values" => values,
            "time" => times
        )
    end

    # Write the entire dictionary to the JSON file
    open("values.json", "w") do file
        # write intended with 4 spaces
        JSON.print(file, series_data, 4)
    end

    return nothing
end

function initialize_post_callback(discrete_callback, u, t, integrator)

    return nothing
end

function pressure_change(vu_ode, semi)
    @unpack systems = semi

    # Search for the first system with a valid dp value
    system_with_dp = findfirst(system -> pressure_change(system)[1], systems)

    if system_with_dp !== nothing
        # If a system with a valid dp value is found, return true and its dp value
        dp_value = pressure_change(systems[system_with_dp])[2]
        #delete!(systems[system_with_dp].pp_values, "dp")
        return true, dp_value
    else
        # If no system with a valid dp value is found, return false
        return false, 0.0
    end
end

function pressure_change(system)
    return false, 0.0
end
