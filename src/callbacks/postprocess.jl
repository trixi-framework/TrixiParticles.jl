mutable struct PostprocessCallback
    values::Dict{String, Any}
    dt::Vector{Float64}
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
    # save dt
    push!(post_callback.dt, integrator.dt)

    return isfinished(integrator)
end

# affect!
function (post_callback::PostprocessCallback)(integrator)
    # write json file with stored values
    data = Dict("series"=> Dict("name"=> "dt", "datatype" => typeof(post_callback.dt),
                "novalues" => length(post_callback.dt), "values"=>post_callback.dt))
    open("values.json", "w") do file
        # write intended with 4 spaces
        JSON.print(file, data, 4)
    end

    return nothing
end

function initialize_post_callback(discrete_callback, u, t, integrator)

    return nothing
end
