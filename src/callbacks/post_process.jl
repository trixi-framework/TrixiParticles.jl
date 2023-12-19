using JSON

struct DataEntry
    value::Float64
    time::Float64
end

"""
    PostprocessCallback
This struct holds the values and time deltas (dt) collected during the simulation. The `values` field is a dictionary
that maps a string key to an array of `DataEntry` structs. The `dt` field is an array of time deltas.
"""
mutable struct PostprocessCallback
    values::Dict{String, Vector{DataEntry}}
    dt::Vector{Float64}
end

function PostprocessCallback(system; interval=0)
    post_callback = PostprocessCallback(Dict(), Float64[])

    DiscreteCallback(post_callback, post_callback,
                     save_positions=(false, false),
                     initialize=initialize_post_callback)
end

function postprocess_value(system, key, system_index, u, semi)
    data_available, value = pp_value(system, key)
    if !data_available
        if key == "ekin"
            value = calculate_ekin(system, system_index, u, semi)
            data_available = true
        end
    end
    return data_available, value
end

# This function is called at each timestep. It adds the current dt to the array and
# checks each system for new values to add to the `values` field.
function (post_callback::PostprocessCallback)(u, t, integrator)
    push!(post_callback.dt, integrator.dt)

    systems = integrator.p.systems
    semi = integrator.p

    foreach_enumerate(systems) do (system_index, system)
        keys = pp_keys(system)
        if keys !== nothing
            for key in keys
                data_available, value = postprocess_value(system, key, system_index, u, semi)
                if data_available
                    value_array = get!(post_callback.values, key, DataEntry[])
                    push!(value_array, DataEntry(value, t))
                end
            end
        end
    end
    return isfinished(integrator)
end

# After the simulation has finished, this function is called to write the data to a JSON file.
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

# This function creates a unique filename by appending a number to the base name if needed.
function get_unique_filename(base_name, extension)
    filename = base_name * extension
    counter = 1

    while isfile(filename)
        filename = base_name * string(counter) * extension
        counter += 1
    end

    return filename
end

# This function prepares the data for writing to a JSON file by creating a dictionary
# that maps each key to a dictionary with the associated values and times.
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

function calculate_ekin(system, system_index, u, semi)
    return system
end

function calculate_ekin(system::WeaklyCompressibleSPHSystem, system_index, u, semi)
    @unpack mass = system
    v_ode, u_ode = u.x

    v = wrap_v(v_ode, system_index, system, semi)
    ekin = 0.0
    for particle in each_moving_particle(system)
        velocity = current_velocity(v, system, particle)
        ekin += 0.5 * mass[particle] * dot(velocity, velocity)
    end
    return ekin
end

function compute_pressure!(system, v, pp_values::Dict)
    @unpack state_equation, pressure, cache = system

    if haskey(pp_values, "dp")
        p_old = copy(system.pressure)
        compute_pressure!(system, v, pressure, state_equation)
        dp = p_old-system.pressure
        pp_values["dp"] = sqrt(dot(dp, dp))
        cache.pp_update["dp"] = true
    else
        compute_pressure!(system, v, pressure, state_equation)
    end
    return system
end

function pp_value(system::WeaklyCompressibleSPHSystem, pp_key)
    if system.pp_values !== nothing
        if haskey(system.pp_values, pp_key)
            return system.cache.pp_update[pp_key], system.pp_values[pp_key]
        end
    end

    return false, 0.0
end
