using JSON

struct DataEntry
    value::Float64
    time::Float64
    system::String
end

"""
    PostprocessCallback
This struct holds the values and time deltas (dt) collected during the simulation. The `values` field is a dictionary
that maps a string key to an array of `DataEntry` structs. The `dt` field is an array of time deltas.
"""
mutable struct PostprocessCallback{I, F}
    interval::I
    last_t::Float64
    values::Dict{String, Vector{DataEntry}}
    exclude_bnd::Bool
    func::F
end

function PostprocessCallback(func; interval=0, exclude_bnd=true)
    post_callback = PostprocessCallback(interval, -Inf, Dict{String, Vector{DataEntry}}(), exclude_bnd, func)

    DiscreteCallback(post_callback, post_callback,
                     save_positions=(false, false),
                     initialize=initialize_post_callback!)
end



function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    callback = cb.affect!
    print(io, "PostprocessCallback(interval=", callback.interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:PostprocessCallback})
    @nospecialize cb # reduce precompilation time
    if get(io, :compact, false)
        show(io, cb)
    else
        callback = cb.affect!
        setup = [
            "interval" => callback.interval,
        ]
        summary_box(io, "PostprocessCallback", setup)
    end
end

function initialize_post_callback!(cb, u, t, integrator)
    initialize_post_callback!(cb.affect!, u, t, integrator)
end

function initialize_post_callback!(cb::PostprocessCallback, u, t, integrator)
    cb.last_t = t
    return nothing
end

# condition with interval
function (pp::PostprocessCallback{Int, Any})(u, t, integrator)
    (; interval) = pp

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && ((integrator.stats.naccept % interval == 0) &&
            !(integrator.stats.naccept == 0 && integrator.iter > 0))
end

# condition with dt
function (pp::PostprocessCallback)(u, t, integrator)
    (; interval, last_t) = pp

    return (t - last_t) > interval
end

# affect! function for a single function
function (pp::PostprocessCallback{I,F})(integrator) where {I, F<:Function}
    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t
    pp.last_t = integrator.t
    filenames = system_names(semi.systems)

    foreach_system(semi) do system
        if system isa BoundarySystem
            return
        end

        system_index = system_indices(system, semi)

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        pp.func(pp, t, system, u, v, filenames[system_index])
    end

    if isfinished(integrator)
        pp(integrator, true)
    end
end

# affect! function for an array of functions
function (pp::PostprocessCallback{I,F})(integrator) where {I, F<:Array}
    vu_ode = integrator.u
    v_ode, u_ode = vu_ode.x
    semi = integrator.p
    t = integrator.t
    pp.last_t = integrator.t
    filenames = system_names(semi.systems)

    foreach_system(semi) do system
        if system isa BoundarySystem
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
end

# This function prepares the data for writing to a JSON file by creating a dictionary
# that maps each key to separate arrays of times and values, sorted by time, and includes system name.
function prepare_series_data(post_callback)
    series_data = Dict()

    for (key, data_array) in post_callback.values
        # Sort the data_array by time
        sorted_data_array = sort(data_array, by = data -> data.time)

        # Extracting times and values into separate arrays
        data_times = [data.time for data in sorted_data_array]
        data_values = [data.value for data in sorted_data_array]

        # Assuming each DataEntry in sorted_data_array has a `system` field.
        system_name = isempty(sorted_data_array) ? "" : sorted_data_array[1].system

        series_data[key] = create_series_dict(data_values, data_times, system_name)
    end

    return series_data
end

function create_series_dict(values, times, system_name="")
    Dict("type" => "series", "datatype" => eltype(values),
         "novalues" => length(values), "values" => values,
         "time" => times, "system_name" => system_name)
end


# After the simulation has finished, this function is called to write the data to a JSON file.
function (pp::PostprocessCallback)(integrator, finished::Bool)
    if isempty(pp.values)
        return nothing
    end

    series_data = prepare_series_data(pp)
    filename = get_unique_filename("values", ".json")
    println("writing a postproccessing results to ", filename)

    open(filename, "w") do file
        JSON.print(file, series_data, 4)
    end
end

function add_entry!(pp, entry_key, t, value, system_name)
    # Get the list of DataEntry for the system, or initialize it if it doesn't exist
    entries = get!(pp.values, entry_key*"_"*system_name, DataEntry[])

    # Add the new entry to the list
    push!(entries, DataEntry(value, t, system_name))
end

function test_function(pp, t, system, u, v, system_name)
    add_entry!(pp, "test", t, 2*t, system_name)
end

function calculate_ekin(pp, t, system, u, v, system_name)
    mass = system.mass

    ekin = 0.0
    for particle in each_moving_particle(system)
        velocity = current_velocity(v, system, particle)
        ekin += 0.5 * mass[particle] * dot(velocity, velocity)
    end
    add_entry!(pp, "ekin", t, ekin, system_name)
end

function calculate_total_mass(pp, t, system, u, v, system_name)
    mass = system.mass
    total_mass = 0.0
    for particle in each_moving_particle(system)
        total_mass += mass[particle]
    end
    add_entry!(pp, "totalm", t, total_mass, system_name)
end

function max_pressure(pp, t, system, u, v, system_name)

end

function min_pressure(pp, t, system, u, v, system_name)

end

function avg_pressure(pp, t, system, u, v, system_name)

end

function max_density(pp, t, system, u, v, system_name)

end

function min_density(pp, t, system, u, v, system_name)

end

function avg_density(pp, t, system, u, v, system_name)

end

# # This function prepares the data for writing to a JSON file by creating a dictionary
# # that maps each key to a dictionary with the associated values and times.
# function prepare_series_data(post_callback)
#     series_data = Dict("dt" => create_dict(post_callback.dt))

#     for (key, data_array) in post_callback.values
#         data_values = [data.value for data in data_array]
#         data_times = [data.time for data in data_array]
#         series_data[key] = create_dict(data_values, data_times)
#     end

#     return series_data
# end

# function create_dict(values, times=nothing)
#     Dict("type" => "series", "datatype" => typeof(values),
#          "novalues" => length(values), "values" => values, "time" => times)
# end


# function postprocess_value(system, key, system_index, u, semi)
#     data_available, value = pp_value(system, key)
#     if !data_available
#         if key == "ekin"
#             value = calculate_ekin(system, system_index, u, semi)
#             data_available = true
#         end
#     end
#     return data_available, value
# end

# # This function is called at each timestep. It adds the current dt to the array and
# # checks each system for new values to add to the `values` field.
# function (post_callback::PostprocessCallback)(u, t, integrator)
#     push!(post_callback.dt, integrator.dt)

#     systems = integrator.p.systems
#     semi = integrator.p

#     foreach_enumerate(systems) do (system_index, system)
#         keys = pp_keys(system)
#         if keys !== nothing
#             for key in keys
#                 data_available, value = postprocess_value(system, key, system_index, u, semi)
#                 if data_available
#                     value_array = get!(post_callback.values, key, DataEntry[])
#                     push!(value_array, DataEntry(value, t))
#                 end
#             end
#         end
#     end
#     return isfinished(integrator)
# end



# # This function prepares the data for writing to a JSON file by creating a dictionary
# # that maps each key to a dictionary with the associated values and times.
# function prepare_series_data(post_callback)
#     series_data = Dict("dt" => create_dict(post_callback.dt))

#     for (key, data_array) in post_callback.values
#         data_values = [data.value for data in data_array]
#         data_times = [data.time for data in data_array]
#         series_data[key] = create_dict(data_values, data_times)
#     end

#     return series_data
# end

# function create_dict(values, times=nothing)
#     Dict("type" => "series", "datatype" => typeof(values),
#          "novalues" => length(values), "values" => values, "time" => times)
# end

# # function calculate_ekin(system, system_index, u, semi)
# #     return system
# # end

# function calculate_ekin(system, system_index, u, semi)
#     mass = system.mass
#     v_ode, u_ode = u.x

#     v = wrap_v(v_ode, system_index, system, semi)
#     ekin = 0.0
#     for particle in each_moving_particle(system)
#         velocity = current_velocity(v, system, particle)
#         ekin += 0.5 * mass[particle] * dot(velocity, velocity)
#     end
#     return ekin
# end

# function compute_pressure!(system, v, pp_values::Dict)
#     (; state_equation, pressure, cache) = system

#     if haskey(pp_values, "dp")
#         p_old = copy(system.pressure)
#         compute_pressure!(system, v, pressure, state_equation)
#         dp = p_old-system.pressure
#         pp_values["dp"] = sqrt(dot(dp, dp))
#         cache.pp_update["dp"] = true
#     else
#         compute_pressure!(system, v, pressure, state_equation)
#     end
#     return system
# end

# function pp_value(system, pp_key)
#     if system.pp_values !== nothing
#         if haskey(system.pp_values, pp_key)
#             return system.cache.pp_update[pp_key], system.pp_values[pp_key]
#         end
#     end

#     return false, 0.0
# end
