"""
    SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the residual falls below the threshold
specified by `abstol, reltol`.
"""
mutable struct SteadyStateCallback{RealT <: Real, F}
    abstol           :: RealT
    reltol           :: RealT
    previous_ekin    :: Vector{Float64}
    interval_size    :: Int
    exclude_boundary :: Bool
    data             :: Dict{String, Vector{Any}}
    filename         :: String
    output_directory :: String
    func             :: F
end

function SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6, interval_size::Integer=10,
                             exclude_boundary=true, output_directory="out",
                             filename="values", funcs...)
    abstol, reltol = promote(abstol, reltol)

    steady_state_callback = SteadyStateCallback(abstol, reltol, [Inf64], interval_size,
                                                exclude_boundary,
                                                Dict{String, Vector{Any}}(),
                                                filename, output_directory, funcs)

    DiscreteCallback(steady_state_callback, steady_state_callback,
                     save_positions=(false, false))
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    steady_state_callback = cb.affect!
    print(io, "SteadyStateCallback(abstol=", steady_state_callback.abstol, ", ",
          "reltol=", steady_state_callback.reltol, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SteadyStateCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        steady_state_callback = cb.affect!

        setup = ["absolute tolerance" => steady_state_callback.abstol,
            "relative tolerance" => steady_state_callback.reltol]
        summary_box(io, "SteadyStateCallback", setup)
    end
end

# `affect!`
(::SteadyStateCallback)(integrator) = terminate!(integrator)

# `condition`
function (steady_state_callback::SteadyStateCallback)(vu_ode, t, integrator)
    (; abstol, reltol, previous_ekin, interval_size,
    output_directory, filename) = steady_state_callback
    v_ode, u_ode = vu_ode.x
    semi = integrator.p

    filenames = system_names(semi.systems)

    # Calculate kinetic energy and store custom data
    ekin = 0.0
    foreach_system(semi) do system
        if system isa BoundarySystem && steady_state_callback.exclude_boundary

            # Exclude boundary systems
            return
        end

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        # Calculate kintetic energy
        for particle in each_moving_particle(system)
            velocity = current_velocity(v, system, particle)
            ekin += 0.5 * system.mass[particle] * dot(velocity, velocity)
        end

        system_index = system_indices(system, semi)
        # Store custom data
        for (key, f) in steady_state_callback.func
            result = f(v, u, t, system)
            if result !== nothing
                add_entry!(steady_state_callback, string(key), t, result,
                           filenames[system_index])
            end
        end
    end

    push!(previous_ekin, ekin)

    terminate = false

    if integrator.stats.naccept > interval_size
        popfirst!(previous_ekin)

        for key in keys(steady_state_callback.data)
            popfirst!(steady_state_callback.data[key])
        end

        # Calculate MSE only over the `interval_size`
        mse = 0.0
        for index in 1:interval_size
            mse += (previous_ekin[index] - ekin)^2
        end

        mse /= interval_size

        threshold = abstol + reltol * ekin

        terminate = mse <= threshold
    end

    # Write data
    if terminate || isfinished(integrator)
        abs_file_path = joinpath(abspath(output_directory), filename)

        for (key, values) in steady_state_callback.data
            values_ = stack(values)

            df = DataFrame(values_, [Symbol(key * "_$i") for i in 1:interval_size])

            df[!, Symbol(key * "_avg")] = [sum(values_, dims=2)...] ./ interval_size

            # Write the DataFrame to a CSV file
            CSV.write(abs_file_path * "_" * key * ".csv", df)
        end
    end

    return terminate
end
