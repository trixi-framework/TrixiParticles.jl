# Inspired by [Trixi.jl](https://github.com/trixi-framework/Trixi.jl).
"""
    SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6)

Terminates the integration when the [`residual_steady_state(du, equations)`](@ref)
falls below the threshold specified by `abstol, reltol`.
"""
mutable struct SteadyStateCallback{RealT <: Real}
    abstol::RealT
    reltol::RealT
    previous_ekin::Vector{Float64}
    interval_size::Int
end

function SteadyStateCallback(; abstol=1.0e-8, reltol=1.0e-6, interval_size::Integer=10)
    abstol, reltol = promote(abstol, reltol)

    steady_state_callback = SteadyStateCallback(abstol, reltol, [Inf64], interval_size)

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

# affect!
(::SteadyStateCallback)(integrator) = terminate!(integrator)

# the condition
function (steady_state_callback::SteadyStateCallback)(vu_ode, t, integrator)
    (; abstol, reltol, previous_ekin, interval_size) = steady_state_callback
    v_ode, u_ode = vu_ode.x
    semi = integrator.p

    # Calculate kinetic energy
    ekin = 0.0
    foreach_system(semi) do system
        v = wrap_v(v_ode, system, semi)

        for particle in each_moving_particle(system)
            velocity = current_velocity(v, system, particle)
            ekin += 0.5 * system.mass[particle] * dot(velocity, velocity)
        end
    end

    terminate = false

    if integrator.stats.naccept > interval_size
        popfirst!(previous_ekin)

        # Calculate MSE only over the `interval_size`
        mse = 0.0
        for index in 1:interval_size
            mse += (previous_ekin[index] - ekin)^2
        end
        mse /= interval_size

        threshold = abstol + reltol * ekin

        terminate = mse <= threshold
    end

    !(terminate) && push!(previous_ekin, ekin)

    return terminate
end
