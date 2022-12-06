mutable struct MoveParticleCallback
    start_intervall::Float64
    callback_interval::Int
end


function MoveParticleCallback(; callback_interval=0)
    move_callback = MoveParticleCallback(0.0, callback_interval)

    DiscreteCallback(move_callback, move_callback,
                     save_positions=(false, false))
end


# condition
function (move_callback::MoveParticleCallback)(u, t, integrator)
    @unpack callback_interval = move_callback

    return callback_interval == 0 ||
        integrator.destats.naccept % callback_interval == 0 || t==0.0
end

# affect!
function (move_callback::MoveParticleCallback)(integrator)
    semi = integrator.p
    @unpack particle_containers = semi
    container = particle_containers[3]
    t = integrator.t
    f(t) = -285.115*t^3 + 72.305*t^2 + 0.1463*t

    container.current_coordinates[2,:] .+= f(t)
    # Tell OrdinaryDiffEq that u has not been modified
    u_modified!(integrator, false)
    return nothing
end
