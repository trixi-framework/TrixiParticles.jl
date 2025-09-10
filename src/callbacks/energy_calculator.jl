struct EnergyCalculatorCallback{T}
    interval :: Int
    t        :: T
    energy   :: T
end

function EnergyCalculatorCallback{ELTYPE}(; interval=1) where {ELTYPE <: Real}
    cb = EnergyCalculatorCallback(interval, Ref(zero(ELTYPE)), Ref(zero(ELTYPE)))

    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(cb, cb,
                            save_positions=(false, false))
end

# `condition`
function (update_callback!::EnergyCalculatorCallback)(u, t, integrator)
    (; interval) = update_callback!

    return condition_integrator_interval(integrator, interval)
end

# `affect!`
function (callback::EnergyCalculatorCallback)(integrator)
    # Determine time step size as difference to last time this callback was called
    t = integrator.t
    dt = t - callback.t[]

    # Update time of last call
    callback.t[] = t

    semi = integrator.p
    v_ode, u_ode = integrator.u.x
    energy = callback.energy

    update_energy_calculator!(energy, v_ode, u_ode, semi.systems[1], semi, t, dt)

    # Tell OrdinaryDiffEq that `u` has not been modified
    u_modified!(integrator, false)

    return integrator
end

function update_energy_calculator!(energy, v_ode, u_ode,
                                   system::SolidSystem, semi, t, dt)
    @trixi_timeit timer() "calculate energy" begin
        # Update quantities that are stored in the systems. These quantities (e.g. pressure)
        # still have the values from the last stage of the previous step if not updated here.
        @trixi_timeit timer() "update systems and nhs" begin
            # Don't create sub-timers here to avoid cluttering the timer output
            @notimeit timer() update_systems_and_nhs(v_ode, u_ode, semi, t)
        end

        dv_fixed = system.cache.dv_fixed
        set_zero!(dv_fixed)

        dv_ = zero(wrap_v(v_ode, system, semi))
        dv = extend_dv(system, dv_)
        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)
        eachparticle = (n_moving_particles(system) + 1):nparticles(system)

        foreach_system(semi) do neighbor_system
            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)

            interact!(dv, v, u, v_neighbor, u_neighbor,
                      system, neighbor_system, semi,
                      eachparticle=eachparticle)
        end

        @threaded semi for particle in eachparticle
            # Dispatch by system type to exclude boundary systems
            add_acceleration!(dv, particle, system)
            add_source_terms_inner!(dv, v, u, particle, system, source_terms(system), t)
        end

        n_fixed_particles = nparticles(system) - n_moving_particles(system)
        for fixed_particle in 1:n_fixed_particles
            particle = fixed_particle + n_moving_particles(system)
            velocity = current_velocity(nothing, system, particle)
            dv_particle = extract_svector(dv_fixed, system, fixed_particle)

            # The force on the fixed particle is mass times acceleration
            F_particle = system.mass[particle] * dv_particle

            # To obtain energy, we need to integrate the instantaneous power.
            # Instantaneous power is force done BY the particle times prescribed velocity.
            # The work done BY the particle is the negative of the work done ON it.
            energy[] -= dot(F_particle, velocity) * dt
        end
    end
end

# function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})
#     @nospecialize cb # reduce precompilation time
#     print(io, "EnergyCalculatorCallback(interval=", cb.affect!.interval, ")")
# end

# function Base.show(io::IO, ::MIME"text/plain",
#                    cb::DiscreteCallback{<:Any, <:EnergyCalculatorCallback})
#     @nospecialize cb # reduce precompilation time

#     if get(io, :compact, false)
#         show(io, cb)
#     else
#         update_cb = cb.affect!
#         setup = [
#             "interval" => update_cb.interval
#         ]
#         summary_box(io, "EnergyCalculatorCallback", setup)
#     end
# end
