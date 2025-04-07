function ParticleShiftingCallback()
    # The first one is the `condition`, the second the `affect!`
    return DiscreteCallback(particle_shifting_condition, particle_shifting!,
                            initialize=initial_shifting!,
                            save_positions=(false, false))
end

# `condition`
function particle_shifting_condition(u, t, integrator)
    return true
end

# `affect!`
function particle_shifting!(integrator)
    t = integrator.t
    semi = integrator.p
    v_ode, u_ode = integrator.u.x
    dt = integrator.dt
    # Internal cache vector, which is safe to use as temporary array
    u_cache = first(get_tmp_cache(integrator))

    # Update quantities that are stored in the systems. These quantities (e.g. pressure)
    # still have the values from the last stage of the previous step if not updated here.
    update_systems_and_nhs(v_ode, u_ode, semi, t; update_from_callback=true)

    @trixi_timeit timer() "particle shifting" foreach_system(semi) do system
        u = wrap_u(u_ode, system, semi)
        v = wrap_v(v_ode, system, semi)
        particle_shifting!(u, v, system, v_ode, u_ode, semi, u_cache, dt)
    end

    # Tell OrdinaryDiffEq that `u` has been modified
    u_modified!(integrator, true)

    return integrator
end

# `initialize`
function initial_shifting!(cb::DiscreteCallback{<:Any, typeof(particle_shifting!)}, u, t,
                           integrator)
    # particle_shifting!(integrator)
end

function particle_shifting!(u, v, system, v_ode, u_ode, semi, u_cache, dt)
    return u
end

function particle_shifting!(u, v, system::WeaklyCompressibleSPHSystem, v_ode, u_ode, semi,
                            u_cache, dt)
    # According to Sun et al. (2017), CFL * Ma can be rewritten as
    # Δt * v_max / h (see p. 29, right above eq. 9).
    v_max = maximum(particle -> norm(current_velocity(v, system, particle)),
                    eachparticle(system))
    R = 0.2
    n = 4

    delta_r = wrap_u(u_cache, system, semi)
    set_zero!(delta_r)

    foreach_system(semi) do neighbor_system
        u_neighbor = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor = wrap_v(v_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi) do particle, neighbor, pos_diff, distance
            Wdx = smoothing_kernel(system, particle_spacing(system, 1), 1)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            rho_a = particle_density(v, system, particle)
            rho_b = particle_density(v_neighbor, neighbor_system, neighbor)
            h = smoothing_length(system, particle)

            kernel = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            delta_r_ = -dt * v_max * 4 * (h / 2) * (1 + R * (kernel / Wdx)^n) * m_b /
                       (rho_a + rho_b) *
                       grad_kernel

            for i in eachindex(delta_r_)
                @inbounds delta_r[i, particle] += delta_r_[i]
            end
        end
    end

    @threaded semi for particle in eachparticle(system)
        for i in 1:ndims(system)
            @inbounds u[i, particle] += delta_r[i, particle]
        end
    end

    return u
end

# function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ParticleShiftingCallback})
#     @nospecialize cb # reduce precompilation time
#     print(io, "ParticleShiftingCallback(interval=", cb.affect!.interval, ")")
# end

# function Base.show(io::IO,
#                    cb::DiscreteCallback{<:Any,
#                                         <:PeriodicCallbackAffect{<:ParticleShiftingCallback}})
#     @nospecialize cb # reduce precompilation time
#     print(io, "ParticleShiftingCallback(dt=", cb.affect!.affect!.interval, ")")
# end

# function Base.show(io::IO, ::MIME"text/plain",
#                    cb::DiscreteCallback{<:Any, <:ParticleShiftingCallback})
#     @nospecialize cb # reduce precompilation time

#     if get(io, :compact, false)
#         show(io, cb)
#     else
#         update_cb = cb.affect!
#         setup = [
#             "interval" => update_cb.interval
#         ]
#         summary_box(io, "ParticleShiftingCallback", setup)
#     end
# end

# function Base.show(io::IO, ::MIME"text/plain",
#                    cb::DiscreteCallback{<:Any,
#                                         <:PeriodicCallbackAffect{<:ParticleShiftingCallback}})
#     @nospecialize cb # reduce precompilation time

#     if get(io, :compact, false)
#         show(io, cb)
#     else
#         update_cb = cb.affect!.affect!
#         setup = [
#             "dt" => update_cb.interval
#         ]
#         summary_box(io, "ParticleShiftingCallback", setup)
#     end
# end
