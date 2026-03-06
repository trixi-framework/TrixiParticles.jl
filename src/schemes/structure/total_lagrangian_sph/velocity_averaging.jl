struct VelocityAveraging{ELTYPE}
    time_constant::ELTYPE
end

# No velocity averaging for a system by default
@inline function velocity_averaging(system)
    return nothing
end

@inline function velocity_for_viscosity(v, system, particle)
    return velocity_for_viscosity(v, velocity_averaging(system), system, particle)
end

@inline function velocity_for_viscosity(v, ::Nothing, system, particle)
    return current_velocity(v, system, particle)
end

@inline function velocity_for_viscosity(v, ::VelocityAveraging, system, particle)
    if particle <= system.n_integrated_particles
        return extract_svector(system.cache.averaged_velocity, system, particle)
    end

    return current_clamped_velocity(v, system, system.clamped_particles_motion, particle)
end

function initialize_averaged_velocity!(system, v_ode, semi, t)
    initialize_averaged_velocity!(system, velocity_averaging(system), v_ode, semi, t)
end

initialize_averaged_velocity!(system, velocity_averaging, v_ode, semi, t) = system

function initialize_averaged_velocity!(system, ::VelocityAveraging, v_ode, semi, t)
    v = wrap_v(v_ode, system, semi)

    # Make sure the averaged velocity is zero for non-integrated particles
    set_zero!(system.cache.averaged_velocity)

    @threaded semi for particle in each_integrated_particle(system)
        v_particle = current_velocity(v, system, particle)

        for i in eachindex(v_particle)
            system.cache.averaged_velocity[i, particle] = v_particle[i]
        end
    end
    system.cache.t_last_averaging[] = t

    return system
end

function compute_averaged_velocity!(system, v_ode, semi, t_new)
    compute_averaged_velocity!(system, velocity_averaging(system), v_ode, semi, t_new)
end

compute_averaged_velocity!(system, velocity_averaging, v_ode, semi, t_new) = system

function compute_averaged_velocity!(system, velocity_averaging::VelocityAveraging,
                                    v_ode, semi, t_new)
    time_constant = velocity_averaging.time_constant
    averaged_velocity = system.cache.averaged_velocity
    dt = t_new - system.cache.t_last_averaging[]
    system.cache.t_last_averaging[] = t_new

    @trixi_timeit timer() "compute averaged velocity" begin
        v = wrap_v(v_ode, system, semi)

        # Compute an exponential moving average of the velocity with the given time constant
        alpha = 1 - exp(-dt / time_constant)

        @threaded semi for particle in each_integrated_particle(system)
            v_particle = current_velocity(v, system, particle)

            for i in eachindex(v_particle)
                averaged_velocity[i, particle] = (1 - alpha) * averaged_velocity[i, particle] + alpha * v_particle[i]
            end
        end
    end

    return system
end
