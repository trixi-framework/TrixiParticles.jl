@doc raw"""
    VelocityAveraging(; time_constant)

Exponential moving average (EMA) of the structure velocity used **only** for the
fluid-structure viscous coupling (no-slip boundary condition) of a
[`TotalLagrangianSPHSystem`](@ref).

This is a stabilization option that is typically only needed for challenging FSI
cases with very stiff structures, where high-frequency noise in the structure velocity
can trigger instabilities or spurious pressure waves (due to aliasing) in the fluid.

See [the docs](@ref velocity_averaging) for more details.

!!! note "Callback Required"
    Velocity averaging requires either an [`UpdateCallback`](@ref)
    or a [`SplitIntegrationCallback`](@ref) to be used in the simulation.
    In the typical use case of stabilizing a challenging FSI simulation, a
    [`SplitIntegrationCallback`](@ref) will usually be used anyway for performance reasons.

# Keywords
- `time_constant`: Time constant ``\tau`` of the EMA (same units as simulation time).
                   Smaller ``\tau`` smooths only very fast oscillations, which is often
                   sufficient for stabilizing FSI simulations; larger ``\tau``
                   can also suppress slower oscillations that are causing spurious
                   pressure waves without destabilizing the simulation.

# Examples
```jldoctest; output=false
VelocityAveraging(time_constant=1e-5)

# output
VelocityAveraging{Float64}(1.0e-5)
```
"""
struct VelocityAveraging{ELTYPE}
    time_constant::ELTYPE

    function VelocityAveraging(; time_constant)
        return new{typeof(time_constant)}(time_constant)
    end
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
