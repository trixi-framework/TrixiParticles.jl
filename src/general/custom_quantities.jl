"""
    kinetic_energy

Returns the total kinetic energy of all particles in a system.
"""
function kinetic_energy(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)

    # TODO: `current_velocity` should only contain active particles
    # (see https://github.com/trixi-framework/TrixiParticles.jl/issues/850)
    velocity = reinterpret(reshape, SVector{ndims(system), eltype(v)},
                           view(current_velocity(v, system), :,
                                each_active_particle(system)))
    mass = view(system.mass, each_active_particle(system))

    return mapreduce(+, velocity, mass) do v_i, m_i
        return m_i * dot(v_i, v_i) / 2
    end
end

function kinetic_energy(system::AbstractBoundarySystem,
                        dv_ode, du_ode, v_ode, u_ode, semi, t)
    return zero(eltype(system))
end

"""
    total_mass

Returns the total mass of all particles in a system.
"""
function total_mass(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return sum(system.mass)
end

function total_mass(system::AbstractBoundarySystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    # It does not make sense to return a mass for boundary systems.
    # The material density and therefore the physical mass of the boundary is not relevant
    # when simulating a solid, stationary wall. The boundary always behaves as if it had
    # infinite mass. There is no momentum transferred to the boundary on impact.
    #
    # When the dummy particles model is used, i.e., boundary particles behave like fluid
    # particles when interacting with actual fluid particles, the boundary particles do have
    # a "hydrodynamic mass", which corresponds to the fluid density, but this is only
    # relevant for the fluid interaction, and it has no connection to the physical mass
    # of the boundary. Returning the "hydrodynamic mass" here would thus be misleading.
    return NaN
end

"""
    max_pressure

Returns the maximum pressure over all particles in a system.
"""
function max_pressure(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return maximum(current_pressure(v, system))
end

function max_pressure(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    min_pressure

Returns the minimum pressure over all particles in a system.
"""
function min_pressure(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return minimum(current_pressure(v, system))
end

function min_pressure(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    avg_pressure

Returns the average pressure over all particles in a system.
"""
function avg_pressure(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    sum_ = sum(current_pressure(v, system))
    return sum_ / nparticles(system)
end

function avg_pressure(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    max_density

Returns the maximum density over all particles in a system.
"""
function max_density(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return maximum(current_density(v, system))
end

function max_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    min_density

Returns the minimum density over all particles in a system.
"""
function min_density(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    return minimum(current_density(v, system))
end

function min_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end

"""
    avg_density

Returns the average_density over all particles in a system.
"""
function avg_density(system::AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode, semi, t)
    v = wrap_v(v_ode, system, semi)
    sum_ = sum(current_density(v, system))
    return sum_ / nparticles(system)
end

function avg_density(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
    return NaN
end
