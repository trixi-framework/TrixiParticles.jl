"""
    kinetic_energy

Returns the total kinetic energy of all particles in a system.
"""
function kinetic_energy(v, u, t, system)
    # If `each_moving_particle` is empty (no moving particles), return zero
    return sum(each_moving_particle(system), init=0.0) do particle
        velocity = current_velocity(v, system, particle)
        return 0.5 * system.mass[particle] * dot(velocity, velocity)
    end
end

"""
    total_mass

Returns the total mass of all particles in a system.
"""
function total_mass(v, u, t, system)
    return sum(eachparticle(system)) do particle
        return system.mass[particle]
    end
end

function total_mass(v, u, t, system::BoundarySystem)
    # It does not make sense to return a mass for boundary systems.
    # The material density and therefore the physical mass of the boundary is not relevant
    # when simulating a solid, stationary wall. The boundary always behaves as if it had
    # infinite mass. There is no momentum transferred to the boundary on impact.
    #
    # When the dummy particles model is used, i.e., boundary particles behave like fliud
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
function max_pressure(v, u, t, system)
    return maximum(particle -> particle_pressure(v, system, particle),
                   each_moving_particle(system))
end

"""
    min_pressure

Returns the minimum pressure over all particles in a system.
"""
function min_pressure(v, u, t, system)
    return minimum(particle -> particle_pressure(v, system, particle),
                   each_moving_particle(system))
end

"""
    avg_pressure

Returns the average pressure over all particles in a system.
"""
function avg_pressure(v, u, t, system)
    if n_moving_particles(system) == 0
        return 0.0
    end

    sum_ = sum(particle -> particle_pressure(v, system, particle),
               each_moving_particle(system))
    return sum_ / n_moving_particles(system)
end

"""
    max_density

Returns the maximum density over all particles in a system.
"""
function max_density(v, u, t, system)
    return maximum(particle -> particle_density(v, system, particle),
                   each_moving_particle(system))
end

"""
    min_density

Returns the minimum density over all particles in a system.
"""
function min_density(v, u, t, system)
    return minimum(particle -> particle_density(v, system, particle),
                   each_moving_particle(system))
end

"""
    avg_density

Returns the average_density over all particles in a system.
"""
function avg_density(v, u, t, system)
    if n_moving_particles(system) == 0
        return 0.0
    end

    sum_ = sum(particle -> particle_density(v, system, particle),
               each_moving_particle(system))
    return sum_ / n_moving_particles(system)
end
