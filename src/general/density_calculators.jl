@doc raw"""
    SummationDensity()

Density calculator to use the summation formula
```math
\rho(r) = \sum_{b} m_b W(\Vert r - r_b \Vert, h),
```
for the density estimation,
where ``r_b`` denotes the coordinates and ``m_b`` the mass of particle ``b``.
"""
struct SummationDensity end

@doc raw"""
    ContinuityDensity()

Density calculator to integrate the density from the continuity equation
```math
\frac{\mathrm{d}\rho_a}{\mathrm{d}t} = \sum_{b} m_b v_{ab} \cdot \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
```
where ``\rho_a`` denotes the density of particle ``a`` and ``r_{ab} = r_a - r_b`` is the
difference of the coordinates, ``v_{ab} = v_a - v_b`` of the velocities of particles
``a`` and ``b``.
"""
struct ContinuityDensity end

@propagate_inbounds function particle_density(v, system, particle)
    particle_density(v, system.density_calculator, system, particle)
end

@propagate_inbounds function particle_density(v, ::SummationDensity, system, particle)
    return system.cache.density[particle]
end

@propagate_inbounds function particle_density(v, ::ContinuityDensity, system, particle)
    return v[end, particle]
end

# WARNING!
# These functions are intended to be used internally to set the density
# of newly activated particles in a callback.
# DO NOT use outside a callback. OrdinaryDiffEq does not allow changing `v` and `u`
# outside of callbacks.
@inline set_particle_density!(v, system, ::SummationDensity, particle, density) = v

@inline function set_particle_density!(v, system, ::ContinuityDensity, particle, density)
    v[end, particle] = density

    return v
end

function summation_density!(system, semi, u, u_ode, density;
                            particles=each_moving_particle(system))
    set_zero!(density)

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute density" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        nhs = get_neighborhood_search(system, neighbor_system, semi)

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords, nhs,
                               points=particles) do particle, neighbor,
                                                    pos_diff, distance
            mass = hydrodynamic_mass(neighbor_system, neighbor)
            density[particle] += mass * smoothing_kernel(system, distance)
        end
    end
end
