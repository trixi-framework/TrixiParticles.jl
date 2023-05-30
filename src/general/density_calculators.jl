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
where ``\rho_a`` denotes the density of particle ``a``, ``r_a`` and ``r_b`` denote the coordinates
of particles ``a`` and ``b`` respectively, and ``v_{ab} = v_a - v_b`` is the difference of the
velocities of particles ``a`` and ``b``.
"""
struct ContinuityDensity end

@inline function particle_density(v, system, particle)
    particle_density(v, system.density_calculator, system, particle)
end

@inline function particle_density(v, ::SummationDensity, system, particle)
    return system.cache.density[particle]
end

@inline function particle_density(v, ::ContinuityDensity, system, particle)
    return v[end, particle]
end

# *Note* that these functions are intended to internally set the density for buffer particles
# and density correction. It cannot be used to set up an initial condition,
# as the particle density depends on the particle positions.
@inline function set_particle_density(particle, v, system, density)
    set_particle_density(particle, v, system.density_calculator, system, density)
end

@inline function set_particle_density(particle, v, ::SummationDensity, system, density)
    system.cache.density[particle] = density
end

@inline function set_particle_density(particle, v, ::ContinuityDensity, system, density)
    v[end, particle] = density
end

compute_density!(system, system_index, semi, u, u_ode, ::ContinuityDensity) = system

function summation_density!(system, system_index, semi, u, u_ode, density;
                            particles=each_moving_particle(system))
    @unpack systems, neighborhood_searches = semi

    density .= zero(eltype(density))

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute density" foreach_enumerate(systems) do (neighbor_system_index,
                                                                           neighbor_system)
        u_neighbor_system = wrap_u(u_ode, neighbor_system_index,
                                   neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        neighborhood_search = neighborhood_searches[system_index][neighbor_system_index]

        # Loop over all pairs of particles and neighbors within the kernel cutoff.
        for_particle_neighbor(system, neighbor_system,
                              system_coords, neighbor_coords,
                              neighborhood_search,
                              particles=particles) do particle, neighbor, pos_diff, distance
            mass = hydrodynamic_mass(neighbor_system, neighbor)
            density[particle] += mass * smoothing_kernel(system, distance)
        end
    end
end
