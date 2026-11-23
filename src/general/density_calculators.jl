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

@inline ignore_density_neighbor(particle, weight) = nothing

# `system` provides the particles and the neighbor search. `kernel_data` provides the smoothing
# kernel and the smoothing length. It defaults to `system`; boundary models pass themselves,
# so that structures use the kernel and hydrodynamic neighborhood of their boundary model.
# `neighbor_callback(neighbor_system)` optionally supplies an action for each kernel sample.
function summation_density!(system, semi, u, u_ode, density;
                            particles=each_integrated_particle(system), kernel_data=system,
                            neighbor_callback=Returns(ignore_density_neighbor))
    set_zero!(density)

    # Use enabled neighbor systems for the density summation.
    @trixi_timeit timer() "compute density" begin
        foreach_system(semi) do neighbor_system
            has_hydrodynamic_interaction(system, neighbor_system, semi) || return

            u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

            system_coords = current_coordinates(u, system)
            neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
            neighborhood_search = get_summation_neighborhood_search(kernel_data, system,
                                                                    neighbor_system, semi)
            on_neighbor = neighbor_callback(neighbor_system)
            sum_neighbor_density!(on_neighbor, density, neighbor_system, kernel_data,
                                  system_coords, neighbor_coords, neighborhood_search,
                                  particles, semi.parallelization_backend)
        end
    end
end

# Specialize the hot loop on the callback, which may differ between neighbor systems.
@inline function sum_neighbor_density!(on_neighbor::F, density, neighbor_system,
                                       kernel_data,
                                       system_coords, neighbor_coords, neighborhood_search,
                                       particles, parallelization_backend) where {F}
    foreach_point_neighbor(system_coords, neighbor_coords, neighborhood_search;
                           points=particles,
                           parallelization_backend) do particle, neighbor,
                                                       pos_diff, distance
        mass = hydrodynamic_mass(neighbor_system, neighbor)
        weight = smoothing_kernel(kernel_data, distance, particle)
        density[particle] += mass * weight
        on_neighbor(particle, weight)
    end
end

@inline density_calculator(system) = system.density_calculator
