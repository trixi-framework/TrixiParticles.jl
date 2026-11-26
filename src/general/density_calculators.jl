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

# `system` provides the particles and the neighbor search. `kernel_data` provides the smoothing
# kernel and the smoothing length. It defaults to `system`; boundary models pass themselves,
# so that structures use the kernel and hydrodynamic neighborhood of their boundary model.
function summation_density!(system, semi, u, u_ode, density;
                            particles=each_integrated_particle(system), kernel_data=system)
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

            # Loop over all pairs of particles and neighbors within the kernel cutoff.
            foreach_point_neighbor(system_coords, neighbor_coords, neighborhood_search;
                                   points=particles,
                                   parallelization_backend=semi.parallelization_backend) do particle,
                                                                                            neighbor,
                                                                                            pos_diff,
                                                                                            distance
                mass = hydrodynamic_mass(neighbor_system, neighbor)
                density[particle] += mass *
                                     smoothing_kernel(kernel_data, distance, particle)
            end
        end
    end
end

@inline density_calculator(system) = system.density_calculator
