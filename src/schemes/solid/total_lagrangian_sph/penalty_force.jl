@doc raw"""
    PenaltyForceGanzenmueller(; alpha=0.1)

Penalty force to ensure regular particle positions under large deformations.

In FEM, underintegrated elements can deform without an associated increase of energy.
This is caused by the stiffness matrix having zero eigenvalues (so-called hourglass modes).
The name "hourglass modes" comes from the fact that elements can deform into an hourglass shape.

Similar effects can occur in SPH as well.
Particles can change positions without changing the SPH approximation of the deformation gradient $\bm{J}$,
thus, without causing an increase of energy.
To ensure regular particle positions, we can apply similar correction forces as are used in FEM.

(Ganzenmüller, 2015) introduced a so-called hourglass correction force or penalty force $f^{PF}$,
which is given by
```math
\bm{f}_a^{PF} = \frac{1}{2} \alpha \sum_b \frac{m_{0a} m_{0b} W_{0ab}}{\rho_{0a}\rho_{0b} |\bm{X}_{ab}|^2}
                \left( E \delta_{ab}^a + E \delta_{ba}^b \right) \frac{\bm{x}_{ab}}{|\bm{x}_{ab}|}
```
The subscripts $a$ and $b$ denote quantities of particle $a$ and $b$, respectively.
The zero subscript on quantities denotes that the quantity is to be measured in the initial configuration.
The difference in the initial coordinates is denoted by $\bm{X}_{ab} = \bm{X}_a - \bm{X}_b$,
the difference in the current coordinates is denoted by $\bm{x}_{ab} = \bm{x}_a - \bm{x}_b$.
Note that (Ganzenmüller, 2015) has a flipped sign here because they define $\bm{x}_{ab}$ the other way around.

This correction force is based on the potential energy density of a Hookean material.
Thus, $E$ is the Young's modulus and $\alpha$ is a dimensionless coefficient that controls
the amplitude of hourglass correction.
The separation vector $\delta_{ab}^a$ indicates the change of distance which the particle separation should attain
in order to minimize the error and is given by
```math
    \delta_{ab}^a = \frac{\bm{\epsilon}_{ab}^a \cdot \bm{x_{ab}}}{|\bm{x}_{ab}|},
```
where the error vector is defined as
```math
    \bm{\epsilon}_{ab}^a = \bm{J}_a \bm{X}_{ab} - \bm{x}_{ab}.
```


## References:
- Georg C. Ganzenmüller.
  "An hourglass control algorithm for Lagrangian Smooth Particle Hydrodynamics".
  In: Computer Methods in Applied Mechanics and Engineering 286 (2015).
  [doi: 10.1016/j.cma.2014.12.005](https://doi.org/10.1016/j.cma.2014.12.005)
"""
struct PenaltyForceGanzenmueller{ELTYPE}
    alpha::ELTYPE
    function PenaltyForceGanzenmueller(; alpha=0.1)
        new{typeof(alpha)}(alpha)
    end
end

@inline function calc_penalty_force!(dv, particle, neighbor, initial_pos_diff,
                                     initial_distance, system,
                                     penalty_force::PenaltyForceGanzenmueller)
    (; mass, material_density, young_modulus) = system

    current_pos_diff = current_coords(system, particle) -
                       current_coords(system, neighbor)
    current_distance = norm(current_pos_diff)

    volume_particle = mass[particle] / material_density[particle]
    volume_neighbor = mass[neighbor] / material_density[neighbor]

    kernel_weight = smoothing_kernel(system, initial_distance)

    J_a = deformation_gradient(system, particle)
    J_b = deformation_gradient(system, neighbor)

    # Use the symmetry of epsilon to simplify computations
    eps_sum = (J_a + J_b) * initial_pos_diff - 2 * current_pos_diff
    delta_sum = dot(eps_sum, current_pos_diff) / current_distance

    f = 0.5 * penalty_force.alpha * volume_particle * volume_neighbor *
        kernel_weight / initial_distance^2 * young_modulus * delta_sum *
        current_pos_diff / current_distance

    for i in 1:ndims(system)
        # Divide force by mass to obtain acceleration
        dv[i, particle] += f[i] / mass[particle]
    end

    return dv
end
