# Define an abstract type for contact models.
abstract type AbstractContactModel end

@doc raw"""
    HertzContactModel(; elastic_modulus, poissons_ratio)

Non-linear contact model based on Hertzian contact theory ([DiRenzo2004](@cite)).

This model calculates the normal contact force between two spherical particles (or a particle
and a boundary represented by an equivalent sphere) based on their material properties and
the overlap ``\delta``. The elastic part of the force is given by:
```math
F_{\text{elastic}} = \frac{4}{3} E^* \sqrt{R^*} \delta^{3/2},
```
where ``E^*`` is the effective Young's modulus and ``R^*`` is the effective radius.

The effective Young's modulus ``E^*`` is calculated from the Young's moduli ``E_a, E_b``
and Poisson's ratios ``\nu_a, \nu_b`` of the two contacting bodies:
```math
E^* = \left( \frac{1 - \nu_a^2}{E_a} + \frac{1 - \nu_b^2}{E_b} \right)^{-1}.
```

The effective radius ``R^*`` is calculated from the radii of the two particles ``R_a`` and  ``R_b``:
```math
R^* = \left( \frac{1}{R_a} + \frac{1}{R_b} \right)^{-1} = \frac{R_a R_b}{R_a + R_b}.
```
For particle-wall interactions, ``R_b \to \infty``, so ``R^* = R_a``.

The implementation also includes a damping force based on the approach described in
[DiRenzo2004](@cite), proportional to the normal component of the relative velocity
``v_{\text{rel,n}}``:
```math
F_{\text{damping}} = C_{\text{damp}} \gamma_c v_{\text{rel,n}},
```
where ``C_{\text{damp}}`` is the user-provided damping coefficient (damping ratio), and
``\gamma_c`` is a non-linear critical damping coefficient:
```math
\gamma_c = 2 \sqrt{m^* K_{\text{nonlin}}}
```
with ``m^*`` being the effective mass and ``K_{\text{nonlin}}`` being a non-linear stiffness term
related to the current state:
```math
K_{\text{nonlin}} = \frac{F_{\text{elastic}}}{\delta} = \frac{4}{3} E^* \sqrt{R^* \delta}.
```

The total normal force is ``F_n = F_{\text{elastic}} + F_{\text{damping}}``.

# Fields
- `elastic_modulus::Float64`: Material Young's modulus ``E``.
- `poissons_ratio::Float64`: Material Poisson's ratio ``\nu``.
"""
struct HertzContactModel{ELTYPE <: Real} <: AbstractContactModel
    elastic_modulus::ELTYPE  # Material elastic modulus
    poissons_ratio::ELTYPE   # Material Poisson's ratio
end

@inline function collision_force_normal(model::HertzContactModel,
                                        particle_system, neighbor_system,
                                        overlap, normal, v_particle_system,
                                        v_neighbor_system,
                                        particle, neighbor, damping_coefficient)
    # Use material properties from the Hertz contact model.
    E_a = model.elastic_modulus
    nu_a = model.poissons_ratio

    # TODO: needs to be dispatched and handled for boundaries
    # For the neighbor, use its properties if available (otherwise, assume identical material)
    if neighbor_system isa DEMSystem && neighbor_system.contact_model isa HertzContactModel
        E_b = neighbor_system.contact_model.elastic_modulus
        nu_b = neighbor_system.contact_model.poissons_ratio
    else
        E_b = E_a
        nu_b = nu_a
    end

    # Compute relative velocity along the contact normal
    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b
    v_diff_norm = dot(v_diff, normal)

    # Compute the effective elastic modulus (Hertzian theory)
    E_star = 1 / (((1 - nu_a^2) / E_a) + ((1 - nu_b^2) / E_b))

    # Compute the effective radius.
    r_a = particle_system.radius[particle]
    r_b = neighbor_system.radius[neighbor]
    r_star = (r_a * r_b) / (r_a + r_b)

    # Non-linear stiffness for Hertzian contact
    elastic_force_per_overlap = 4 * E_star * sqrt(r_star * overlap) / 3

    # Compute effective mass for damping
    if neighbor_system isa DEMSystem
        m_a = particle_system.mass[particle]
        m_b = neighbor_system.mass[neighbor]
        m_star = (m_a * m_b) / (m_a + m_b)
    else
        m_star = particle_system.mass[particle]
    end

    # Critical damping coefficient
    gamma_c = 2 * sqrt(m_star * elastic_force_per_overlap)

    # Total normal force: elastic + damping term
    force_magnitude = elastic_force_per_overlap * overlap +
                      damping_coefficient * gamma_c * v_diff_norm

    return force_magnitude * normal
end

@doc raw"""
    LinearContactModel(; normal_stiffness)

Linear spring-dashpot contact model ([Cundall1979](@cite)).

This model calculates the normal contact force between two objects based on a linear spring
law for the elastic component and a linear viscous damping law for the dissipative component.
The total normal force ``F_n`` is given by
```math
F_n = k_n \delta + \gamma_d v_{\text{rel,n}},
```
where ``k_n`` is the normal stiffness, ``\delta`` is the overlap between the objects,
``v_{\text{rel,n}}`` is the normal component of the relative velocity, and ``\gamma_d``
is the damping coefficient.

The damping coefficient ``\gamma_d`` is calculated based on the critical damping
coefficient ``\gamma_c`` and a user-provided damping ratio ``C_{\text{damp}}``:
```math
\gamma_d = C_{\text{damp}} \gamma_c,
```
where the critical damping for this linear system is
```math
\gamma_c = 2 \sqrt{m^* k_n}
```
and ``m^*`` is the effective mass of the colliding pair.

The total force is applied along the normal direction connecting the centers of the
contacting objects.

# Fields
- `normal_stiffness::Real`: Constant spring stiffness ``k_n`` for the normal direction.
"""
struct LinearContactModel{ELTYPE <: Real} <: AbstractContactModel
    normal_stiffness::ELTYPE
end

@inline function collision_force_normal(model::LinearContactModel,
                                        particle_system, neighbor_system,
                                        overlap, normal, v_particle_system,
                                        v_neighbor_system,
                                        particle, neighbor, damping_coefficient)
    normal_stiffness = model.normal_stiffness

    v_a = current_velocity(v_particle_system, particle_system, particle)
    v_b = current_velocity(v_neighbor_system, neighbor_system, neighbor)
    v_diff = v_a - v_b
    v_diff_norm = dot(v_diff, normal)

    # Compute effective mass for damping
    m_star = compute_effective_mass(particle_system, particle, neighbor_system, neighbor)

    gamma_c = 2 * sqrt(m_star * normal_stiffness)
    force_magnitude = normal_stiffness * overlap +
                      damping_coefficient * gamma_c * v_diff_norm

    return force_magnitude * normal
end

# TODO: add dispatches to the neighbor systems
function compute_effective_mass(particle_system, particle, neighbor_system, neighbor)
    return particle_system.mass[particle]
end

function compute_effective_mass(particle_system, particle, neighbor_system::DEMSystem,
                                neighbor)
    m_a = particle_system.mass[particle]
    m_b = neighbor_system.mass[neighbor]
    return (m_a * m_b) / (m_a + m_b)
end
