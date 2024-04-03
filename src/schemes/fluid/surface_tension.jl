abstract type AkinciTypeSurfaceTension end

@doc raw"""
CohesionForceAkinci(surface_tension_coefficient=1.0)

Implements the cohesion force model by Akinci, focusing on intra-particle forces
to simulate surface tension and adhesion in fluid dynamics. This model is a crucial component
for capturing the complex interactions at the fluid surface, such as droplet formation
and the merging or breaking of fluid bodies.

# Keywords
- `surface_tension_coefficient=1.0`: Modifies the intensity of the surface tension-induced force,
 enabling the tuning of the fluid's surface tension properties within the simulation.

# Mathematical Formulation and Implementation Details
The model calculates the cohesion force based on the distance between particles and the smoothing length.
This force is determined using two distinct regimes within the support radius:

- For particles closer than half the support radius,
  a repulsive force is calculated to prevent particle clustering too tightly,
  enhancing the simulation's stability and realism.

- Beyond half the support radius and within the full support radius,
  an attractive force is computed, simulating the effects of surface tension that draw particles together.

The cohesion force, \( F_{\text{cohesion}} \), for a pair of particles is given by:

```math
F_{\text{cohesion}} = -\sigma m_b C \frac{\vec{r}}{|\vec{r}|}
```
where:
- σ represents the surface_tension_coefficient, adjusting the overall strength of the cohesion effect.
- C is a scalar function of the distance between particles.

# Reference:
- Akinci et al. "Versatile Surface Tension and Adhesion for SPH Fluids".
  In: Proceedings of Siggraph Asia 2013.
  [doi: 10.1145/2508363.2508395](https://doi.org/10.1145/2508363.2508395)
"""
struct CohesionForceAkinci{ELTYPE} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function CohesionForceAkinci(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

@doc raw"""
SurfaceTensionAkinci(surface_tension_coefficient=1.0)

Implements a model for simulating surface tension and adhesion effects drawing upon the
principles outlined by Akinci et al. This model is instrumental in capturing the nuanced
behaviors of fluid surfaces, such as droplet formation and the dynamics of merging or
separation, by utilizing intra-particle forces.

# Keywords
- `surface_tension_coefficient=1.0`: A parameter to adjust the magnitude of
   surface tension forces, facilitating the fine-tuning of how surface tension phenomena
   are represented in the simulation.

# Mathematical Formulation and Implementation Details
The cohesion force between particles is computed considering their separation and the
influence radius, with the force's nature—repulsive or attractive—determined by the
particle's relative proximity within the support radius:

- When particles are closer than half the support radius, the model calculates a
  repulsive force to prevent excessive aggregation, thus enhancing the simulation's stability and realism.
- For distances beyond half the support radius and up to the full extent of the support radius,
  the model computes an attractive force, reflecting the cohesive nature of surface tension
  that tends to draw particles together.

The total force exerted on a particle by another is described by:

```math
F_{\text{total}} = F_{\text{cohesion}} - \sigma (n_a - n_b) \frac{\vec{r}}{|\vec{r}|}
```
where:
- σ represents the surface_tension_coefficient, adjusting the overall strength of the cohesion effect.
- `C`` is a scalar function of the distance between particles.
- `n` being the normal vector
- `F_{cohesion}` being the cohesion/repulsion force excerted on a particle pair.

# Reference:
- Akinci et al. "Versatile Surface Tension and Adhesion for SPH Fluids".
  In: Proceedings of Siggraph Asia 2013.
  [doi: 10.1145/2508363.2508395](https://doi.org/10.1145/2508363.2508395)
"""
struct SurfaceTensionAkinci{ELTYPE} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionAkinci(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function (surface_tension::Union{CohesionForceAkinci, SurfaceTensionAkinci})(smoothing_length, mb, pos_diff, distance)
    return cohesion_force_akinci(surface_tension, smoothing_length, mb, pos_diff, distance)
end

function (surface_tension::SurfaceTensionAkinci)(support_radius, mb, na, nb, pos_diff,
                                                 distance)
    (; surface_tension_coefficient) = surface_tension
    return cohesion_force_akinci(surface_tension, support_radius, mb, pos_diff,
                                 distance) .- (surface_tension_coefficient * (na - nb))
end

@fastpow @inline function cohesion_force_akinci(surface_tension::AkinciTypeSurfaceTension,
    support_radius, mb, pos_diff, distance)
    (; surface_tension_coefficient) = surface_tension

    # Eq. 2
    # we only reach this function when distance > eps
    C = 0
    if distance <= support_radius
        if distance > 0.5 * support_radius
            # attractive force
            C = (support_radius - distance)^3 * distance^3
        else
            # distance < 0.5 * support_radius
            # repulsive force
            C = 2 * (support_radius - distance)^3 * distance^3 - support_radius^6 / 64.0
        end
        C *= 32.0 / (pi * support_radius^9)
    end

    # Eq. 1 in acceleration form
    cohesion_force = -surface_tension_coefficient * mb * C * pos_diff / distance

    return cohesion_force
end

@inline function adhesion_force_akinci(surface_tension::AkinciTypeSurfaceTension,
    support_radius, mb, pos_diff, distance, adhesion_coefficient)
    # Eq. 7
    # we only reach this function when distance > eps
    A = 0
    if distance <= support_radius
        if distance > 0.5 * support_radius
            A = 0.007/support_radius^3.25 * (-4*distance^2/support_radius + 6 * distance - 2 * support_radius)^0.25
        end
    end

    # Eq. 6 in acceleration form with mb being the boundary mass calculated as mb=rho_0 * volume (Akinci boundary condition treatment)
    adhesion_force = -adhesion_coefficient * mb * A * pos_diff / distance

    return adhesion_force
end

# section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: most of the time this only leads to an approximation of the surface normal
function calc_normal_akinci(surface_tension::SurfaceTensionAkinci, u_system,
                            v_neighbor_container, u_neighbor_container,
                            neighborhood_search, system, neighbor_system::FluidSystem)
    (; smoothing_kernel, smoothing_length, cache) = system

    @threaded for particle in each_moving_particle(system)
        particle_coords = current_coords(u_system, system, particle)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_system, system, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)
            # correctness strongly depends on this being a symmetric distribution of particles
            if sqrt(eps()) < distance <= compact_support(smoothing_kernel, smoothing_length)
                m_b = hydrodynamic_mass(neighbor_system, neighbor)
                density_neighbor = particle_density(v_neighbor_container,
                                                    neighbor_system, neighbor)
                grad_kernel = smoothing_kernel_grad(system, pos_diff, distance,
                                                    particle)
                @simd for i in 1:ndims(system)
                    cache.surface_normal[i, particle] += m_b / density_neighbor *
                                                         grad_kernel[i]
                end
            end
        end

        for i in 1:ndims(system)
            cache.surface_normal[i, particle] *= smoothing_length
        end
    end
end

function calc_normal_akinci(surface_tension, u_system,
                            v_neighbor_container, u_neighbor_container,
                            neighborhood_search, system,
                            neighbor_system)
    # normal not needed
end
