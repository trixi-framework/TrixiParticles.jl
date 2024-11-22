abstract type SurfaceTension end
abstract type AkinciTypeSurfaceTension <: SurfaceTension end

@doc raw"""
    CohesionForceAkinci(surface_tension_coefficient=1.0)

This model only implements the cohesion force of the [Akinci2013](@cite) surface tension model.

# Keywords
- `surface_tension_coefficient=1.0`: Modifies the intensity of the surface tension-induced force,
   enabling the tuning of the fluid's surface tension properties within the simulation.
"""
struct CohesionForceAkinci{ELTYPE} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function CohesionForceAkinci(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

@doc raw"""
    SurfaceTensionAkinci(surface_tension_coefficient=1.0)

Implements a model for surface tension and adhesion effects drawing upon the
principles outlined by [Akinci2013](@cite). This model is instrumental in capturing the nuanced
behaviors of fluid surfaces, such as droplet formation and the dynamics of merging or
separation, by utilizing intra-particle forces.

# Keywords
- `surface_tension_coefficient=1.0`: A parameter to adjust the magnitude of
   surface tension forces, facilitating the fine-tuning of how surface tension phenomena
   are represented in the simulation.
"""
struct SurfaceTensionAkinci{ELTYPE} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionAkinci(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

@doc raw"""
    SurfaceTensionMorris(surface_tension_coefficient=1.0)

This model implements the surface tension approach described by [Morris2000](@cite).
It calculates surface tension forces based on the curvature of the fluid interface
using particle normals and their divergence, making it suitable for simulating
phenomena like droplet formation and capillary wave dynamics.

# Details
The method estimates curvature by combining particle color gradients and smoothing
functions to derive surface normals. The curvature is then used to compute forces
acting perpendicular to the interface. While this method provides accurate
surface tension forces, it does not conserve momentum explicitly.

# Keywords
- `surface_tension_coefficient=1.0`: Adjusts the magnitude of the surface tension
   forces, enabling tuning of fluid surface behaviors in simulations.
"""
struct SurfaceTensionMorris{ELTYPE} <: SurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionMorris(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function create_cache_surface_tension(surface_tension, ELTYPE, NDIMS, nparticles)
    return (;)
end

function create_cache_surface_tension(::SurfaceTensionMorris, ELTYPE, NDIMS, nparticles)
    curvature = Array{ELTYPE, 1}(undef, nparticles)
    return (; curvature)
end

@doc raw"""
    SurfaceTensionMomentumMorris(surface_tension_coefficient=1.0)

This model implements the momentum-conserving surface tension approach outlined by
[Morris2000](@cite). It calculates surface tension forces using the gradient of a stress
tensor, ensuring exact conservation of linear momentum. This method is particularly
useful for simulations where momentum conservation is critical, though it may require
numerical adjustments at higher resolutions.

# Details
The stress tensor approach replaces explicit curvature calculations, avoiding the
singularities associated with resolution increases. However, the method is computationally
intensive and may require stabilization techniques to handle tensile instability at high
particle densities.

# Keywords
- `surface_tension_coefficient=1.0`: A parameter to adjust the strength of surface tension
   forces, allowing fine-tuning to replicate physical behavior.
"""
struct SurfaceTensionMomentumMorris{ELTYPE} <: SurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function create_cache_surface_tension(::SurfaceTensionMomentumMorris, ELTYPE, NDIMS,
                                      nparticles)
    # Allocate stress tensor for each particle: NDIMS x NDIMS x nparticles
    delta_s = Array{ELTYPE, 1}(undef, nparticles)
    stress_tensor = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
    return (; stress_tensor, delta_s)
end

# Note that `floating_point_number^integer_literal` is lowered to `Base.literal_pow`.
# Currently, specializations reducing this to simple multiplications exist only up
# to a power of three, see
# https://github.com/JuliaLang/julia/blob/34934736fa4dcb30697ac1b23d11d5ad394d6a4d/base/intfuncs.jl#L327-L339
# By using the `@fastpow` macro, we are consciously trading off some precision in the result
# for enhanced computational speed. This is especially useful in scenarios where performance
# is a higher priority than exact precision.
@fastpow @inline function cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                pos_diff, distance)
    (; surface_tension_coefficient) = surface_tension

    # Eq. 2
    # We only reach this function when `sqrt(eps()) < distance <= support_radius`
    if distance > 0.5 * support_radius
        # Attractive force
        C = (support_radius - distance)^3 * distance^3
    else
        # `distance < 0.5 * support_radius`
        # Repulsive force
        C = 2 * (support_radius - distance)^3 * distance^3 - support_radius^6 / 64.0
    end
    C *= 32.0 / (pi * support_radius^9)

    # Eq. 1 in acceleration form
    cohesion_force = -surface_tension_coefficient * m_b * C * pos_diff / distance

    return cohesion_force
end

@inline function adhesion_force_akinci(surface_tension, support_radius, m_b, pos_diff,
                                       distance, adhesion_coefficient)

    # The neighborhood search has an `<=` check, but for `distance == support_radius`
    # the term inside the parentheses might be very slightly negative, causing an error with `^0.25`.
    # TODO Change this in the neighborhood search?
    # See https://github.com/trixi-framework/PointNeighbors.jl/issues/19
    distance >= support_radius && return zero(pos_diff)

    distance <= 0.5 * support_radius && return zero(pos_diff)

    # Eq. 7
    A = 0.007 / support_radius^3.25 *
        (-4 * distance^2 / support_radius + 6 * distance - 2 * support_radius)^0.25

    # Eq. 6 in acceleration form with `m_b` being the boundary mass calculated as
    # `m_b = rho_0 * volume` (Akinci boundary condition treatment)
    adhesion_force = -adhesion_coefficient * m_b * A * pos_diff / distance

    return adhesion_force
end

# Skip
@inline function surface_tension_force(surface_tension_a, surface_tension_b,
                                       particle_system, neighbor_system, particle, neighbor,
                                       pos_diff, distance, rho_a, rho_b, grad_kernel)
    return zero(pos_diff)
end

@inline function surface_tension_force(surface_tension_a::CohesionForceAkinci,
                                       surface_tension_b::CohesionForceAkinci,
                                       particle_system::FluidSystem,
                                       neighbor_system::FluidSystem, particle, neighbor,
                                       pos_diff, distance, rho_a, rho_b, grad_kernel)
    (; smoothing_length) = particle_system
    # No cohesion with oneself
    distance < sqrt(eps()) && return zero(pos_diff)

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    support_radius = compact_support(smoothing_kernel, smoothing_length)

    return cohesion_force_akinci(surface_tension_a, support_radius, m_b, pos_diff, distance)
end

@inline function surface_tension_force(surface_tension_a::SurfaceTensionAkinci,
                                       surface_tension_b::SurfaceTensionAkinci,
                                       particle_system::FluidSystem,
                                       neighbor_system::FluidSystem, particle, neighbor,
                                       pos_diff, distance, rho_a, rho_b, grad_kernel)
    (; smoothing_length, smoothing_kernel) = particle_system
    (; surface_tension_coefficient) = surface_tension_a

    # No surface tension with oneself
    distance < sqrt(eps()) && return zero(pos_diff)

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    n_a = surface_normal(particle_system, particle)
    n_b = surface_normal(neighbor_system, neighbor)
    support_radius = compact_support(smoothing_kernel, smoothing_length)

    return cohesion_force_akinci(surface_tension_a, support_radius, m_b,
                                 pos_diff, distance) .-
           (surface_tension_coefficient * (n_a - n_b) * smoothing_length)
end

@inline function surface_tension_force(surface_tension_a::SurfaceTensionMorris,
                                       surface_tension_b::SurfaceTensionMorris,
                                       particle_system::FluidSystem,
                                       neighbor_system::FluidSystem, particle, neighbor,
                                       pos_diff, distance, rho_a, rho_b, grad_kernel)
    (; surface_tension_coefficient) = surface_tension_a

    # No surface tension with oneself
    distance < sqrt(eps()) && return zero(pos_diff)

    n_a = surface_normal(particle_system, particle)
    curvature_a = curvature(particle_system, particle)

    return -surface_tension_coefficient / rho_a * curvature_a * n_a
end

@inline function surface_tension_force(surface_tension_a::SurfaceTensionMomentumMorris,
                                       surface_tension_b::SurfaceTensionMomentumMorris,
                                       particle_system::FluidSystem,
                                       neighbor_system::FluidSystem, particle, neighbor,
                                       pos_diff, distance, rho_a, rho_b, grad_kernel)
    (; surface_tension_coefficient) = surface_tension_a

    # No surface tension with oneself
    distance < sqrt(eps()) && return zero(pos_diff)

    S_a = particle_system.cache.stress_tensor[:, :, particle]
    S_b = neighbor_system.cache.stress_tensor[:, :, neighbor]

    m_b = hydrodynamic_mass(neighbor_system, neighbor)

    return surface_tension_coefficient * m_b * (S_a + S_b) / (rho_a * rho_b) * grad_kernel
end

@inline function adhesion_force(surface_tension::AkinciTypeSurfaceTension,
                                particle_system::FluidSystem,
                                neighbor_system::BoundarySystem, particle, neighbor,
                                pos_diff, distance)
    (; smoothing_length, smoothing_kernel) = particle_system
    (; adhesion_coefficient, boundary_model) = neighbor_system

    # No adhesion with oneself
    distance < sqrt(eps()) && return zero(pos_diff)

    # No reason to calculate the adhesion force if adhesion coefficient is near zero
    abs(adhesion_coefficient) < eps() && return zero(pos_diff)

    m_b = hydrodynamic_mass(neighbor_system, neighbor)

    support_radius = compact_support(smoothing_kernel, smoothing_length)
    return adhesion_force_akinci(surface_tension, support_radius, m_b, pos_diff, distance,
                                 adhesion_coefficient)
end

@inline function adhesion_force(surface_tension, particle_system, neighbor_system, particle,
                                neighbor, pos_diff, distance)
    return zero(pos_diff)
end

struct HuberContactModel{ELTYPE} end

@inline function contact_force(contact_model::HuberContactModel,
                               particle_system::FluidSystem,
                               neighbor_system::BoundarySystem, particle, neighbor,
                               pos_diff, distance)
end

@inline function contact_force(surface_tension, particle_system, neighbor_system, particle,
                               neighbor, pos_diff, distance)
    return zero(pos_diff)
end
