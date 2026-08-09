abstract type AbstractSurfaceTension end
abstract type AkinciTypeSurfaceTension <: AbstractSurfaceTension end

@doc raw"""
    CohesionForceAkinci(surface_tension_coefficient=1.0)

This model only implements the cohesion force of the Akinci [Akinci2013](@cite) surface tension model.

See [`surface_tension`](@ref) for more details.

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
principles outlined by Akinci [Akinci2013](@cite). This model is instrumental in capturing the nuanced
behaviors of fluid surfaces, such as droplet formation and the dynamics of merging or
separation, by utilizing intra-particle forces.

See [`surface_tension`](@ref) for more details.

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

This model implements the surface tension approach described by Morris [Morris2000](@cite).
It calculates surface tension forces based on the curvature of the fluid interface
using particle normals and their divergence, making it suitable for simulating
phenomena like droplet formation and capillary wave dynamics.

The one-phase color-gradient magnitude is retained as a surface delta. The local
continuum-surface-force acceleration is evaluated once per particle as
``-sigma * kappa * delta_s * n_hat / rho``. A smooth interface activity avoids discrete normal
and curvature-stencil switches when using [`ColorfieldSurfaceNormal`](@ref).

See [`surface_tension`](@ref) for more details.


# Keywords
- `surface_tension_coefficient=1.0`: Adjusts the magnitude of the surface tension
   forces, enabling tuning of fluid surface behaviors in simulations.
"""
struct SurfaceTensionMorris{ELTYPE} <: AbstractSurfaceTension
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
    delta_s = Array{ELTYPE, 1}(undef, nparticles)
    interface_activity = Array{ELTYPE, 1}(undef, nparticles)
    support_moment = Array{ELTYPE, 1}(undef, nparticles)
    return (; curvature, delta_s, interface_activity, support_moment)
end

@doc raw"""
    SurfaceTensionMomentumMorris(surface_tension_coefficient=1.0)

This model implements the conservative continuum-surface-stress (CSS) approach outlined by
Morris [Morris2000](@cite). It computes the divergence of
``\sigma\delta_s(I - \hat{n}\otimes\hat{n})`` with the same symmetric pair operator used by
the fluid momentum equation. This avoids an explicit curvature estimate and conserves linear
momentum exactly for constant smoothing length.

The unnormalized color-gradient magnitude is retained as the surface delta ``\delta_s`` before
the gradient is converted to a unit normal. The stress projection is evaluated directly during
the fluid interaction, so no per-particle stress tensor or global reduction is required. A
symmetric scalar reproducing correction is accumulated during the normal pass and applied to the
stress divergence. It restores first-order scaling near truncated kernel support without another
neighbor traversal or loss of pairwise momentum conservation.

See [`surface_tension`](@ref) for more details.

# Keywords
- `surface_tension_coefficient=1.0`: Physical surface tension coefficient in N/m.
"""
struct SurfaceTensionMomentumMorris{ELTYPE} <: AbstractSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function create_cache_surface_tension(::SurfaceTensionMomentumMorris, ELTYPE, NDIMS,
                                      nparticles)
    delta_s = Array{ELTYPE, 1}(undef, nparticles)
    interface_activity = Array{ELTYPE, 1}(undef, nparticles)
    divergence_correction = Array{ELTYPE, 1}(undef, nparticles)
    return (; delta_s, interface_activity, divergence_correction)
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
@inline function surface_tension_force!(dv_particle, surface_tension_a, surface_tension_b,
                                        particle_system, neighbor_system, particle,
                                        neighbor, pos_diff, distance, rho_a, rho_b,
                                        grad_kernel, surface_tension_correction)
    return dv_particle
end

@inline function surface_tension_force!(dv_particle,
                                        surface_tension_a::CohesionForceAkinci,
                                        surface_tension_b::CohesionForceAkinci,
                                        particle_system::AbstractFluidSystem,
                                        neighbor_system::AbstractFluidSystem,
                                        particle, neighbor, pos_diff, distance,
                                        rho_a, rho_b, grad_kernel,
                                        surface_tension_correction)
    (; smoothing_kernel) = particle_system

    # No cohesion with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    support_radius = compact_support(smoothing_kernel,
                                     smoothing_length(particle_system, particle))

    dv_particle[] += surface_tension_correction *
                     cohesion_force_akinci(surface_tension_a, support_radius, m_b,
                                           pos_diff, distance)

    return dv_particle
end

@inline function surface_tension_force!(dv_particle,
                                        surface_tension_a::SurfaceTensionAkinci,
                                        surface_tension_b::SurfaceTensionAkinci,
                                        particle_system::AbstractFluidSystem,
                                        neighbor_system::AbstractFluidSystem, particle,
                                        neighbor,
                                        pos_diff, distance, rho_a, rho_b, grad_kernel,
                                        surface_tension_correction)
    (; smoothing_kernel) = particle_system
    (; surface_tension_coefficient) = surface_tension_a

    smoothing_length_ = smoothing_length(particle_system, particle)
    # No surface tension with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    n_a = surface_normal(particle_system, particle)
    n_b = surface_normal(neighbor_system, neighbor)
    support_radius = compact_support(smoothing_kernel, smoothing_length_)

    dv_particle[] += surface_tension_correction *
                     cohesion_force_akinci(surface_tension_a, support_radius, m_b,
                                           pos_diff, distance)
    dv_particle[] -= surface_tension_correction * surface_tension_coefficient *
                     (n_a - n_b) * smoothing_length_

    return dv_particle
end
@inline function surface_tension_force!(dv_particle,
                                        surface_tension_a::SurfaceTensionMorris,
                                        surface_tension_b::SurfaceTensionMorris,
                                        particle_system::AbstractFluidSystem,
                                        neighbor_system::AbstractFluidSystem,
                                        particle, neighbor, pos_diff, distance,
                                        rho_a, rho_b, grad_kernel,
                                        surface_tension_correction)
    # Morris CSF is a particle-local continuum force. It is added once outside the
    # neighbor loop by `surface_tension_acceleration`.
    return dv_particle
end

@inline function surface_tension_acceleration(surface_tension, particle_system, particle,
                                              rho_a, vector_template)
    return zero(vector_template)
end

@inline function surface_tension_acceleration(surface_tension::SurfaceTensionMorris,
                                              particle_system, particle, rho_a,
                                              vector_template)
    delta_s = @inbounds particle_system.cache.delta_s[particle]
    iszero(delta_s) && return zero(vector_template)

    normal = surface_normal(particle_system, particle)
    curvature_a = curvature(particle_system, particle)
    return -surface_tension.surface_tension_coefficient / rho_a * curvature_a * delta_s *
           normal
end

@inline function surface_stress_times_gradient(particle_system, particle, grad_kernel)
    delta_s = @inbounds particle_system.cache.delta_s[particle]
    iszero(delta_s) && return zero(grad_kernel)

    normal = surface_normal(particle_system, particle)
    return delta_s * (grad_kernel - normal * dot(normal, grad_kernel))
end

@inline function symmetric_surface_divergence_correction(particle_system,
                                                         neighbor_system,
                                                         particle, neighbor)
    correction_a = @inbounds particle_system.cache.divergence_correction[particle]
    correction_b = @inbounds neighbor_system.cache.divergence_correction[neighbor]
    denominator = correction_a + correction_b
    denominator > eps(denominator) || return zero(denominator)
    return 2 / denominator
end

@inline function surface_tension_force!(dv_particle,
                                        surface_tension_a::SurfaceTensionMomentumMorris,
                                        surface_tension_b::SurfaceTensionMomentumMorris,
                                        particle_system::AbstractFluidSystem,
                                        neighbor_system::AbstractFluidSystem,
                                        particle, neighbor, pos_diff, distance,
                                        rho_a, rho_b, grad_kernel,
                                        surface_tension_correction)
    (; surface_tension_coefficient) = surface_tension_a

    # No surface tension with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    stress_gradient_a = surface_stress_times_gradient(particle_system, particle,
                                                      grad_kernel)
    stress_gradient_b = surface_stress_times_gradient(neighbor_system, neighbor,
                                                      grad_kernel)
    divergence_correction = symmetric_surface_divergence_correction(particle_system,
                                                                    neighbor_system,
                                                                    particle, neighbor)

    # This uses the same symmetric stress-divergence operator as the pressure force. The
    # Akinci free-surface correction is deliberately not applied to a continuum stress.
    dv_particle[] += divergence_correction * surface_tension_coefficient * m_b /
                     (rho_a * rho_b) * (stress_gradient_a + stress_gradient_b)

    return dv_particle
end

@inline function adhesion_force!(dv_particle,
                                 surface_tension::AkinciTypeSurfaceTension,
                                 particle_system::AbstractFluidSystem,
                                 neighbor_system::AbstractBoundarySystem, particle,
                                 neighbor,
                                 pos_diff, distance)
    (; adhesion_coefficient) = neighbor_system

    # No adhesion with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    # No reason to calculate the adhesion force if adhesion coefficient is near zero
    abs(adhesion_coefficient) < eps() && return dv_particle

    m_b = hydrodynamic_mass(neighbor_system, neighbor)

    support_radius = compact_support(particle_system.smoothing_kernel,
                                     smoothing_length(particle_system, particle))
    dv_particle[] += adhesion_force_akinci(surface_tension, support_radius, m_b, pos_diff,
                                           distance, adhesion_coefficient)

    return dv_particle
end

@inline function adhesion_force!(dv_particle, surface_tension, particle_system,
                                 neighbor_system, particle, neighbor, pos_diff, distance)
    return dv_particle
end
