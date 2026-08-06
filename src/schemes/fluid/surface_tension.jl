abstract type AbstractSurfaceTension end
abstract type AkinciTypeSurfaceTension <: AbstractSurfaceTension end

function validate_surface_tension_coefficient(surface_tension_coefficient)
    if !(surface_tension_coefficient isa Real) ||
       !isfinite(surface_tension_coefficient) || surface_tension_coefficient < 0
        throw(ArgumentError("`surface_tension_coefficient` must be a finite, non-negative real number"))
    end

    return surface_tension_coefficient
end

@doc raw"""
    CohesionForceAkinci(surface_tension_coefficient=1.0)

This model only implements the cohesion force of the Akinci [Akinci2013](@cite) surface tension model.
It does not require a surface-normal method.

The three-dimensional cohesion kernel uses the normalization published by Akinci et al. In two
dimensions, TrixiParticles.jl uses an integral-matched extension that is independent of particle
resolution.

See [`surface_tension`](@ref) for more details.

# Keywords
- `surface_tension_coefficient=1.0`: Finite, non-negative coefficient modifying the
  fluid-fluid cohesion force. Zero disables this force; wall adhesion is controlled by the
  boundary's `adhesion_coefficient`.
"""
struct CohesionForceAkinci{ELTYPE <: Real} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function CohesionForceAkinci(; surface_tension_coefficient=1.0)
        coefficient = validate_surface_tension_coefficient(surface_tension_coefficient)
        new{typeof(coefficient)}(coefficient)
    end
end

const AKINCI_COHESION_SURFACE_ENERGY_FACTOR_3D = 21 / 7040

@doc raw"""
    SurfaceTensionAkinciCohesionPhysical(; surface_tension_coefficient,
                                          reference_density)

Three-dimensional, cohesion-only Akinci model with a physical surface tension coefficient.
The model uses the central pair force of [`CohesionForceAkinci`](@ref), but converts the
surface tension ``\sigma`` in N/m to the internal Akinci coefficient at the current compact
support radius ``h_c`` according to

```math
\gamma = \frac{7040\sigma}{21\rho_0^2h_c^2}.
```

This conversion follows from the continuum surface energy of a planar interface. It removes
the support-radius dependence of the original coefficient, requires no surface normals, and
preserves the pair force's exact linear- and angular-momentum conservation.

For wall interaction, the boundary's `adhesion_coefficient` is a dimensionless multiplier of
the same cohesion kernel. The Young-Dupre mapping for a desired contact angle ``\theta`` is
`adhesion_coefficient = (1 + cosd(theta)) / 2`. Thus, zero represents ``180^\circ`` and one
represents ``0^\circ``. Values outside this range can be used for empirical tuning.

This model is only supported in three dimensions. The original [`CohesionForceAkinci`](@ref)
remains available when an empirical coefficient or a two-dimensional model is desired.

# Keywords
- `surface_tension_coefficient`: Finite, non-negative physical surface tension ``\sigma`` in
  N/m. Zero disables fluid-fluid cohesion.
- `reference_density`: Finite, positive rest density ``\rho_0`` in kg/m^3.
"""
struct SurfaceTensionAkinciCohesionPhysical{ELTYPE <: Real} <:
       AkinciTypeSurfaceTension
    surface_tension_coefficient :: ELTYPE
    reference_density           :: ELTYPE

    function SurfaceTensionAkinciCohesionPhysical(; surface_tension_coefficient,
                                                  reference_density)
        coefficient = validate_surface_tension_coefficient(surface_tension_coefficient)
        if !(reference_density isa Real) || !isfinite(reference_density) ||
           reference_density <= 0
            throw(ArgumentError("`reference_density` must be a finite, positive real number"))
        end

        coefficient_, reference_density_ = promote(coefficient, reference_density)
        new{typeof(coefficient_)}(coefficient_, reference_density_)
    end
end

@inline function akinci_physical_cohesion_coefficient(surface_tension, support_radius)
    factor = oftype(support_radius, AKINCI_COHESION_SURFACE_ENERGY_FACTOR_3D)
    return surface_tension.surface_tension_coefficient /
           (factor * surface_tension.reference_density^2 * support_radius^2)
end

@doc raw"""
    SurfaceTensionAkinci(surface_tension_coefficient=1.0)

Implements a model for surface tension and adhesion effects drawing upon the
principles outlined by Akinci [Akinci2013](@cite). This model is instrumental in capturing the nuanced
behaviors of fluid surfaces, such as droplet formation and the dynamics of merging or
separation, by utilizing intra-particle forces.

The three-dimensional cohesion and adhesion kernels use the normalizations published by Akinci
et al. In two dimensions, TrixiParticles.jl uses integral-matched extensions that are independent
of particle resolution.

See [`surface_tension`](@ref) for more details.

# Keywords
- `surface_tension_coefficient=1.0`: Finite, non-negative coefficient adjusting the
  magnitude of surface tension forces. Zero disables the fluid-fluid force.
"""
struct SurfaceTensionAkinci{ELTYPE <: Real} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionAkinci(; surface_tension_coefficient=1.0)
        coefficient = validate_surface_tension_coefficient(surface_tension_coefficient)
        new{typeof(coefficient)}(coefficient)
    end
end

@doc raw"""
    SurfaceTensionMorris(surface_tension_coefficient=1.0)

This model implements the surface tension approach described by Morris [Morris2000](@cite).
It calculates surface tension forces based on the curvature of the fluid interface
using particle normals and their divergence, making it suitable for simulating
phenomena like droplet formation and capillary wave dynamics.

The one-phase color-gradient magnitude is retained as a normalized surface delta. The local
continuum-surface-force acceleration is evaluated once per particle as
``-sigma * kappa * delta_s * n_hat / rho``. Smooth interface activity is shared with
[`SurfaceTensionMomentumMorris`](@ref), avoiding discrete normal and curvature-stencil switches.

See [`surface_tension`](@ref) for more details.


# Keywords
- `surface_tension_coefficient=1.0`: Finite, non-negative physical surface tension in N/m.
  Zero disables the force.
"""
struct SurfaceTensionMorris{ELTYPE <: Real} <: AbstractSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionMorris(; surface_tension_coefficient=1.0)
        coefficient = validate_surface_tension_coefficient(surface_tension_coefficient)
        new{typeof(coefficient)}(coefficient)
    end
end

function create_cache_surface_tension(surface_tension, ELTYPE, NDIMS, nparticles)
    return (;)
end

function create_cache_surface_tension(::SurfaceTensionAkinciCohesionPhysical, ELTYPE,
                                      NDIMS, nparticles)
    if NDIMS != 3
        throw(ArgumentError("`SurfaceTensionAkinciCohesionPhysical` is only supported in three dimensions"))
    end

    return (;)
end

function create_cache_surface_tension(::AkinciTypeSurfaceTension, ELTYPE, NDIMS,
                                      nparticles)
    if NDIMS != 2 && NDIMS != 3
        throw(ArgumentError("Akinci surface tension is only supported in two and three dimensions"))
    end

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

This is a one-phase free-surface formulation. Validated wetted-wall energy can be enabled
explicitly with
`ColorfieldSurfaceNormal(contact_model=WettedAreaContactAngle(theta))`; omitting the contact model
preserves the no-wetting default.

See [`surface_tension`](@ref) for more details.

# Keywords
- `surface_tension_coefficient=1.0`: Finite, non-negative physical surface tension in N/m.
  Zero disables the force.
"""
struct SurfaceTensionMomentumMorris{ELTYPE <: Real} <: AbstractSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        coefficient = validate_surface_tension_coefficient(surface_tension_coefficient)
        new{typeof(coefficient)}(coefficient)
    end
end

# Surface-model capabilities are expressed through dispatch so that constructors and update
# stages do not need to duplicate concrete model checks.
@inline requires_surface_normal(::Nothing) = false
@inline requires_surface_normal(::CohesionForceAkinci) = false
@inline requires_surface_normal(::SurfaceTensionAkinciCohesionPhysical) = false
@inline requires_surface_normal(::Any) = true

function create_cache_surface_tension(::SurfaceTensionMomentumMorris, ELTYPE, NDIMS,
                                      nparticles)
    delta_s = Array{ELTYPE, 1}(undef, nparticles)
    interface_activity = Array{ELTYPE, 1}(undef, nparticles)
    divergence_correction = Array{ELTYPE, 1}(undef, nparticles)
    return (; delta_s, interface_activity, divergence_correction)
end

# `surface_normal` stores the unscaled colorfield gradient, which is also used by the Morris
# models. Equation 3 in Akinci et al. uses the dimensionless normal from their equation 2,
# whose prefactor `h` is the compact-support radius, not the kernel smoothing length.
@inline function akinci_surface_normal(particle_system::AbstractFluidSystem, particle)
    support_radius = compact_support(system_smoothing_kernel(particle_system),
                                     smoothing_length(particle_system, particle))
    return support_radius * surface_normal(particle_system, particle)
end

# Note that `floating_point_number^integer_literal` is lowered to `Base.literal_pow`.
# Currently, specializations reducing this to simple multiplications exist only up
# to a power of three, see
# https://github.com/JuliaLang/julia/blob/34934736fa4dcb30697ac1b23d11d5ad394d6a4d/base/intfuncs.jl#L327-L339
# By using the `@fastpow` macro, we are consciously trading off some precision in the result
# for enhanced computational speed. This is especially useful in scenarios where performance
# is a higher priority than exact precision.
@fastpow @inline function cohesion_kernel_normalization_akinci(support_radius, ::Val{2})
    return oftype(support_radius, 25280 / (627 * pi)) / support_radius^8
end

@fastpow @inline function cohesion_kernel_normalization_akinci(support_radius, ::Val{3})
    return oftype(support_radius, 32 / pi) / support_radius^9
end

@inline function adhesion_kernel_normalization_akinci(support_radius, ::Val{2})
    return oftype(support_radius, 13 / 1200) /
           (support_radius^2 * sqrt(sqrt(support_radius)))
end

@inline function adhesion_kernel_normalization_akinci(support_radius, ::Val{3})
    return oftype(support_radius, 0.007) /
           (support_radius^3 * sqrt(sqrt(support_radius)))
end

@fastpow @inline function cohesion_force_akinci(surface_tension, support_radius, m_b,
                                                pos_diff, distance, dimensions)
    (; surface_tension_coefficient) = surface_tension

    # Eq. 2, using the published normalization in 3D and an integral-matched one in 2D.
    # We only reach this function when `sqrt(eps()) < distance <= support_radius`
    if distance > 0.5 * support_radius
        # Attractive force
        C = (support_radius - distance)^3 * distance^3
    else
        # `distance < 0.5 * support_radius`
        # Repulsive force
        C = 2 * (support_radius - distance)^3 * distance^3 - support_radius^6 / 64.0
    end
    C *= cohesion_kernel_normalization_akinci(support_radius, dimensions)

    # Eq. 1 in acceleration form
    cohesion_force = -surface_tension_coefficient * m_b * C * pos_diff / distance

    return cohesion_force
end

@inline function adhesion_force_akinci(surface_tension, support_radius, m_b, pos_diff,
                                       distance, adhesion_coefficient, dimensions)
    distance >= support_radius && return zero(pos_diff)

    distance <= 0.5 * support_radius && return zero(pos_diff)

    # Eq. 7. The factored radicand avoids cancellation close to the support boundary.
    radicand = 2 * (2 * distance - support_radius) *
               (support_radius - distance) / support_radius
    fourth_root = sqrt(sqrt(max(zero(radicand), radicand)))
    normalization = adhesion_kernel_normalization_akinci(support_radius, dimensions)
    A = normalization * fourth_root

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
                                           pos_diff, distance, Val(ndims(particle_system)))

    return dv_particle
end

@inline function surface_tension_force!(dv_particle,
                                        surface_tension_a::SurfaceTensionAkinciCohesionPhysical,
                                        surface_tension_b::SurfaceTensionAkinciCohesionPhysical,
                                        particle_system::AbstractFluidSystem,
                                        neighbor_system::AbstractFluidSystem,
                                        particle, neighbor, pos_diff, distance, rho_a,
                                        rho_b, grad_kernel,
                                        surface_tension_correction)
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    support_radius = compact_support(system_smoothing_kernel(particle_system),
                                     smoothing_length(particle_system, particle))
    coefficient = akinci_physical_cohesion_coefficient(surface_tension_a,
                                                       support_radius)
    cohesion = (; surface_tension_coefficient=coefficient)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    dv_particle[] += surface_tension_correction *
                     cohesion_force_akinci(cohesion, support_radius, m_b, pos_diff,
                                           distance, Val(ndims(particle_system)))

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

    # No surface tension with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    n_a = akinci_surface_normal(particle_system, particle)
    n_b = akinci_surface_normal(neighbor_system, neighbor)
    support_radius = compact_support(smoothing_kernel,
                                     smoothing_length(particle_system, particle))

    dv_particle[] += surface_tension_correction *
                     cohesion_force_akinci(surface_tension_a, support_radius, m_b,
                                           pos_diff, distance, Val(ndims(particle_system)))
    dv_particle[] -= surface_tension_correction * surface_tension_coefficient *
                     (n_a - n_b)

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

    normal = surface_tension_normal(particle_system, particle)
    curvature_a = curvature(particle_system, particle)
    return -surface_tension.surface_tension_coefficient / rho_a * curvature_a * delta_s *
           normal
end

@inline function contact_angle_acceleration(surface_tension, particle_system,
                                            surface_normal_method, particle, rho_a,
                                            vector_template)
    return zero(vector_template)
end

@inline function surface_stress_times_gradient(particle_system, particle, grad_kernel)
    delta_s = @inbounds particle_system.cache.delta_s[particle]
    iszero(delta_s) && return zero(grad_kernel)

    normal = surface_tension_normal(particle_system, particle)
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
                                           distance, adhesion_coefficient,
                                           Val(ndims(particle_system)))

    return dv_particle
end

@inline function akinci_physical_wall_cohesion_force!(dv_particle,
                                                      surface_tension::SurfaceTensionAkinciCohesionPhysical,
                                                      particle_system::AbstractFluidSystem,
                                                      neighbor_system,
                                                      particle, neighbor, pos_diff,
                                                      distance)
    wall_ratio = neighbor_system.adhesion_coefficient
    iszero(wall_ratio) && return dv_particle
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    support_radius = compact_support(system_smoothing_kernel(particle_system),
                                     smoothing_length(particle_system, particle))
    distance >= support_radius && return dv_particle

    coefficient = wall_ratio *
                  akinci_physical_cohesion_coefficient(surface_tension, support_radius)
    wall_cohesion = (; surface_tension_coefficient=coefficient)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    dv_particle[] += cohesion_force_akinci(wall_cohesion, support_radius, m_b, pos_diff,
                                           distance, Val(ndims(particle_system)))

    return dv_particle
end

@inline function adhesion_force!(dv_particle,
                                 surface_tension::SurfaceTensionAkinciCohesionPhysical,
                                 particle_system::AbstractFluidSystem,
                                 neighbor_system::AbstractBoundarySystem,
                                 particle, neighbor, pos_diff, distance)
    return akinci_physical_wall_cohesion_force!(dv_particle, surface_tension,
                                                particle_system, neighbor_system,
                                                particle, neighbor, pos_diff, distance)
end

@inline function adhesion_force!(dv_particle, surface_tension, particle_system,
                                 neighbor_system, particle, neighbor, pos_diff, distance)
    return dv_particle
end
