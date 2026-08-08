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

See [`surface_tension`](@ref) for more details.


# Keywords
- `surface_tension_coefficient=1.0`: Finite, non-negative coefficient adjusting the
  magnitude of surface tension forces. Zero disables the force.
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

function create_cache_surface_tension(::AkinciTypeSurfaceTension, ELTYPE, NDIMS,
                                      nparticles)
    if NDIMS != 2 && NDIMS != 3
        throw(ArgumentError("Akinci surface tension is only supported in two and three dimensions"))
    end

    return (;)
end

function create_cache_surface_tension(::SurfaceTensionMorris, ELTYPE, NDIMS, nparticles)
    curvature = Array{ELTYPE, 1}(undef, nparticles)
    return (; curvature)
end

@doc raw"""
    SurfaceTensionMomentumMorris(surface_tension_coefficient=1.0)

This model implements the momentum-conserving surface tension approach outlined by Morris
[Morris2000](@cite). It calculates surface tension forces using the divergence of a stress
tensor, ensuring exact conservation of linear momentum. This method is particularly
useful for simulations where momentum conservation is critical, though it may require
numerical adjustments at higher resolutions.

See [`surface_tension`](@ref) for more details.

# Keywords
- `surface_tension_coefficient=1.0`: Finite, non-negative coefficient adjusting the
  strength of surface tension forces. Zero disables the force.
"""
struct SurfaceTensionMomentumMorris{ELTYPE <: Real} <: AbstractSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionMomentumMorris(; surface_tension_coefficient=1.0)
        coefficient = validate_surface_tension_coefficient(surface_tension_coefficient)
        new{typeof(coefficient)}(coefficient)
    end
end

# Surface-model capabilities are expressed through dispatch so that constructors do not need
# to duplicate concrete model checks.
@inline requires_surface_normal(::Nothing) = false
@inline requires_surface_normal(::CohesionForceAkinci) = false
@inline requires_surface_normal(::Any) = true

function create_cache_surface_tension(::SurfaceTensionMomentumMorris, ELTYPE, NDIMS,
                                      nparticles)
    delta_s = Array{ELTYPE, 1}(undef, nparticles)
    # Allocate stress tensor for each particle: NDIMS x NDIMS x nparticles
    stress_tensor = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
    return (; stress_tensor, delta_s)
end

@inline function stress_tensor(particle_system::AbstractFluidSystem, particle)
    return extract_smatrix(particle_system.cache.stress_tensor, particle_system, particle)
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
                                           pos_diff, distance, Val(ndims(particle_system)))
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
    (; surface_tension_coefficient) = surface_tension_a

    # No surface tension with oneself. See `src/general/smoothing_kernels.jl` for more details.
    distance^2 < eps(initial_smoothing_length(particle_system)^2) && return dv_particle

    n_a = surface_normal(particle_system, particle)
    curvature_a = curvature(particle_system, particle)

    dv_particle[] -= surface_tension_correction * surface_tension_coefficient / rho_a *
                     curvature_a * n_a

    return dv_particle
end

function compute_stress_tensors!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

# Section 6 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function compute_stress_tensors!(system::AbstractFluidSystem,
                                 ::SurfaceTensionMomentumMorris,
                                 v, u, v_ode, u_ode, semi, t)
    (; cache) = system
    (; delta_s, stress_tensor) = cache

    # Reset surface stress_tensor
    set_zero!(stress_tensor)

    max_delta_s = maximum(delta_s)
    NDIMS = ndims(system)

    @trixi_timeit timer() "compute surface stress tensor" begin
        @threaded semi for particle in each_integrated_particle(system)
            normal = surface_normal(system, particle)
            delta_s_particle = delta_s[particle]
            if delta_s_particle > eps()
                for i in 1:NDIMS, j in 1:NDIMS
                    delta_ij = (i == j) ? 1 : 0
                    stress_tensor[i, j,
                                  particle] = delta_s_particle *
                                              (delta_ij - normal[i] * normal[j]) -
                                              delta_ij * max_delta_s
                end
            end
        end
    end

    return system
end

function compute_surface_delta_function!(system, surface_tension, semi)
    return system
end

# Eq. 6 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function compute_surface_delta_function!(system, ::SurfaceTensionMomentumMorris, semi)
    (; cache) = system
    (; delta_s) = cache

    set_zero!(delta_s)

    @threaded semi for particle in each_integrated_particle(system)
        delta_s[particle] = norm(surface_normal(system, particle))
    end
    return system
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

    S_a = stress_tensor(particle_system, particle)
    S_b = stress_tensor(neighbor_system, neighbor)

    m_b = hydrodynamic_mass(neighbor_system, neighbor)

    dv_particle[] += surface_tension_correction * surface_tension_coefficient * m_b *
                     (S_a + S_b) / (rho_a * rho_b) * grad_kernel

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

@inline function adhesion_force!(dv_particle, surface_tension, particle_system,
                                 neighbor_system, particle, neighbor, pos_diff, distance)
    return dv_particle
end
