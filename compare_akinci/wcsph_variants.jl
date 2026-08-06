using TrixiParticles

# Experimental controls used to separate the WCSPH pressure discretization from the
# Akinci surface model. They intentionally live in the comparison workbench until the
# Figure 8 study establishes whether either change is useful.

# Three-dimensional continuum constants for planar interfaces. For a central pair force
# `-gamma * m_a * m_b * C(r) * r_hat`, the Akinci cohesion kernel gives
#
#   sigma = pi / 8 * gamma * rho^2 * integral(r^4 C(r), r=0..H)
#         = AKINCI_COHESION_VIRIAL_3D * gamma * rho^2 * H^2.
#
# The second constant is the ratio of the corresponding planar work integral for the
# published adhesion kernel A to that of C. It explains why beta/gamma values below one
# produce only weak wetting when the two kernels are used together.
const AKINCI_COHESION_VIRIAL_3D = 21 / 7040
const AKINCI_ADHESION_TO_COHESION_WORK_3D = 0.10743711881286003
# Exact planar cleavage moment for the Akinci cohesion potential on a cubic lattice
# with `support_radius / particle_spacing = 2.8`.
const AKINCI_COHESION_LATTICE_MOMENT_3D_H28 = 0.0026426355182533943

function akinci_cohesion_surface_tension(coefficient, reference_density, support_radius)
    values = promote(coefficient, reference_density, support_radius)
    coefficient_, density_, support_ = values
    isfinite(coefficient_) && coefficient_ >= 0 ||
        throw(ArgumentError("`coefficient` must be finite and non-negative"))
    isfinite(density_) && density_ > 0 ||
        throw(ArgumentError("`reference_density` must be finite and positive"))
    isfinite(support_) && support_ > 0 ||
        throw(ArgumentError("`support_radius` must be finite and positive"))
    return AKINCI_COHESION_VIRIAL_3D * coefficient_ * density_^2 * support_^2
end

function akinci_cohesion_coefficient(surface_tension, reference_density, support_radius)
    unit_surface_tension = akinci_cohesion_surface_tension(one(surface_tension),
                                                           reference_density,
                                                           support_radius)
    isfinite(surface_tension) && surface_tension >= 0 ||
        throw(ArgumentError("`surface_tension` must be finite and non-negative"))
    return surface_tension / unit_surface_tension
end

function akinci_wall_cohesion_coefficient(surface_tension, contact_angle,
                                          reference_density, support_radius)
    isfinite(contact_angle) && 0 <= contact_angle <= 180 ||
        throw(ArgumentError("`contact_angle` must be in [0, 180] degrees"))
    cohesion_coefficient = akinci_cohesion_coefficient(surface_tension,
                                                       reference_density,
                                                       support_radius)
    return cohesion_coefficient * (1 + cosd(contact_angle)) / 2
end

struct FreeSurfaceDensityDiffusionAntuono{T} <: TrixiParticles.AbstractDensityDiffusion
    delta::T
    reference_density::T
    lower_density_ratio::T
    upper_density_ratio::T

    function FreeSurfaceDensityDiffusionAntuono(; delta, reference_density,
                                                lower_density_ratio=0.6,
                                                upper_density_ratio=0.9)
        values = promote(delta, reference_density, lower_density_ratio,
                         upper_density_ratio)
        delta_, reference_density_, lower_, upper_ = values
        isfinite(delta_) && delta_ >= 0 ||
            throw(ArgumentError("`delta` must be finite and non-negative"))
        isfinite(reference_density_) && reference_density_ > 0 ||
            throw(ArgumentError("`reference_density` must be positive"))
        0 <= lower_ < upper_ <= 1 ||
            throw(ArgumentError("density ratios must satisfy `0 <= lower < upper <= 1`"))
        new{typeof(delta_)}(delta_, reference_density_, lower_, upper_)
    end
end

function Base.show(io::IO, diffusion::FreeSurfaceDensityDiffusionAntuono)
    print(io, "FreeSurfaceDensityDiffusionAntuono(", diffusion.delta, ", ",
          diffusion.lower_density_ratio, "-", diffusion.upper_density_ratio, ")")
end

function TrixiParticles.create_cache_density_diffusion(initial_condition,
                                                       diffusion::FreeSurfaceDensityDiffusionAntuono)
    base = DensityDiffusionAntuono(; delta=diffusion.delta)
    base_cache = TrixiParticles.create_cache_density_diffusion(initial_condition, base)
    free_surface_summation_density = similar(initial_condition.density)
    return (; base_cache..., free_surface_summation_density)
end

function TrixiParticles.update!(diffusion::FreeSurfaceDensityDiffusionAntuono,
                                v, u, system, semi)
    base = DensityDiffusionAntuono(; delta=diffusion.delta)
    TrixiParticles.update!(base, v, u, system, semi)
    summation_density = system.cache.free_surface_summation_density
    TrixiParticles.set_zero!(summation_density)
    coordinates = TrixiParticles.current_coordinates(u, system)
    points = TrixiParticles.each_integrated_particle(system)
    TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                                          points) do particle, neighbor, pos_diff, distance
        mass = TrixiParticles.hydrodynamic_mass(system, neighbor)
        summation_density[particle] += mass *
                                       TrixiParticles.smoothing_kernel(system, distance,
                                                                       particle)
    end
    return diffusion
end

@inline function free_surface_diffusion_weight(diffusion, summation_density)
    ratio = summation_density / diffusion.reference_density
    return clamp((ratio - diffusion.lower_density_ratio) /
                 (diffusion.upper_density_ratio - diffusion.lower_density_ratio), 0, 1)
end

@inline function TrixiParticles.density_diffusion_psi(diffusion::FreeSurfaceDensityDiffusionAntuono,
                                                      rho_a, rho_b, pos_diff, distance,
                                                      system, particle, neighbor)
    summation_density = system.cache.free_surface_summation_density
    weight_a = free_surface_diffusion_weight(diffusion, summation_density[particle])
    weight_b = free_surface_diffusion_weight(diffusion, summation_density[neighbor])
    base = DensityDiffusionAntuono(; delta=diffusion.delta)
    psi = TrixiParticles.density_diffusion_psi(base, rho_a, rho_b, pos_diff, distance,
                                               system, particle, neighbor)
    return min(weight_a, weight_b) * psi
end

struct WCSPHAkinciSurfaceNormal{K, T}
    smoothing_kernel::K
    smoothing_length::T
end

function TrixiParticles.create_cache_surface_normal(::WCSPHAkinciSurfaceNormal,
                                                    ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_normal, neighbor_count)
end

function TrixiParticles.compute_surface_normal!(system::TrixiParticles.AbstractFluidSystem,
                                                method::WCSPHAkinciSurfaceNormal,
                                                v, u, v_ode, u_ode, semi, t)
    cache = system.cache
    TrixiParticles.set_zero!(cache.surface_normal)
    TrixiParticles.set_zero!(cache.neighbor_count)
    system_coordinates = TrixiParticles.current_coordinates(u, system)

    for neighbor_system in semi.systems
        neighbor_system isa TrixiParticles.AbstractFluidSystem || continue
        v_neighbor = TrixiParticles.wrap_v(v_ode, neighbor_system, semi)
        u_neighbor = TrixiParticles.wrap_u(u_ode, neighbor_system, semi)
        neighbor_coordinates = TrixiParticles.current_coordinates(u_neighbor,
                                                                  neighbor_system)
        support_radius = TrixiParticles.compact_support(method.smoothing_kernel,
                                                        method.smoothing_length)
        points = TrixiParticles.each_integrated_particle(system)

        TrixiParticles.foreach_point_neighbor(system, neighbor_system,
                                              system_coordinates, neighbor_coordinates,
                                              semi;
                                              points) do particle, neighbor, pos_diff,
                                                         distance
            distance >= support_radius && return
            mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
            density_b = TrixiParticles.current_density(v_neighbor, neighbor_system,
                                                       neighbor)
            gradient = TrixiParticles.kernel_grad(method.smoothing_kernel, pos_diff,
                                                  distance, method.smoothing_length)
            for dimension in 1:TrixiParticles.ndims(system)
                cache.surface_normal[dimension,
                                     particle] += mass_b / density_b *
                                                  gradient[dimension]
            end
            cache.neighbor_count[particle] += 1
        end
    end

    TrixiParticles.remove_invalid_normals!(system,
                                           TrixiParticles.surface_tension_model(system),
                                           method)
    return system
end

struct SurfaceTensionAkinciWCSPH{T} <: TrixiParticles.AkinciTypeSurfaceTension
    surface_tension_coefficient::T
    support_radius::T
    curvature_factor::T

    function SurfaceTensionAkinciWCSPH(; surface_tension_coefficient, support_radius,
                                       curvature_factor=1)
        values = promote(surface_tension_coefficient, support_radius, curvature_factor)
        coefficient, support, curvature = values
        isfinite(coefficient) && coefficient >= 0 ||
            throw(ArgumentError("`surface_tension_coefficient` must be finite and non-negative"))
        isfinite(support) && support > 0 ||
            throw(ArgumentError("`support_radius` must be finite and positive"))
        isfinite(curvature) && curvature >= 0 ||
            throw(ArgumentError("`curvature_factor` must be finite and non-negative"))
        new{typeof(coefficient)}(coefficient, support, curvature)
    end
end

TrixiParticles.requires_surface_normal(::SurfaceTensionAkinciWCSPH) = true

@inline function TrixiParticles.surface_tension_force!(dv_particle,
                                                       surface_tension_a::SurfaceTensionAkinciWCSPH,
                                                       surface_tension_b::SurfaceTensionAkinciWCSPH,
                                                       particle_system::TrixiParticles.AbstractFluidSystem,
                                                       neighbor_system::TrixiParticles.AbstractFluidSystem,
                                                       particle, neighbor, pos_diff,
                                                       distance, rho_a, rho_b, grad_kernel,
                                                       surface_tension_correction)
    support_radius = surface_tension_a.support_radius
    distance >= support_radius && return dv_particle
    distance^2 < eps(support_radius^2) && return dv_particle

    mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
    normal_a = support_radius * TrixiParticles.surface_normal(particle_system, particle)
    normal_b = support_radius * TrixiParticles.surface_normal(neighbor_system, neighbor)
    dimensions = Val(TrixiParticles.ndims(particle_system))

    dv_particle[] += surface_tension_correction *
                     TrixiParticles.cohesion_force_akinci(surface_tension_a,
                                                          support_radius, mass_b,
                                                          pos_diff, distance, dimensions)
    dv_particle[] -= surface_tension_correction *
                     surface_tension_a.surface_tension_coefficient *
                     surface_tension_a.curvature_factor * (normal_a - normal_b)
    return dv_particle
end

@inline function TrixiParticles.adhesion_force!(dv_particle,
                                                surface_tension::SurfaceTensionAkinciWCSPH,
                                                particle_system::TrixiParticles.AbstractFluidSystem,
                                                neighbor_system::TrixiParticles.AbstractBoundarySystem,
                                                particle, neighbor, pos_diff, distance)
    adhesion_coefficient = neighbor_system.adhesion_coefficient
    abs(adhesion_coefficient) < eps() && return dv_particle
    distance^2 < eps(surface_tension.support_radius^2) && return dv_particle

    mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
    dv_particle[] += TrixiParticles.adhesion_force_akinci(surface_tension,
                                                          surface_tension.support_radius,
                                                          mass_b, pos_diff, distance,
                                                          adhesion_coefficient,
                                                          Val(TrixiParticles.ndims(particle_system)))
    return dv_particle
end
