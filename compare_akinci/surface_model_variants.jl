using LinearAlgebra
using TrixiParticles

# Preserve Akinci's published coefficients while removing their quadratic support-radius
# dependence. `calibration_factor` is the coefficient multiplier at
# `reference_support_radius`; both fluid-fluid and fluid-wall forces receive the same
# `(reference_support_radius / support_radius)^2` correction.
struct SurfaceTensionAkinciResolutionInvariant{T} <:
       TrixiParticles.AkinciTypeSurfaceTension
    surface_tension_coefficient::T
    calibration_factor::T
    reference_support_radius::T

    function SurfaceTensionAkinciResolutionInvariant(;
                                                     surface_tension_coefficient,
                                                     calibration_factor=1,
                                                     reference_support_radius)
        values = promote(surface_tension_coefficient, calibration_factor,
                         reference_support_radius)
        coefficient, calibration, reference_support = values
        for (name, value) in (("surface_tension_coefficient", coefficient),
             ("calibration_factor", calibration))
            isfinite(value) && value >= 0 ||
                throw(ArgumentError("`$name` must be finite and non-negative"))
        end
        isfinite(reference_support) && reference_support > 0 ||
            throw(ArgumentError("`reference_support_radius` must be finite and positive"))
        new{typeof(coefficient)}(coefficient, calibration, reference_support)
    end
end

@inline function akinci_resolution_scale(surface_tension, support_radius)
    return surface_tension.calibration_factor *
           (surface_tension.reference_support_radius / support_radius)^2
end

function TrixiParticles.default_surface_normal_method(::SurfaceTensionAkinciResolutionInvariant,
                                                      ::Nothing)
    return ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf)
end

@inline function TrixiParticles.surface_tension_force!(dv_particle,
                                                       surface_tension_a::SurfaceTensionAkinciResolutionInvariant,
                                                       surface_tension_b::SurfaceTensionAkinciResolutionInvariant,
                                                       particle_system::TrixiParticles.AbstractFluidSystem,
                                                       neighbor_system::TrixiParticles.AbstractFluidSystem,
                                                       particle, neighbor, pos_diff,
                                                       distance, rho_a, rho_b, grad_kernel,
                                                       surface_tension_correction)
    support_radius = TrixiParticles.compact_support(TrixiParticles.system_smoothing_kernel(particle_system),
                                                    TrixiParticles.smoothing_length(particle_system,
                                                                                    particle))
    coefficient = surface_tension_a.surface_tension_coefficient *
                  akinci_resolution_scale(surface_tension_a, support_radius)
    standard_model = SurfaceTensionAkinci(;
                                          surface_tension_coefficient=coefficient)
    return TrixiParticles.surface_tension_force!(dv_particle, standard_model,
                                                 standard_model, particle_system,
                                                 neighbor_system, particle, neighbor,
                                                 pos_diff, distance, rho_a, rho_b,
                                                 grad_kernel,
                                                 surface_tension_correction)
end

@inline function TrixiParticles.adhesion_force!(dv_particle,
                                                surface_tension::SurfaceTensionAkinciResolutionInvariant,
                                                particle_system::TrixiParticles.AbstractFluidSystem,
                                                neighbor_system::TrixiParticles.AbstractBoundarySystem,
                                                particle, neighbor, pos_diff, distance)
    adhesion_coefficient = neighbor_system.adhesion_coefficient
    iszero(adhesion_coefficient) && return dv_particle

    support_radius = TrixiParticles.compact_support(TrixiParticles.system_smoothing_kernel(particle_system),
                                                    TrixiParticles.smoothing_length(particle_system,
                                                                                    particle))
    distance >= support_radius && return dv_particle
    distance^2 < eps(support_radius^2) && return dv_particle

    scaled_adhesion = adhesion_coefficient *
                      akinci_resolution_scale(surface_tension, support_radius)
    mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
    dv_particle[] += TrixiParticles.adhesion_force_akinci(surface_tension, support_radius,
                                                          mass_b, pos_diff, distance,
                                                          scaled_adhesion,
                                                          Val(TrixiParticles.ndims(particle_system)))
    return dv_particle
end

# Full Akinci fluid-fluid forces with wall attraction evaluated by the cohesion kernel.
# `wall_cohesion_coefficient` is absolute, rather than a ratio to the fluid coefficient,
# because the normal-difference term also contributes to the fluid-vacuum surface energy.
struct SurfaceTensionAkinciWallCohesion{T} <: TrixiParticles.AkinciTypeSurfaceTension
    surface_tension_coefficient::T
    wall_cohesion_coefficient::T

    function SurfaceTensionAkinciWallCohesion(; surface_tension_coefficient,
                                              wall_cohesion_coefficient)
        values = promote(surface_tension_coefficient, wall_cohesion_coefficient)
        fluid_coefficient, wall_coefficient = values
        isfinite(fluid_coefficient) && fluid_coefficient >= 0 ||
            throw(ArgumentError("`surface_tension_coefficient` must be finite and non-negative"))
        isfinite(wall_coefficient) && wall_coefficient >= 0 ||
            throw(ArgumentError("`wall_cohesion_coefficient` must be finite and non-negative"))
        new{typeof(fluid_coefficient)}(fluid_coefficient, wall_coefficient)
    end
end

function TrixiParticles.default_surface_normal_method(::SurfaceTensionAkinciWallCohesion,
                                                      ::Nothing)
    return ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf)
end

@inline function TrixiParticles.surface_tension_force!(dv_particle,
                                                       surface_tension_a::SurfaceTensionAkinciWallCohesion,
                                                       surface_tension_b::SurfaceTensionAkinciWallCohesion,
                                                       particle_system::TrixiParticles.AbstractFluidSystem,
                                                       neighbor_system::TrixiParticles.AbstractFluidSystem,
                                                       particle, neighbor, pos_diff,
                                                       distance, rho_a, rho_b, grad_kernel,
                                                       surface_tension_correction)
    standard_model = SurfaceTensionAkinci(;
                                          surface_tension_coefficient=surface_tension_a.surface_tension_coefficient)
    return TrixiParticles.surface_tension_force!(dv_particle, standard_model,
                                                 standard_model, particle_system,
                                                 neighbor_system, particle, neighbor,
                                                 pos_diff, distance, rho_a, rho_b,
                                                 grad_kernel,
                                                 surface_tension_correction)
end

@inline function wall_cohesion_force!(dv_particle, wall_cohesion_coefficient,
                                      particle_system, neighbor_system, particle,
                                      neighbor, pos_diff, distance)
    iszero(wall_cohesion_coefficient) && return dv_particle

    support_radius = TrixiParticles.compact_support(TrixiParticles.system_smoothing_kernel(particle_system),
                                                    TrixiParticles.smoothing_length(particle_system,
                                                                                    particle))
    distance >= support_radius && return dv_particle
    distance^2 < eps(support_radius^2) && return dv_particle

    wall_model = CohesionForceAkinci(;
                                     surface_tension_coefficient=wall_cohesion_coefficient)
    mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
    dv_particle[] += TrixiParticles.cohesion_force_akinci(wall_model, support_radius,
                                                          mass_b, pos_diff, distance,
                                                          Val(TrixiParticles.ndims(particle_system)))
    return dv_particle
end

@inline function TrixiParticles.adhesion_force!(dv_particle,
                                                surface_tension::SurfaceTensionAkinciWallCohesion,
                                                particle_system::TrixiParticles.AbstractFluidSystem,
                                                neighbor_system::TrixiParticles.AbstractBoundarySystem,
                                                particle, neighbor, pos_diff, distance)
    return wall_cohesion_force!(dv_particle,
                                surface_tension.wall_cohesion_coefficient,
                                particle_system, neighbor_system, particle, neighbor,
                                pos_diff, distance)
end

# Full Akinci fluid-fluid forces with a tangential CSF contact-line force. This variant
# is specialized to the horizontal Figure 8 plate. Unlike pairwise wall attraction, it
# has no wall-normal component and therefore does not compete with pressure support.
struct SurfaceTensionAkinciContactLine{T, V} <:
       TrixiParticles.AkinciTypeSurfaceTension
    surface_tension_coefficient::T
    contact_line_surface_tension::T
    contact_angle_cos::T
    reference_density::T
    wall_normal::V

    function SurfaceTensionAkinciContactLine(; surface_tension_coefficient,
                                             contact_line_surface_tension,
                                             contact_angle,
                                             reference_density=1000.0,
                                             wall_normal=(0.0, 0.0, 1.0))
        values = promote(surface_tension_coefficient,
                         contact_line_surface_tension,
                         reference_density, wall_normal...)
        fluid_coefficient, line_tension, density = values[1:3]
        wall_normal_ = SVector(values[4:end])
        isfinite(fluid_coefficient) && fluid_coefficient >= 0 ||
            throw(ArgumentError("`surface_tension_coefficient` must be finite and non-negative"))
        isfinite(line_tension) && line_tension >= 0 ||
            throw(ArgumentError("`contact_line_surface_tension` must be finite and non-negative"))
        isfinite(contact_angle) && 0 <= contact_angle <= 180 ||
            throw(ArgumentError("`contact_angle` must be in [0, 180] degrees"))
        isfinite(density) && density > 0 ||
            throw(ArgumentError("`reference_density` must be finite and positive"))
        norm(wall_normal_) > eps() ||
            throw(ArgumentError("`wall_normal` must be nonzero"))
        normal = wall_normal_ / norm(wall_normal_)
        contact_angle_cos = oftype(fluid_coefficient, cosd(contact_angle))
        new{typeof(fluid_coefficient), typeof(normal)}(fluid_coefficient,
                                                       line_tension,
                                                       contact_angle_cos,
                                                       density, normal)
    end
end

function TrixiParticles.default_surface_normal_method(::SurfaceTensionAkinciContactLine,
                                                      ::Nothing)
    return ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf)
end

@inline function TrixiParticles.surface_tension_force!(dv_particle,
                                                       surface_tension_a::SurfaceTensionAkinciContactLine,
                                                       surface_tension_b::SurfaceTensionAkinciContactLine,
                                                       particle_system::TrixiParticles.AbstractFluidSystem,
                                                       neighbor_system::TrixiParticles.AbstractFluidSystem,
                                                       particle, neighbor, pos_diff,
                                                       distance, rho_a, rho_b, grad_kernel,
                                                       surface_tension_correction)
    standard_model = SurfaceTensionAkinci(;
                                          surface_tension_coefficient=surface_tension_a.surface_tension_coefficient)
    return TrixiParticles.surface_tension_force!(dv_particle, standard_model,
                                                 standard_model, particle_system,
                                                 neighbor_system, particle, neighbor,
                                                 pos_diff, distance, rho_a, rho_b,
                                                 grad_kernel,
                                                 surface_tension_correction)
end

@inline function TrixiParticles.adhesion_force!(dv_particle,
                                                surface_tension::SurfaceTensionAkinciContactLine,
                                                particle_system::TrixiParticles.AbstractFluidSystem,
                                                neighbor_system::TrixiParticles.AbstractBoundarySystem,
                                                particle, neighbor, pos_diff, distance)
    coefficient = surface_tension.contact_line_surface_tension
    iszero(coefficient) && return dv_particle

    support_radius = TrixiParticles.compact_support(TrixiParticles.system_smoothing_kernel(particle_system),
                                                    TrixiParticles.smoothing_length(particle_system,
                                                                                    particle))
    distance >= support_radius && return dv_particle
    distance^2 < eps(support_radius^2) && return dv_particle

    color_gradient = TrixiParticles.surface_normal(particle_system, particle)
    surface_delta = norm(color_gradient)
    surface_delta < eps(surface_delta) && return dv_particle
    outward_normal = -color_gradient / surface_delta
    wall_normal = surface_tension.wall_normal
    dynamic_angle_cos = clamp(dot(outward_normal, wall_normal), -one(surface_delta),
                              one(surface_delta))
    tangent = outward_normal - dynamic_angle_cos * wall_normal
    tangent_norm = norm(tangent)
    tangent_norm < sqrt(eps(tangent_norm)) && return dv_particle

    mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
    boundary_volume = mass_b / surface_tension.reference_density
    grad_kernel = TrixiParticles.smoothing_kernel_grad(particle_system, pos_diff,
                                                       distance, particle)
    wall_delta = boundary_volume * abs(dot(grad_kernel, wall_normal))
    contact_line_acceleration = coefficient / surface_tension.reference_density *
                                (surface_tension.contact_angle_cos -
                                 dynamic_angle_cos) * surface_delta * wall_delta *
                                tangent / tangent_norm
    dv_particle[] += contact_line_acceleration
    return dv_particle
end

# Dimensionally normalized Morris CSF with an optional pairwise Akinci cohesion component.
struct SurfaceTensionMorrisAkinci{T} <: TrixiParticles.AbstractSurfaceTension
    surface_tension_coefficient::T
    cohesion_coefficient::T
    wall_cohesion_coefficient::T

    function SurfaceTensionMorrisAkinci(; surface_tension_coefficient,
                                        cohesion_coefficient=0,
                                        wall_cohesion_coefficient=0)
        values = promote(surface_tension_coefficient, cohesion_coefficient,
                         wall_cohesion_coefficient)
        coefficient, cohesion, wall = values
        for (name, value) in (("surface_tension_coefficient", coefficient),
             ("cohesion_coefficient", cohesion),
             ("wall_cohesion_coefficient", wall))
            isfinite(value) && value >= 0 ||
                throw(ArgumentError("`$name` must be finite and non-negative"))
        end
        new{typeof(coefficient)}(coefficient, cohesion, wall)
    end
end

@inline function TrixiParticles.accumulate_surface_divergence_correction!(system,
                                                                          ::SurfaceTensionMorrisAkinci,
                                                                          particle, volume,
                                                                          pos_diff,
                                                                          grad_kernel)
    value = -volume * dot(pos_diff, grad_kernel) / TrixiParticles.ndims(system)
    @inbounds system.cache.support_moment[particle] += value
    return system
end

@inline function TrixiParticles.reset_surface_divergence_correction!(system,
                                                                     ::SurfaceTensionMorrisAkinci)
    TrixiParticles.set_zero!(system.cache.support_moment)
    return system
end

@inline function TrixiParticles.surface_interface_activity(::SurfaceTensionMorrisAkinci,
                                                           system, particle)
    return @inbounds system.cache.interface_activity[particle]
end

function TrixiParticles.create_cache_surface_tension(surface_tension::SurfaceTensionMorrisAkinci,
                                                     ELTYPE, NDIMS, nparticles)
    morris = SurfaceTensionMorris(;
                                  surface_tension_coefficient=surface_tension.surface_tension_coefficient)
    return TrixiParticles.create_cache_surface_tension(morris, ELTYPE, NDIMS, nparticles)
end

function TrixiParticles.remove_invalid_normals!(system::TrixiParticles.AbstractFluidSystem,
                                                surface_tension::SurfaceTensionMorrisAkinci,
                                                surface_normal_method::ColorfieldSurfaceNormal)
    morris = SurfaceTensionMorris(;
                                  surface_tension_coefficient=surface_tension.surface_tension_coefficient)
    return TrixiParticles.remove_invalid_normals!(system, morris,
                                                  surface_normal_method)
end

function TrixiParticles.compute_curvature!(system::TrixiParticles.AbstractFluidSystem,
                                           surface_tension::SurfaceTensionMorrisAkinci,
                                           v, u, v_ode, u_ode, semi, t)
    morris = SurfaceTensionMorris(;
                                  surface_tension_coefficient=surface_tension.surface_tension_coefficient)
    return TrixiParticles.compute_curvature!(system, morris, v, u, v_ode, u_ode,
                                             semi, t)
end

@inline function TrixiParticles.surface_tension_force!(dv_particle,
                                                       surface_tension_a::SurfaceTensionMorrisAkinci,
                                                       surface_tension_b::SurfaceTensionMorrisAkinci,
                                                       particle_system::TrixiParticles.AbstractFluidSystem,
                                                       neighbor_system::TrixiParticles.AbstractFluidSystem,
                                                       particle, neighbor, pos_diff,
                                                       distance, rho_a, rho_b, grad_kernel,
                                                       surface_tension_correction)
    distance^2 < eps(TrixiParticles.initial_smoothing_length(particle_system)^2) &&
        return dv_particle

    if !iszero(surface_tension_a.cohesion_coefficient)
        cohesion_model = CohesionForceAkinci(;
                                             surface_tension_coefficient=surface_tension_a.cohesion_coefficient)
        support_radius = TrixiParticles.compact_support(TrixiParticles.system_smoothing_kernel(particle_system),
                                                        TrixiParticles.smoothing_length(particle_system,
                                                                                        particle))
        mass_b = TrixiParticles.hydrodynamic_mass(neighbor_system, neighbor)
        dv_particle[] += surface_tension_correction *
                         TrixiParticles.cohesion_force_akinci(cohesion_model,
                                                              support_radius, mass_b,
                                                              pos_diff,
                                                              distance,
                                                              Val(TrixiParticles.ndims(particle_system)))
    end

    return dv_particle
end

@inline function TrixiParticles.surface_tension_acceleration(surface_tension::SurfaceTensionMorrisAkinci,
                                                             particle_system, particle,
                                                             rho_a, vector_template)
    morris = SurfaceTensionMorris(;
                                  surface_tension_coefficient=surface_tension.surface_tension_coefficient)
    return TrixiParticles.surface_tension_acceleration(morris, particle_system, particle,
                                                       rho_a, vector_template)
end

@inline function TrixiParticles.adhesion_force!(dv_particle,
                                                surface_tension::SurfaceTensionMorrisAkinci,
                                                particle_system::TrixiParticles.AbstractFluidSystem,
                                                neighbor_system::TrixiParticles.AbstractBoundarySystem,
                                                particle, neighbor, pos_diff, distance)
    return wall_cohesion_force!(dv_particle,
                                surface_tension.wall_cohesion_coefficient,
                                particle_system, neighbor_system, particle, neighbor,
                                pos_diff, distance)
end
