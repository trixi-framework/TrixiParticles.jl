abstract type AkinciTypeSurfaceTension end

@doc raw"""
cohesionForceAkinci(support_radius, ma, mb, dx, distance)
Use the cohesion force model by Akinci. This is based on an Intra-particle-force formulation.
# Keywords
- 'surface_tension_coefficient=1.0': Coefficient that linearly scales the surface tension induced force.
Reference:
Versatile Surface Tension and Adhesion for SPH Fluids, Akinci et al, 2013, Siggraph Asia
"""

struct CohesionForceAkinci{ELTYPE} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function CohesionForceAkinci(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function (surface_tension::CohesionForceAkinci)(smoothing_length, mb, pos_diff, distance)
    return cohesion_force_akinci(surface_tension, smoothing_length, mb, pos_diff, distance)
end

@doc raw"""
SurfaceTensionAkinci(support_radius, mb, na, nb, dx, distance)
Use the surface tension model by Akinci. This is based on an Intra-particle-force formulation.
# Keywords
- 'surface_tension_coefficient=1.0': Coefficient that linearly scales the surface tension induced force.
Reference:
Versatile Surface Tension and Adhesion for SPH Fluids, Akinci et al, 2013, Siggraph Asia
"""

struct SurfaceTensionAkinci{ELTYPE} <: AkinciTypeSurfaceTension
    surface_tension_coefficient::ELTYPE

    function SurfaceTensionAkinci(; surface_tension_coefficient=1.0)
        new{typeof(surface_tension_coefficient)}(surface_tension_coefficient)
    end
end

function (surface_tension::SurfaceTensionAkinci)(support_radius, mb, na, nb, pos_diff,
                                                 distance)
    (; surface_tension_coefficient) = surface_tension
    return cohesion_force_akinci(surface_tension, support_radius, mb, pos_diff,
                                 distance) .- (surface_tension_coefficient * (na - nb))
end

# just the cohesion force to compensate near boundaries
function (surface_tension::SurfaceTensionAkinci)(support_radius, mb, pos_diff,
                                                 distance)
    return cohesion_force_akinci(surface_tension, support_radius, mb, pos_diff,
                                 distance)
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

# section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: most of the time this only leads to an approximation of the surface normal
function calc_normal_akinci(surface_tension::SurfaceTensionAkinci, u_system,
                            v_neighbor_container, u_neighbor_container,
                            neighborhood_search, system, neighbor_system::FluidSystem)
    (; smoothing_kernel, smoothing_length, cache) = system

    # TODO: swich to for_particle_neighbor
    @threaded for particle in each_moving_particle(system)
        particle_coords = current_coords(u_system, system, particle)

        for neighbor in eachneighbor(particle_coords, neighborhood_search)
            neighbor_coords = current_coords(u_system, system, neighbor)

            pos_diff = particle_coords - neighbor_coords
            distance = norm(pos_diff)
            # correctness strongly depends on this leading to a symmetric distribution of points!
            if sqrt(eps()) < distance <= smoothing_length
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

function calc_normal_akinci(::Any, u_system,
                            v_neighbor_container, u_neighbor_container,
                            neighborhood_search, system,
                            neighbor_system)
    # normal not needed
end
