@doc raw"""
    ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                              ideal_density_threshold=0.0)

Color field based computation of the interface normals.

# Keywords
- `boundary_contact_threshold=0.1`: If this threshold is reached the fluid is assumed to be in contact with the boundary.
- `interface_threshold=0.01`:       Threshold for normals to be removed as being invalid.
- `ideal_density_threshold=0.0`:    Assume particles are inside if they are above this threshold, which is relative to the `ideal_neighbor_count`.
"""
struct ColorfieldSurfaceNormal{ELTYPE}
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
end

function ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                                 ideal_density_threshold=0.0)
    return ColorfieldSurfaceNormal(boundary_contact_threshold, interface_threshold,
                                   ideal_density_threshold)
end

function create_cache_surface_normal(surface_normal_method, ELTYPE, NDIMS, nparticles)
    return (;)
end

function create_cache_surface_normal(::ColorfieldSurfaceNormal, ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    colorfield = Array{ELTYPE, 1}(undef, nparticles)
    correction_factor = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_normal, neighbor_count, colorfield, correction_factor)
end

@inline function surface_normal(particle_system::AbstractFluidSystem, particle)
    (; cache) = particle_system
    return extract_svector(cache.surface_normal, particle_system, particle)
end

function calc_normal!(system, neighbor_system, u_system, v, v_neighbor_system,
                      u_neighbor_system, semi, surface_normal_method,
                      neighbor_surface_normal_method)
    # Normal not needed
    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# and Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics".
function calc_normal!(system::AbstractFluidSystem, neighbor_system::AbstractFluidSystem,
                      u_system, v,
                      v_neighbor_system, u_neighbor_system, semi, surface_normal_method,
                      ::ColorfieldSurfaceNormal)
    (; cache) = system

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = current_density(v_neighbor_system,
                                           neighbor_system, neighbor)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        for i in 1:ndims(system)
            cache.surface_normal[i, particle] += m_b / density_neighbor * grad_kernel[i]
        end

        cache.neighbor_count[particle] += 1
    end

    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: This is the simplest form of normal approximation commonly used in SPH and comes
# with serious deficits in accuracy especially at corners, small neighborhoods and boundaries
function calc_normal!(system::AbstractFluidSystem, neighbor_system::AbstractBoundarySystem,
                      u_system, v, v_neighbor_system, u_neighbor_system, semi,
                      surface_normal_method, neighbor_surface_normal_method)
    (; cache) = system
    (; colorfield, initial_colorfield) = neighbor_system.boundary_model.cache
    (; boundary_contact_threshold) = surface_normal_method

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # First we need to calculate the smoothed colorfield values of the boundary
    # TODO: move colorfield to extra step
    # TODO: this is only correct for a single fluid

    # Reset to the constant boundary interpolated color values
    colorfield .= initial_colorfield

    # Accumulate fluid neighbors
    foreach_point_neighbor(neighbor_system, system,
                           neighbor_system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        colorfield[particle] += hydrodynamic_mass(system, neighbor) /
                                current_density(v, system, neighbor) * system.cache.color *
                                smoothing_kernel(system, distance, particle)
    end

    maximum_colorfield = maximum(colorfield)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        # We assume that we are in contact with the boundary if the color of the boundary particle
        # is larger than the threshold
        if colorfield[neighbor] / maximum_colorfield > boundary_contact_threshold
            m_b = hydrodynamic_mass(system, particle)
            density_neighbor = current_density(v, system, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            for i in 1:ndims(system)
                cache.surface_normal[i, particle] += m_b / density_neighbor * grad_kernel[i]
            end
            cache.neighbor_count[particle] += 1
        end
    end

    return system
end

function remove_invalid_normals!(system::AbstractFluidSystem, surface_tension,
                                 surface_normal_method)
    (; cache) = system

    # We remove invalid normals (too few neighbors) to reduce the impact of underdefined normals
    for particle in each_integrated_particle(system)
        # A corner has that many neighbors assuming a regular 2 * r distribution and a compact_support of 4r
        if cache.neighbor_count[particle] < 2^ndims(system) + 1
            cache.surface_normal[1:ndims(system), particle] .= 0
        end
    end

    return system
end

# See Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function remove_invalid_normals!(system::AbstractFluidSystem,
                                 surface_tension::Union{SurfaceTensionMorris,
                                                        SurfaceTensionMomentumMorris},
                                 surface_normal_method::ColorfieldSurfaceNormal)
    (; cache, smoothing_kernel) = system
    (; ideal_density_threshold, interface_threshold) = surface_normal_method
    (; neighbor_count) = cache

    smoothing_length_ = initial_smoothing_length(system)

    # We remove invalid normals i.e. they have a small norm (eq. 20)
    normal_condition2 = (interface_threshold /
                         compact_support(smoothing_kernel, smoothing_length_))^2

    for particle in each_integrated_particle(system)

        # Heuristic condition if there is no gas phase to find the free surface.
        # We remove normals for particles which have a lot of support e.g. they are in the interior.
        if ideal_density_threshold > 0 &&
           ideal_density_threshold *
           ideal_neighbor_count(Val(ndims(system)), cache.reference_particle_spacing,
                                compact_support(smoothing_kernel, smoothing_length_)) <
           neighbor_count[particle]
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        particle_surface_normal = surface_normal(system, particle)
        norm2 = dot(particle_surface_normal, particle_surface_normal)

        # See eq. 21
        if norm2 > normal_condition2
            cache.surface_normal[1:ndims(system),
                                 particle] = particle_surface_normal / sqrt(norm2)
        else
            cache.surface_normal[1:ndims(system), particle] .= 0
        end
    end

    return system
end

function compute_surface_normal!(system, surface_normal_method, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_surface_normal!(system::AbstractFluidSystem,
                                 surface_normal_method_::ColorfieldSurfaceNormal,
                                 v, u, v_ode, u_ode, semi, t)
    (; cache, surface_tension) = system

    # Reset surface normal
    set_zero!(cache.surface_normal)
    set_zero!(cache.neighbor_count)

    # TODO: if color values are set only different systems need to be called
    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_normal!(system, neighbor_system, u, v, v_neighbor_system,
                     u_neighbor_system, semi, surface_normal_method_,
                     surface_normal_method(neighbor_system))
    end
    remove_invalid_normals!(system, surface_tension, surface_normal_method_)

    return system
end

function calc_curvature!(system, neighbor_system, u_system, v,
                         v_neighbor_system, u_neighbor_system, semi, surface_normal_method,
                         neighbor_surface_normal_method)
end

# Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_curvature!(system::AbstractFluidSystem, neighbor_system::AbstractFluidSystem,
                         u_system, v, v_neighbor_system, u_neighbor_system, semi,
                         surface_normal_method::ColorfieldSurfaceNormal,
                         neighbor_surface_normal_method::ColorfieldSurfaceNormal)
    (; cache) = system
    (; curvature, correction_factor) = cache

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    set_zero!(correction_factor)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
        n_a = surface_normal(system, particle)
        n_b = surface_normal(neighbor_system, neighbor)
        v_b = m_b / rho_b

        # Eq. 22: we can test against `eps()` here since the surface normals that are invalid have been removed
        if dot(n_a, n_a) > eps() && dot(n_b, n_b) > eps()
            w = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            for i in 1:ndims(system)
                curvature[particle] += v_b * (n_b[i] - n_a[i]) * grad_kernel[i]
            end
            # Eq. 24
            correction_factor[particle] += v_b * w
        end
    end

    # Eq. 23
    for particle in each_integrated_particle(system)
        curvature[particle] /= (correction_factor[particle] + eps())
    end

    return system
end

function compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_curvature!(system::AbstractFluidSystem,
                            surface_tension::SurfaceTensionMorris,
                            v, u, v_ode, u_ode, semi, t)
    (; cache, surface_tension) = system

    # Reset surface curvature
    set_zero!(cache.curvature)

    @trixi_timeit timer() "compute surface curvature" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_curvature!(system, neighbor_system, u, v, v_neighbor_system,
                        u_neighbor_system, semi, surface_normal_method(system),
                        surface_normal_method(neighbor_system))
    end
    return system
end
