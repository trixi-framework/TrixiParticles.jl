struct AkinciSurfaceNormal{ELTYPE, K}
    smoothing_kernel::K
    smoothing_length::ELTYPE
end

function AkinciSurfaceNormal(; smoothing_kernel, smoothing_length)
    return AkinciSurfaceNormal(smoothing_kernel, smoothing_length)
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: This is the simplest form of normal approximation commonly used in SPH and comes
# with serious deficits in accuracy especially at corners, small neighborhoods and boundaries
function calc_normal_akinci!(system, neighbor_system::FluidSystem,
                             surface_tension::SurfaceTensionAkinci, u_system,
                             v_neighbor_system, u_neighbor_system,
                             semi, surfn)
    (; cache) = system
    (; smoothing_kernel, smoothing_length) = surfn

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, semi)

    # println("sml ", smoothing_length != system.smoothing_length)
    # println("smk ", smoothing_kernel !== system.smoothing_kernel)
    if smoothing_length != system.smoothing_length ||
       smoothing_kernel !== system.smoothing_kernel
        search_radius = compact_support(smoothing_kernel, smoothing_length)
        nhs = PointNeighbors.copy_neighborhood_search(nhs, search_radius,
                                                      system_coords, neighbor_system_coords)
    end

    for_particle_neighbor(system, neighbor_system,
                          system_coords, neighbor_system_coords,
                          nhs) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = particle_density(v_neighbor_system,
                                            neighbor_system, neighbor)
        grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
        for i in 1:ndims(system)
            # The `smoothing_length` here is used for scaling
            # TODO move this to the surface tension model since this is not a general thing
            cache.surface_normal[i, particle] += m_b / density_neighbor *
                                                 grad_kernel[i] * smoothing_length
        end

        cache.neighbor_count[particle] += 1
    end

    return system
end

function calc_normal_akinci!(system, neighbor_system, surface_tension, u_system,
                             v_neighbor_system, u_neighbor_system,
                             semi, surfn)
    # Normal not needed
    return system
end

function remove_invalid_normals!(system::FluidSystem, surface_tension::SurfaceTensionAkinci)
    (; cache) = system

    # We remove invalid normals (too few neighbors) to reduce the impact of underdefined normals
    for particle in each_moving_particle(system)
        # A corner has that many neighbors assuming a regular 2 * r distribution and a compact_support of 4r
        if cache.neighbor_count[particle] < 2^ndims(system) + 1
            cache.surface_normal[1:ndims(system), particle] .= 0
        end
    end

    return system
end

function remove_invalid_normals!(system, surface_tension)
    # Normal not needed
    return system
end

function compute_surface_normal!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_surface_normal!(system, surface_tension::SurfaceTensionAkinci, v, u, v_ode,
                                 u_ode, semi, t)
    (; cache, surface_normal_method) = system

    # Reset surface normal
    set_zero!(cache.surface_normal)
    set_zero!(cache.neighbor_count)

    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_normal_akinci!(system, neighbor_system, surface_tension, u, v_neighbor_system,
                            u_neighbor_system, semi, surface_normal_method)
        remove_invalid_normals!(system, surface_tension)
    end
    return system
end
