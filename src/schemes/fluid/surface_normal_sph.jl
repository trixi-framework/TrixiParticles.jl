struct AkinciSurfaceNormal{ELTYPE, K}
    smoothing_kernel                  :: K
    smoothing_length                  :: ELTYPE
end

function AkinciSurfaceNormal(; smoothing_kernel, smoothing_length)
    return AkinciSurfaceNormal(smoothing_kernel, smoothing_length)
end



# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: most of the time this only leads to an approximation of the surface normal
function calc_normal_akinci!(system, neighbor_system::FluidSystem,
                             surface_tension::SurfaceTensionAkinci, u_system,
                             v_neighbor_system, u_neighbor_system,
                             neighborhood_search)
    (; smoothing_length, cache) = system

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)


    for_particle_neighbor(system, neighbor_system,
                          system_coords, neighbor_system_coords,
                          neighborhood_search) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = particle_density(v_neighbor_system,
                                            neighbor_system, neighbor)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance,
                                            particle)
        for i in 1:ndims(system)
            cache.surface_normal[i, particle] += m_b / density_neighbor *
                                                 grad_kernel[i] * smoothing_length
        end
    end

    return system
end

function calc_normal_akinci!(system, neighbor_system, surface_tension, u_system,
                             v_neighbor_system, u_neighbor_system,
                             neighborhood_search)
    # Normal not needed
    return system
end
