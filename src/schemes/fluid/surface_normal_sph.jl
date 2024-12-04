@doc raw"""
    ColorfieldSurfaceNormal()

Color field based computation of the interface normals.
"""
struct ColorfieldSurfaceNormal end

function create_cache_surface_normal(surface_normal_method, ELTYPE, NDIMS, nparticles)
    return (;)
end

function create_cache_surface_normal(::ColorfieldSurfaceNormal, ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_normal, neighbor_count)
end

@inline function surface_normal(particle_system::FluidSystem, particle)
    (; cache) = particle_system
    return extract_svector(cache.surface_normal, particle_system, particle)
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: This is the simplest form of normal approximation commonly used in SPH and comes
# with serious deficits in accuracy especially at corners, small neighborhoods and boundaries
function calc_normal_akinci!(system, neighbor_system::FluidSystem, u_system, v,
                             v_neighbor_system, u_neighbor_system, semi, surfn)
    (; cache) = system

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = particle_density(v_neighbor_system,
                                            neighbor_system, neighbor)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
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
function calc_normal_akinci!(system, neighbor_system::BoundarySystem, u_system, v,
                             v_neighbor_system, u_neighbor_system, semi, surfn)
    (; cache) = system
    (; colorfield, colorfield_bnd) = neighbor_system.boundary_model.cache

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    # First we need to calculate the smoothed colorfield values
    # TODO: move colorfield to extra step
    # TODO: this is only correct for a single fluid
    set_zero!(colorfield)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        colorfield[neighbor] += smoothing_kernel(system, distance)
    end

    @threaded neighbor_system for bnd_particle in eachparticle(neighbor_system)
        colorfield[bnd_particle] = colorfield[bnd_particle] / (colorfield[bnd_particle] +
                                    colorfield_bnd[bnd_particle])
    end

    maximum_colorfield = maximum(colorfield)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        if colorfield[neighbor] / maximum_colorfield > 0.1
            m_b = hydrodynamic_mass(system, particle)
            density_neighbor = particle_density(v, system, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance)
            for i in 1:ndims(system)
                cache.surface_normal[i, particle] += m_b / density_neighbor * grad_kernel[i]
            end
            cache.neighbor_count[particle] += 1
        end
    end

    return system
end

function calc_normal_akinci!(system, neighbor_system, u_system, v, v_neighbor_system,
                             u_neighbor_system, semi, surfn)
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

function compute_surface_normal!(system::FluidSystem, surface_tension::SurfaceTensionAkinci,
                                 v, u, v_ode, u_ode, semi, t)
    (; cache, surface_normal_method) = system

    # Reset surface normal
    set_zero!(cache.surface_normal)
    set_zero!(cache.neighbor_count)

    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_normal_akinci!(system, neighbor_system, u, v, v_neighbor_system,
                            u_neighbor_system, semi, surface_normal_method)
        remove_invalid_normals!(system, surface_tension)
    end
    return system
end
