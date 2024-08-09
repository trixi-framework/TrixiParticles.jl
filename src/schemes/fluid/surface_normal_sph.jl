struct ColorfieldSurfaceNormal{ELTYPE, K}
    smoothing_kernel::K
    smoothing_length::ELTYPE
end

function ColorfieldSurfaceNormal(; smoothing_kernel, smoothing_length)
    return ColorfieldSurfaceNormal(smoothing_kernel, smoothing_length)
end

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

function calc_normal!(system, neighbor_system, u_system, v, v_neighbor_system,
                      u_neighbor_system, semi, surfn, nsurfn)
    # Normal not needed
    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# and Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_normal!(system::FluidSystem, neighbor_system::FluidSystem, u_system, v,
                      v_neighbor_system, u_neighbor_system, semi, surfn,
                      ::ColorfieldSurfaceNormal)
    (; cache) = system
    (; smoothing_kernel, smoothing_length) = surfn

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    if smoothing_length != system.smoothing_length ||
       smoothing_kernel !== system.smoothing_kernel
        # TODO: this is really slow but there is no way to easily implement multiple search radia
        search_radius = compact_support(smoothing_kernel, smoothing_length)
        nhs = PointNeighbors.copy_neighborhood_search(nhs, search_radius,
                                                      nparticles(system))
        PointNeighbors.initialize!(nhs, system_coords, neighbor_system_coords)
    end

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = particle_density(v_neighbor_system,
                                            neighbor_system, neighbor)
        grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
        for i in 1:ndims(system)
            cache.surface_normal[i, particle] += m_b / density_neighbor *
                                                 grad_kernel[i]
        end

        cache.neighbor_count[particle] += 1
    end

    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# and Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_normal!(system::FluidSystem, neighbor_system::BoundarySystem, u_system, v,
                      v_neighbor_system, u_neighbor_system, semi, surfn, nsurfn)
    (; cache) = system
    (; colorfield, colorfield_bnd) = neighbor_system.boundary_model.cache
    (; smoothing_kernel, smoothing_length) = surfn

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    # if smoothing_length != system.smoothing_length ||
    #    smoothing_kernel !== system.smoothing_kernel
    #     # TODO: this is really slow but there is no way to easily implement multiple search radia
    #     search_radius = compact_support(smoothing_kernel, smoothing_length)
    #     nhs = PointNeighbors.copy_neighborhood_search(nhs, search_radius,
    #                                                   nparticles(system))
    #     nhs_bnd = PointNeighbors.copy_neighborhood_search(nhs_bnd, search_radius,
    #                                                   nparticles(neighbor_system))
    #     PointNeighbors.initialize!(nhs, system_coords, neighbor_system_coords)
    #     PointNeighbors.initialize!(nhs_bnd, neighbor_system_coords, neighbor_system_coords)
    # end

    # First we need to calculate the smoothed colorfield values
    # TODO: move colorfield to extra step
    # TODO: this is only correct for a single fluid
    set_zero!(colorfield)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        colorfield[neighbor] += kernel(smoothing_kernel, distance, smoothing_length)
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
            grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance,
                                      smoothing_length)
            for i in 1:ndims(system)
                cache.surface_normal[i, particle] += m_b / density_neighbor *
                                                     grad_kernel[i]
            end
            cache.neighbor_count[particle] += 1
        end
    end

    return system
end

function remove_invalid_normals!(system, surface_tension)
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

# see Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function remove_invalid_normals!(system::FluidSystem, surface_tension::SurfaceTensionMorris)
    (; cache, smoothing_length, smoothing_kernel, number_density) = system

    # println("compact_support ", compact_support(smoothing_kernel, smoothing_length))

    # TODO: make settable
    # We remove invalid normals i.e. they have a small norm (eq. 20)
    normal_condition2 = (0.01 / compact_support(smoothing_kernel, smoothing_length))^2

    for particle in each_moving_particle(system)

        # TODO: make selectable
        # TODO: make settable
        # heuristic condition if there is no gas phase to find the free surface
        if 0.75 * number_density < cache.neighbor_count[particle]
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        particle_surface_normal = cache.surface_normal[1:ndims(system), particle]
        norm2 = dot(particle_surface_normal, particle_surface_normal)

        # println(norm2, " > ", normal_condition2)
        # see eq. 21
        if norm2 > normal_condition2
            cache.surface_normal[1:ndims(system), particle] = particle_surface_normal /
                                                              sqrt(norm2)
        else
            cache.surface_normal[1:ndims(system), particle] .= 0
        end
    end

    # println("after removable: ")
    # println(cache.surface_normal)

    return system
end

function compute_surface_normal!(system, surface_normal_method, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_surface_normal!(system::FluidSystem,
                                 surface_normal_method_::ColorfieldSurfaceNormal,
                                 v, u, v_ode, u_ode, semi, t)
    (; cache, surface_tension) = system

    # Reset surface normal
    set_zero!(cache.surface_normal)
    set_zero!(cache.neighbor_count)

    # TODO: if color values are set only different systems need to be called
    # TODO: what to do if there is no gas phase? -> config values
    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_normal!(system, neighbor_system, u, v, v_neighbor_system,
                     u_neighbor_system, semi, surface_normal_method_,
                     surface_normal_method(neighbor_system))
    end
    remove_invalid_normals!(system, surface_tension)

    return system
end

function calc_curvature!(system, neighbor_system, u_system, v,
                         v_neighbor_system, u_neighbor_system, semi, surfn, nsurfn)
end

# Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_curvature!(system::FluidSystem, neighbor_system::FluidSystem, u_system, v,
                         v_neighbor_system, u_neighbor_system, semi,
                         surfn::ColorfieldSurfaceNormal, nsurfn::ColorfieldSurfaceNormal)
    (; cache) = system
    (; smoothing_kernel, smoothing_length) = surfn
    (; curvature) = cache

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)
    correction_factor = fill(eps(eltype(system)), n_moving_particles(system))

    if smoothing_length != system.smoothing_length ||
       smoothing_kernel !== system.smoothing_kernel
        # TODO: this is really slow but there is no way to easily implement multiple search radia
        search_radius = compact_support(smoothing_kernel, smoothing_length)
        nhs = PointNeighbors.copy_neighborhood_search(nhs, search_radius,
                                                      nparticles(system))
        PointNeighbors.initialize!(nhs, system_coords, neighbor_system_coords)
    end

    no_valid_neighbors = 0

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        n_a = surface_normal(system, particle)
        n_b = surface_normal(neighbor_system, neighbor)
        grad_kernel = kernel_grad(smoothing_kernel, pos_diff, distance, smoothing_length)
        v_b = m_b / rho_b

        # eq. 22
        if dot(n_a, n_a) > eps() && dot(n_b, n_b) > eps()
            # for i in 1:ndims(system)
            curvature[particle] += v_b * dot((n_b .- n_a), grad_kernel)
            # end
            # eq. 24
            correction_factor[particle] += v_b * kernel(smoothing_kernel, distance,
                                                  smoothing_length)
            # prevent NaNs from systems that are entirely skipped
            no_valid_neighbors += 1
        end
    end

    # eq. 23
    if no_valid_neighbors > 0
        for i in 1:n_moving_particles(system)
            curvature[i] /= correction_factor[i]
        end
    end

    # println("after curvature")
    # println("surf_norm ", cache.surface_normal)
    # println("curv ", cache.curvature)
    # println("C ", correction_factor)

    return system
end

function compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_curvature!(system::FluidSystem, surface_tension::SurfaceTensionMorris, v,
                            u, v_ode, u_ode, semi, t)
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
