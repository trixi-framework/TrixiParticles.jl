@doc raw"""
    ColorfieldSurfaceNormal()

Color field based computation of the interface normals.
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
# and Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_normal!(system::FluidSystem, neighbor_system::BoundarySystem, u_system, v,
                      v_neighbor_system, u_neighbor_system, semi, surfn, nsurfn)
    (; cache) = system
    (; colorfield, colorfield_bnd) = neighbor_system.boundary_model.cache
    (; boundary_contact_threshold) = surfn

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    # First we need to calculate the smoothed colorfield values of the boundary
    # TODO: move colorfield to extra step
    # TODO: this is only correct for a single fluid

    # Reset to the constant boundary interpolated color values
    colorfield .= colorfield_bnd

    # Accumulate fluid neighbors
    foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        colorfield[neighbor] += hydrodynamic_mass(system, particle) /
                                particle_density(v, system, particle) * system.color *
                                smoothing_kernel(system, distance)
    end

    maximum_colorfield = maximum(colorfield)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        # we assume that we are in contact with the boundary if the color of the boundary particle is larger than the threshold
        if colorfield[neighbor] / maximum_colorfield > boundary_contact_threshold
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

function remove_invalid_normals!(system::FluidSystem, surface_tension, surfn)
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
function remove_invalid_normals!(system::FluidSystem,
                                 surface_tension::Union{SurfaceTensionMorris,
                                                        SurfaceTensionMomentumMorris},
                                 surfn::ColorfieldSurfaceNormal)
    (; cache, smoothing_length, smoothing_kernel, ideal_neighbor_count) = system
    (; ideal_density_threshold, interface_threshold) = surfn

    # We remove invalid normals i.e. they have a small norm (eq. 20)
    normal_condition2 = (interface_threshold /
                         compact_support(smoothing_kernel, smoothing_length))^2

    for particle in each_moving_particle(system)

        # heuristic condition if there is no gas phase to find the free surface
        if ideal_density_threshold > 0 &&
           ideal_density_threshold * ideal_neighbor_count < cache.neighbor_count[particle]
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        particle_surface_normal = cache.surface_normal[1:ndims(system), particle]
        norm2 = dot(particle_surface_normal, particle_surface_normal)

        # see eq. 21
        if norm2 > normal_condition2
            cache.surface_normal[1:ndims(system), particle] = particle_surface_normal /
                                                              sqrt(norm2)
        else
            cache.surface_normal[1:ndims(system), particle] .= 0
        end
    end

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
                         v_neighbor_system, u_neighbor_system, semi, surfn, nsurfn)
end

# Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_curvature!(system::FluidSystem, neighbor_system::FluidSystem, u_system, v,
                         v_neighbor_system, u_neighbor_system, semi,
                         surfn::ColorfieldSurfaceNormal, nsurfn::ColorfieldSurfaceNormal)
    (; cache) = system
    (; curvature) = cache

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)
    correction_factor = fill(eps(eltype(system)), n_moving_particles(system))

    no_valid_neighbors = 0

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = particle_density(v_neighbor_system, neighbor_system, neighbor)
        n_a = surface_normal(system, particle)
        n_b = surface_normal(neighbor_system, neighbor)
        v_b = m_b / rho_b

        # eq. 22 we can test against eps() here since the surface normals that are invalid have been reset
        if dot(n_a, n_a) > eps() && dot(n_b, n_b) > eps()
            w = smoothing_kernel(system, distance)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            for i in 1:ndims(system)
                curvature[particle] += v_b * (n_b[i] - n_a[i]) * grad_kernel[i]
            end
            # eq. 24
            correction_factor[particle] += v_b * w
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

function calc_stress_tensors!(system::FluidSystem, neighbor_system::FluidSystem, u_system,
                              v,
                              v_neighbor_system, u_neighbor_system, semi,
                              surfn::ColorfieldSurfaceNormal,
                              nsurfn::ColorfieldSurfaceNormal)
    (; cache) = system
    (; smoothing_kernel, smoothing_length) = surfn
    (; stress_tensor, delta_s) = cache

    neighbor_cache = neighbor_system.cache
    neighbor_delta_s = neighbor_cache.delta_s

    NDIMS = ndims(system)
    max_delta_s = max(maximum(delta_s), maximum(neighbor_delta_s))

    for particle in each_moving_particle(system)
        normal = surface_normal(system, particle)
        delta_s_particle = delta_s[particle]
        if delta_s_particle > eps()
            for i in 1:NDIMS
                for j in 1:NDIMS
                    delta_ij = (i == j) ? 1.0 : 0.0
                    stress_tensor[i, j, particle] = delta_s_particle *
                                                    (delta_ij - normal[i] * normal[j]) -
                                                    delta_ij * max_delta_s
                end
            end
        end
    end

    return system
end

function compute_stress_tensors!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

# Section 6 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function compute_stress_tensors!(system::FluidSystem, ::SurfaceTensionMomentumMorris,
                                 v, u, v_ode, u_ode, semi, t)
    (; cache) = system
    (; delta_s, stress_tensor) = cache

    # Reset surface stress_tensor
    set_zero!(stress_tensor)

    max_delta_s = maximum(delta_s)
    NDIMS = ndims(system)

    @trixi_timeit timer() "compute surface stress tensor" for particle in each_moving_particle(system)
        normal = surface_normal(system, particle)
        delta_s_particle = delta_s[particle]
        if delta_s_particle > eps()
            for i in 1:NDIMS
                for j in 1:NDIMS
                    delta_ij = (i == j) ? 1.0 : 0.0
                    stress_tensor[i, j, particle] = delta_s_particle *
                                                    (delta_ij - normal[i] * normal[j]) -
                                                    delta_ij * max_delta_s
                end
            end
        end
    end

    return system
end

function compute_surface_delta_function!(system, surface_tension)
    return system
end

# eq. 6 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function compute_surface_delta_function!(system, ::SurfaceTensionMomentumMorris)
    (; cache) = system
    (; delta_s) = cache

    set_zero!(delta_s)

    for particle in each_moving_particle(system)
        delta_s[particle] = norm(surface_normal(system, particle))
    end
    return system
end
