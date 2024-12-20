# TODO: handle corners with StaticNormals
# TODO: add method that determines local static normals
# TODO: add method for TLSPH
@doc raw"""
    StaticNormals(normal_vectors::Tuple{Vararg{ELTYPE, NDIMS}}) where {NDIMS, ELTYPE <: Real}

Represents a unit normal vector and its corresponding unit tangential vector in
2D or 3D space. The input normal vector is normalized to ensure it has a length
of 1, and the tangential vector is computed as a vector perpendicular to the
normal vector.

# Keywords
- `normal_vectors::Tuple{Vararg{ELTYPE, NDIMS}}`: A tuple representing the normal
  vector in NDIMS-dimensional space. It will be normalized internally.

# Tangential Vector Calculation
- In 2D: The tangential vector is calculated as `[-n[2], n[1]]`, ensuring
  it is perpendicular to the normal vector.
- In 3D: The tangential vector is computed using a cross product with a
  reference vector that is not parallel to the normal vector. The result is
  normalized to ensure unit length.

# Errors
- Throws `ArgumentError` if the provided normal vector has a length of 0.

# Example
```julia
sn2d = StaticNormals((3.0, 4.0))  # Normalizes to (0.6, 0.8) and computes tangential (-0.8, 0.6)
sn3d = StaticNormals((0.0, 1.0, 0.0))  # Computes normal and tangential vectors in 3D
```
"""
struct StaticNormals{NDIMS, ELTYPE <: Real}
    normal_vectors::SVector{NDIMS, ELTYPE}
    tangential_vectors::SVector{NDIMS, ELTYPE}
end

function StaticNormals(normal_vectors::Tuple{Vararg{ELTYPE, NDIMS}}) where {NDIMS,
                                                                            ELTYPE <: Real}
    norm_value = sqrt(dot(normal_vectors, normal_vectors))
    if norm_value == 0
        throw(ArgumentError("Normal vector cannot be zero-length."))
    end
    normalized_vector = SVector{NDIMS, ELTYPE}(normal_vectors) / norm_value
    tangential_vector = calculate_tangential_vector(normalized_vector)
    return StaticNormals(normalized_vector, tangential_vector)
end

# Helper function to calculate tangential vector
function calculate_tangential_vector(n::SVector{2, ELTYPE}) where {ELTYPE}
    # Perpendicular vector in 2D
    return SVector(-n[2], n[1])
end

function calculate_tangential_vector(n::SVector{3, ELTYPE}) where {ELTYPE}
    # Cross product with a reference vector to get a perpendicular vector in 3D
    ref = abs(n[1]) < abs(n[2]) ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    t = cross(ref, n)  # Perpendicular to 'n'
    t_norm = norm(t)
    if t_norm == 0
        throw(ArgumentError("Cannot compute tangential vector; invalid input normal vector."))
    end
    return t / t_norm  # Normalize
end

function wall_tangential(particle_system::BoundarySystem, particle)
    wall_tangential(particle_system, particle, particle_system.surface_normal_method)
end

function wall_tangential(particle_system::BoundarySystem, particle, surface_normal_method)
    return zero(SVector{ndims(particle_system), eltype(particle_system)})
end

function wall_tangential(::BoundarySystem, particle, surface_normal_method::StaticNormals)
    return surface_normal_method.tangential_vectors
end

@doc raw"""
    ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                             ideal_density_threshold=0.0)

Implements a model for computing surface normals based on a color field representation.
This approach is commonly used in fluid simulations to determine interface normals
for multiphase flows or free surfaces.

# Keywords
- `boundary_contact_threshold=0.0`: The threshold value used to determine contact with boundaries.
   Adjust this to refine the detection of surface interfaces near boundaries.
- `interface_threshold=0.01`: The threshold value that defines the presence of an interface.
   Lower values can improve sensitivity but may introduce noise.
- `ideal_density_threshold=0.0`: The ideal density threshold used for interface calculations.
   This value can be tuned based on the density variations in the simulation.
"""
struct ColorfieldSurfaceNormal{ELTYPE}
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
end

function ColorfieldSurfaceNormal(; boundary_contact_threshold=0.0, interface_threshold=0.01,
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
    return (; surface_normal, neighbor_count, colorfield)
end

@inline function surface_normal(particle_system::FluidSystem, particle)
    (; cache) = particle_system
    return extract_svector(cache.surface_normal, particle_system, particle)
end

@inline function surface_normal(particle_system::BoundarySystem, particle)
    (; surface_normal_method) = particle_system
    return surface_normal(particle_system, particle, surface_normal_method)
end

@inline function surface_normal(particle_system::BoundarySystem, particle,
                                surface_normal_method)
    return zero(SVector{ndims(particle_system), eltype(particle_system)})
end

@inline function surface_normal(::BoundarySystem, particle,
                                surface_normal_method::StaticNormals)
    return surface_normal_method.normal_vectors
end

function calc_normal!(system, neighbor_system, u_system, v, v_neighbor_system,
                      u_neighbor_system, semi, surfn, nsurfn)
    # Normal not needed
    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# and Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_normal!(system::FluidSystem, neighbor_system::FluidSystem, v, u,
                      v_neighbor_system, u_neighbor_system, semi, surfn,
                      ::ColorfieldSurfaceNormal)
    (; cache) = system

    # TODO: actually calculate if there is more than one colored fluid
    cache.colorfield .= system.color
    system_coords = current_coordinates(u, system)
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
function calc_normal!(system::FluidSystem, neighbor_system::BoundarySystem, v, u,
                      v_neighbor_system, u_neighbor_system, semi, surfn, nsurfn)
    (; cache) = system
    (; colorfield, colorfield_bnd) = neighbor_system.boundary_model.cache
    (; boundary_contact_threshold) = surfn

    system_coords = current_coordinates(u, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(neighbor_system, system, semi)

    # First we need to calculate the smoothed colorfield values of the boundary
    # TODO: move colorfield to extra step
    # TODO: this is only correct for a single fluid

    # # Reset to the constant boundary interpolated color values
    # colorfield .= colorfield_bnd

    # Accumulate fluid neighbors
    foreach_point_neighbor(neighbor_system, system, neighbor_system_coords, system_coords,
                           nhs, points=eachparticle(neighbor_system)) do particle, neighbor, pos_diff, distance
        colorfield[particle] += hydrodynamic_mass(system, neighbor) /
                                particle_density(v, system, neighbor) * system.color *
                                smoothing_kernel(system, distance)
    end

    if boundary_contact_threshold < eps()
        return system
    end

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        # we assume that we are in contact with the boundary if the color of the boundary particle is larger than the threshold
        if colorfield[neighbor] > boundary_contact_threshold
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
# Note: this also normalizes the normals
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
        # We remove normals for particles which have alot of support e.g. they are in the inside
        if ideal_density_threshold > 0 &&
           ideal_density_threshold * ideal_neighbor_count < cache.neighbor_count[particle]
            if surface_tension.contact_model isa HuberContactModel
                cache.normal_v[1:ndims(system), particle] .= 0
            end
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        particle_surface_normal = cache.surface_normal[1:ndims(system), particle]
        norm2 = dot(particle_surface_normal, particle_surface_normal)

        # see eq. 21
        if norm2 > normal_condition2
            if surface_tension.contact_model isa HuberContactModel
                cache.normal_v[1:ndims(system), particle] = particle_surface_normal
            end

            cache.surface_normal[1:ndims(system), particle] = particle_surface_normal /
                                                              sqrt(norm2)
        else
            if surface_tension.contact_model isa HuberContactModel
                cache.normal_v[1:ndims(system), particle] .= 0
            end
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

        calc_normal!(system, neighbor_system, v, u, v_neighbor_system,
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

function calc_wall_distance_vector!(system, neighbor_system,
                                    v, u, v_neighbor_system, u_neighbor_system,
                                    semi, contact_model)
end

# Computes the wall distance vector \( \mathbf{d}_a \) and its normalized version \( \hat{\mathbf{d}}_a \) for each particle in the fluid system.
# The distance vector \( \mathbf{d}_a \) is used to calculate dynamic contact angles and the delta function in wall-contact models.
function calc_wall_distance_vector!(system::FluidSystem, neighbor_system::BoundarySystem,
                                    v, u, v_neighbor_system, u_neighbor_system,
                                    semi, contact_model::HuberContactModel)
    cache = system.cache
    (; d_hat, d_vec) = cache
    NDIMS = ndims(system)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        V_b = hydrodynamic_mass(neighbor_system, neighbor) /
              particle_density(v_neighbor_system, neighbor_system, neighbor)
        W_ab = smoothing_kernel(system, distance)

        # equation 51
        for i in 1:NDIMS
            d_vec[i, particle] += V_b * pos_diff[i] * W_ab
        end
    end

    # d_hat is the unit version of d_vec
    for particle in each_moving_particle(system)
        norm_d = sqrt(sum(d_vec[i, particle]^2 for i in 1:NDIMS))
        if norm_d > eps()
            for i in 1:NDIMS
                d_hat[i, particle] = d_vec[i, particle] / norm_d
            end
        else
            for i in 1:NDIMS
                d_hat[i, particle] = 0.0
            end
        end
    end

    return system
end

function calc_wall_contact_values!(system, neighbor_system,
                                   v, u, v_neighbor_system, u_neighbor_system,
                                   semi, contact_model)
end

function calc_wall_contact_values!(system::FluidSystem, neighbor_system::BoundarySystem,
                                   v, u, v_neighbor_system, u_neighbor_system,
                                   semi, contact_model::HuberContactModel)
    cache = system.cache
    (; d_hat, delta_wns, normal_v, nu_hat, d_vec) = cache
    NDIMS = ndims(system)

    system_coords = current_coordinates(u, system)
    neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
    nhs = get_neighborhood_search(system, neighbor_system, semi)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_coords,
                           nhs) do particle, neighbor, pos_diff, distance
        V_b = hydrodynamic_mass(neighbor_system, neighbor) /
              particle_density(v_neighbor_system, neighbor_system, neighbor)

        grad_W_ab = smoothing_kernel_grad(system, pos_diff, distance)

        # equation 53 computing the tangential direction vector
        for i in 1:NDIMS
            nu_hat[i, particle] = dot(d_vec[:, particle], d_vec[:, particle]) *
                                  normal_v[i, particle] -
                                  (d_vec[i, particle] * normal_v[i, particle]) *
                                  d_vec[i, particle]
        end

        nu_norm = norm(nu_hat[:, particle])
        if nu_norm > eps()
            for i in 1:NDIMS
                nu_hat[i, particle] /= nu_norm
            end
        else
            for i in 1:NDIMS
                nu_hat[i, particle] = 0.0
            end
        end

        # equation 54 delta function
        dot_nu_n = sum(nu_hat[i, particle] * normal_v[i, particle] for i in 1:NDIMS)
        dot_d_gradW = sum(d_hat[i, particle] * grad_W_ab[i] for i in 1:NDIMS)

        delta_wns[particle] += 2 * V_b * dot_d_gradW * dot_nu_n
    end

    return system
end

function compute_wall_contact_values!(system, contact_model, v, u, v_ode, u_ode, semi, t)
end

function compute_wall_contact_values!(system::FluidSystem, contact_model::HuberContactModel,
                                      v, u, v_ode, u_ode, semi, t)
    cache = system.cache
    (; d_hat, delta_wns) = cache

    # Reset d_hat and delta_wns_partial
    set_zero!(d_hat)
    set_zero!(delta_wns)

    # Loop over boundary neighbor systems
    @trixi_timeit timer() "compute wall contact values" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_wall_distance_vector!(system, neighbor_system, v, u,
                                   v_neighbor_system, u_neighbor_system, semi,
                                   contact_model)
    end

    # Loop over boundary neighbor systems
    @trixi_timeit timer() "compute wall contact values" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        # Call calc_wall_distance_vector!
        calc_wall_contact_values!(system, neighbor_system, v, u,
                                  v_neighbor_system, u_neighbor_system, semi,
                                  contact_model)
    end

    return system
end
