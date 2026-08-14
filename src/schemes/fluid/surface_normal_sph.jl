@doc raw"""
    ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                              ideal_density_threshold=0.0)

Color-field-based computation of fluid-interface normals. Interface normals describe local
interface geometry and can be computed for analysis, output, or use by models that require
interface orientation. Every interacting fluid system contributes its `color_value` to the
discrete color gradient, even when that neighboring system does not compute its own normals.

Without a surface-tension model and with [`SurfaceTensionAkinci`](@ref), the stored quantity is
the filtered, unnormalized color gradient. The Morris models store unit normals and retain the
raw gradient magnitude separately where required by the formulation.

# Keywords
- `boundary_contact_threshold=0.1`: Finite value in `[0, 1]`. A dummy-boundary
  particle is treated as being in contact with fluid when the magnitude of its smoothed
  color field, normalized by the maximum magnitude, exceeds this value.
- `interface_threshold=0.01`: Finite, non-negative dimensionless cutoff ``\epsilon``.
  A raw color gradient ``n`` is discarded when ``R\lVert n\rVert \leq \epsilon``, where
  ``R`` is the kernel support radius.
- `ideal_density_threshold=0.0`: Finite value in `[0, 1]` controlling an optional
  neighbor-count heuristic for free surfaces without a represented exterior phase. Zero
  disables the heuristic. Keep this at zero for interfaces between represented phases,
  since those interfaces can have full particle support.
"""
struct ColorfieldSurfaceNormal{ELTYPE}
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
end

function ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                                 ideal_density_threshold=0.0)
    boundary_threshold = validate_surface_normal_threshold(boundary_contact_threshold,
                                                           "boundary_contact_threshold";
                                                           upper_bound=1)
    normal_threshold = validate_surface_normal_threshold(interface_threshold,
                                                         "interface_threshold")
    density_threshold = validate_surface_normal_threshold(ideal_density_threshold,
                                                          "ideal_density_threshold";
                                                          upper_bound=1)
    thresholds = promote(boundary_threshold, normal_threshold, density_threshold)
    return ColorfieldSurfaceNormal(thresholds...)
end

function validate_surface_normal_threshold(threshold, name; upper_bound=nothing)
    if !(threshold isa Real) || !isfinite(threshold) || threshold < 0 ||
       (!isnothing(upper_bound) && threshold > upper_bound)
        interval = isnothing(upper_bound) ? "non-negative" : "in [0, $upper_bound]"
        throw(ArgumentError("`$name` must be a finite real number $interval"))
    end

    return threshold
end

@inline function default_surface_normal_method(surface_tension, surface_normal_method)
    if isnothing(surface_normal_method) && requires_surface_normal(surface_tension)
        return ColorfieldSurfaceNormal()
    end

    return surface_normal_method
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

@inline surface_normal_density(system, particle, density) = density

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
                      v_neighbor_system, u_neighbor_system, semi,
                      surface_normal_method::ColorfieldSurfaceNormal,
                      neighbor_surface_normal_method)
    (; cache) = system
    color_b = neighbor_system.cache.color

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = current_density(v_neighbor_system,
                                           neighbor_system, neighbor)
        density_neighbor = surface_normal_density(neighbor_system, neighbor,
                                                  density_neighbor)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        for i in 1:ndims(system)
            cache.surface_normal[i,
                                 particle] += m_b / density_neighbor * color_b *
                                              grad_kernel[i]
        end

        cache.neighbor_count[particle] += 1
    end

    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: This is the simplest form of normal approximation commonly used in SPH and comes
# with serious deficits in accuracy especially at corners, small neighborhoods and boundaries
function calc_boundary_normal!(system::AbstractFluidSystem, neighbor_system, u_system, v,
                               u_neighbor_system, semi, surface_normal_method)
    (; cache) = system
    (; colorfield, initial_colorfield) = neighbor_system.boundary_model.cache
    (; boundary_contact_threshold) = surface_normal_method

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # First we need to calculate the smoothed colorfield values of the boundary
    # TODO: move colorfield to extra step
    # TODO: this is only correct for a single fluid

    # Contact detection depends on color magnitude, not interface orientation.
    colorfield .= abs.(initial_colorfield)

    # Accumulate fluid neighbors
    foreach_point_neighbor(neighbor_system, system,
                           neighbor_system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        colorfield[particle] += hydrodynamic_mass(system, neighbor) /
                                current_density(v, system, neighbor) *
                                abs(system.cache.color) *
                                smoothing_kernel(system, distance, particle)
    end

    maximum_colorfield = maximum(colorfield)
    iszero(maximum_colorfield) && return system

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

function calc_normal!(system::AbstractFluidSystem, neighbor_system::AbstractBoundarySystem,
                      u_system, v, v_neighbor_system, u_neighbor_system, semi,
                      surface_normal_method, neighbor_surface_normal_method)
    return calc_boundary_normal!(system, neighbor_system, u_system, v, u_neighbor_system,
                                 semi, surface_normal_method)
end

function remove_invalid_normals!(system::AbstractFluidSystem, surface_tension,
                                 surface_normal_method)
    return system
end

function remove_invalid_normals!(system::AbstractFluidSystem,
                                 surface_tension,
                                 surface_normal_method::ColorfieldSurfaceNormal)
    (; cache, smoothing_kernel) = system
    (; ideal_density_threshold, interface_threshold) = surface_normal_method
    (; neighbor_count) = cache

    smoothing_length_ = initial_smoothing_length(system)
    support_radius = compact_support(smoothing_kernel, smoothing_length_)
    minimum_neighbor_count = 2^ndims(system) + 1

    # Eq. 20 in Morris (2000) compares the color-gradient magnitude with ε/h.
    normal_condition2 = (interface_threshold / support_radius)^2
    reset_surface_delta!(system, surface_tension)

    for particle in each_integrated_particle(system)
        # Heuristic condition if there is no gas phase to find the free surface.
        # This must stay disabled for fully supported interfaces between represented phases.
        is_interior = ideal_density_threshold > 0 &&
                      ideal_density_threshold *
                      ideal_neighbor_count(Val(ndims(system)),
                                           cache.reference_particle_spacing,
                                           support_radius) < neighbor_count[particle]

        if neighbor_count[particle] < minimum_neighbor_count || is_interior
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        particle_surface_normal = surface_normal(system, particle)
        norm2 = dot(particle_surface_normal, particle_surface_normal)

        # Eq. 21 in Morris (2000) defines the unit normal after rejecting weak gradients.
        if norm2 > normal_condition2
            normal_magnitude = sqrt(norm2)
            store_surface_delta!(system, surface_tension, particle, normal_magnitude)

            if normalize_surface_normals(surface_tension)
                cache.surface_normal[1:ndims(system),
                                     particle] = particle_surface_normal / normal_magnitude
            end
        else
            cache.surface_normal[1:ndims(system), particle] .= 0
        end
    end

    return system
end

@inline normalize_surface_normals(surface_tension) = false
@inline normalize_surface_normals(::SurfaceTensionMorris) = true
@inline normalize_surface_normals(::SurfaceTensionMomentumMorris) = true

@inline reset_surface_delta!(system, surface_tension) = system

@inline function reset_surface_delta!(system, ::SurfaceTensionMomentumMorris)
    set_zero!(system.cache.delta_s)
    return system
end

@inline store_surface_delta!(system, surface_tension, particle, normal_magnitude) = system

@inline function store_surface_delta!(system, ::SurfaceTensionMomentumMorris, particle,
                                      normal_magnitude)
    system.cache.delta_s[particle] = normal_magnitude
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

    @trixi_timeit timer() "compute surface normal" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            if !has_system_interaction(system, neighbor_system, semi)
                # No interaction between these systems.
                return
            end

            calc_normal!(system, neighbor_system, u, v, v_neighbor_system,
                         u_neighbor_system, semi, surface_normal_method_,
                         surface_normal_method(neighbor_system))
        end
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

    @trixi_timeit timer() "compute surface curvature" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            if !has_system_interaction(system, neighbor_system, semi)
                # No interaction between these systems.
                return
            end

            calc_curvature!(system, neighbor_system, u, v, v_neighbor_system,
                            u_neighbor_system, semi, surface_normal_method(system),
                            surface_normal_method(neighbor_system))
        end
    end
    return system
end
