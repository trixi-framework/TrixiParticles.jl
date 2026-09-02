abstract type AbstractSurfaceMethod end
abstract type AbstractSurfaceNormalMethod <: AbstractSurfaceMethod end

@doc raw"""
    ColorfieldSurfaceDetection(; boundary_contact_threshold=0.1,
                                 interface_threshold=0.01,
                                 ideal_density_threshold=0.0,
                                 interface_taper_start=0.8,
                                 interpolation_surface_threshold=0.45)

Detect fluid surfaces from the magnitude of a colorfield gradient. `color_value` is a
categorical phase identifier: unequal values detect represented fluid-fluid interfaces,
while incomplete same-phase support detects a free surface. This method computes
[`surface_activity`](@ref), but does not expose a surface normal.
"""
struct ColorfieldSurfaceDetection{ELTYPE} <: AbstractSurfaceMethod
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
    interface_taper_start::ELTYPE
    interpolation_surface_threshold::ELTYPE
end

@doc raw"""
    ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1,
                              interface_threshold=0.01,
                              ideal_density_threshold=0.0,
                              interface_taper_start=0.8,
                              interpolation_surface_threshold=0.45)

Compute colorfield surface normals and [`surface_activity`](@ref). The detection stage is
identical to [`ColorfieldSurfaceDetection`](@ref). The raw gradient is filtered after its
magnitude has been converted to activity and is normalized when required by the configured
surface-tension model.

# Keywords
- `boundary_contact_threshold=0.1`: Finite value in `[0, 1]` used to detect contact with
  dummy-particle boundaries.
- `interface_threshold=0.01`: Finite, non-negative dimensionless gradient threshold.
- `ideal_density_threshold=0.0`: Optional neighbor-count heuristic for unrepresented exterior
  phases. Zero disables it; keep it disabled for fully supported multiphase interfaces.
- `interface_taper_start=0.8`: Start of the smooth activity transition as a fraction of
  `interface_threshold`.
- `interpolation_surface_threshold=0.45`: Minimum normalized reference-color contribution
  retained by interpolated output.
"""
struct ColorfieldSurfaceNormal{ELTYPE} <: AbstractSurfaceNormalMethod
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
    interface_taper_start::ELTYPE
    interpolation_surface_threshold::ELTYPE
end

const ColorfieldSurfaceMethod = Union{ColorfieldSurfaceDetection, ColorfieldSurfaceNormal}

function validate_surface_threshold(threshold, name; upper_bound=nothing,
                                    strict_upper_bound=false)
    interval = if isnothing(upper_bound)
        "non-negative"
    elseif strict_upper_bound
        "in [0, $upper_bound)"
    else
        "in [0, $upper_bound]"
    end
    threshold isa Real ||
        throw(ArgumentError("`$name` must be a finite real number $interval"))

    valid_upper_bound = isnothing(upper_bound) ||
                        (strict_upper_bound ? threshold < upper_bound :
                         threshold <= upper_bound)
    if !isfinite(threshold) || threshold < 0 || !valid_upper_bound
        throw(ArgumentError("`$name` must be a finite real number $interval"))
    end

    return threshold
end

function colorfield_surface_parameters(; boundary_contact_threshold=0.1,
                                       interface_threshold=0.01,
                                       ideal_density_threshold=0.0,
                                       interface_taper_start=nothing,
                                       interpolation_surface_threshold=nothing)
    boundary_threshold = validate_surface_threshold(boundary_contact_threshold,
                                                    "boundary_contact_threshold";
                                                    upper_bound=1)
    normal_threshold = validate_surface_threshold(interface_threshold,
                                                  "interface_threshold")
    density_threshold = validate_surface_threshold(ideal_density_threshold,
                                                   "ideal_density_threshold";
                                                   upper_bound=1)
    base_parameters = promote(boundary_threshold, normal_threshold, density_threshold)
    BASE_ELTYPE = eltype(base_parameters) <: Integer ? Float64 : eltype(base_parameters)
    taper_start_ = isnothing(interface_taper_start) ? convert(BASE_ELTYPE, 0.8) :
                   interface_taper_start
    interpolation_threshold_ = isnothing(interpolation_surface_threshold) ?
                               convert(BASE_ELTYPE, 0.45) :
                               interpolation_surface_threshold
    taper_start = validate_surface_threshold(taper_start_,
                                             "interface_taper_start";
                                             upper_bound=1,
                                             strict_upper_bound=true)
    interpolation_threshold = validate_surface_threshold(interpolation_threshold_,
                                                         "interpolation_surface_threshold";
                                                         upper_bound=1)
    parameters = promote(boundary_threshold, normal_threshold, density_threshold,
                         taper_start, interpolation_threshold)
    return eltype(parameters) <: Integer ? float.(parameters) : parameters
end

function ColorfieldSurfaceDetection(; kwargs...)
    return ColorfieldSurfaceDetection(colorfield_surface_parameters(; kwargs...)...)
end

function ColorfieldSurfaceNormal(; kwargs...)
    return ColorfieldSurfaceNormal(colorfield_surface_parameters(; kwargs...)...)
end

function ColorfieldSurfaceNormal(boundary_contact_threshold, interface_threshold,
                                 ideal_density_threshold)
    ELTYPE = promote_type(typeof(boundary_contact_threshold), typeof(interface_threshold),
                          typeof(ideal_density_threshold))
    return ColorfieldSurfaceNormal(; boundary_contact_threshold, interface_threshold,
                                   ideal_density_threshold,
                                   interface_taper_start=convert(ELTYPE, 0.8),
                                   interpolation_surface_threshold=convert(ELTYPE, 0.45))
end

@inline computes_surface_normal(surface_method) = false
@inline computes_surface_normal(::AbstractSurfaceNormalMethod) = true

@inline is_colorfield_surface_method(surface_method) = false
@inline is_colorfield_surface_method(::ColorfieldSurfaceMethod) = true

function check_surface_method_eltype(surface_method::Union{ColorfieldSurfaceDetection{METHOD_ELTYPE},
                                                           ColorfieldSurfaceNormal{METHOD_ELTYPE}},
                                     ::Type{SYSTEM_ELTYPE}) where {METHOD_ELTYPE,
                                                                   SYSTEM_ELTYPE}
    METHOD_ELTYPE === SYSTEM_ELTYPE && return surface_method
    throw(ArgumentError("the element type of `$surface_method` must match the system element type `$SYSTEM_ELTYPE`"))
end

@inline check_surface_method_eltype(surface_method,
                                    ::Type{ELTYPE}) where {ELTYPE} = surface_method

@inline contributes_boundary_colorfield(system) = false

@inline function colorfield_phase_weight(color_a, color_b, ::Type{ELTYPE}) where {ELTYPE}
    return ifelse(color_a == color_b, one(ELTYPE), zero(ELTYPE))
end

@inline function default_surface_method(surface_tension, surface_method,
                                        ::Type{ELTYPE}) where {ELTYPE}
    if isnothing(surface_method) && requires_surface_normal(surface_tension)
        return ColorfieldSurfaceNormal(convert(ELTYPE, 0.1), convert(ELTYPE, 0.01),
                                       zero(ELTYPE), convert(ELTYPE, 0.8),
                                       convert(ELTYPE, 0.45))
    end

    return surface_method
end

function select_surface_method(surface_tension, surface_method, surface_normal_method,
                               ELTYPE)
    if !isnothing(surface_method) && !isnothing(surface_normal_method)
        throw(ArgumentError("`surface_method` and deprecated `surface_normal_method` cannot both be set"))
    end

    if !isnothing(surface_normal_method)
        Base.depwarn("`surface_normal_method` is deprecated; use `surface_method` instead",
                     :surface_normal_method)
        surface_method = surface_normal_method
    end

    surface_method = default_surface_method(surface_tension, surface_method, ELTYPE)
    if !(surface_method isa Union{Nothing, AbstractSurfaceMethod})
        throw(ArgumentError("`surface_method` must be an `AbstractSurfaceMethod` or `nothing`"))
    end
    if requires_surface_normal(surface_tension) && !computes_surface_normal(surface_method)
        throw(ArgumentError("$(typeof(surface_tension)) requires a surface method that computes surface normals"))
    end

    return surface_method
end

@inline function cubic_smoothstep(value)
    value <= zero(value) && return zero(value)
    value >= one(value) && return one(value)
    return value^2 * (3 - 2value)
end

@inline function gradient_surface_activity(normal_norm, support_radius,
                                           surface_method::ColorfieldSurfaceMethod)
    threshold = surface_method.interface_threshold
    dimensionless_norm = support_radius * normal_norm
    if iszero(threshold)
        return iszero(dimensionless_norm) ? zero(dimensionless_norm) :
               one(dimensionless_norm)
    end

    lower_bound = surface_method.interface_taper_start * threshold
    transition_coordinate = (dimensionless_norm - lower_bound) /
                            (threshold - lower_bound)
    return cubic_smoothstep(transition_coordinate)
end

function create_cache_surface(surface_method, ELTYPE, NDIMS, nparticles)
    return (;)
end

function create_cache_surface(::ColorfieldSurfaceDetection, ELTYPE, NDIMS, nparticles)
    surface_gradient = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    surface_activity = Array{ELTYPE, 1}(undef, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    colorfield = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_gradient, surface_activity, neighbor_count, colorfield)
end

function create_cache_surface(::ColorfieldSurfaceNormal, ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    surface_activity = Array{ELTYPE, 1}(undef, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    colorfield = Array{ELTYPE, 1}(undef, nparticles)
    correction_factor = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_normal, surface_activity, neighbor_count, colorfield,
            correction_factor)
end

@inline function surface_gradient(cache, ::ColorfieldSurfaceDetection)
    return cache.surface_gradient
end

@inline function surface_gradient(cache, ::ColorfieldSurfaceNormal)
    return cache.surface_normal
end

@inline function surface_normal(particle_system::AbstractFluidSystem, particle)
    return extract_svector(particle_system.cache.surface_normal, particle_system, particle)
end

@inline function surface_activity(particle_system::AbstractFluidSystem, particle)
    return @inbounds particle_system.cache.surface_activity[particle]
end

function calc_surface!(system, neighbor_system, u_system, v, v_neighbor_system,
                       u_neighbor_system, semi, surface_method, neighbor_surface_method)
    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# and Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics".
function calc_surface!(system::AbstractFluidSystem,
                       neighbor_system::AbstractFluidSystem,
                       u_system, v, v_neighbor_system, u_neighbor_system, semi,
                       surface_method::ColorfieldSurfaceMethod,
                       neighbor_surface_method)
    contributes_to_colorfield(neighbor_system) || return system

    (; cache) = system
    gradient = surface_gradient(cache, surface_method)
    phase_weight = colorfield_phase_weight(system.cache.color,
                                           neighbor_system.cache.color,
                                           eltype(system))
    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        volume_b = hydrodynamic_mass(neighbor_system, neighbor) /
                   current_density(v_neighbor_system, neighbor_system, neighbor)
        grad_kernel = kernel_grad(system_smoothing_kernel(system), pos_diff, distance,
                                  smoothing_length(system, particle))
        for i in 1:ndims(system)
            gradient[i, particle] += volume_b * phase_weight * grad_kernel[i]
        end
        cache.neighbor_count[particle] += 1
    end

    return system
end

function calc_boundary_surface!(system::AbstractFluidSystem, neighbor_system, u_system, v,
                                u_neighbor_system, semi,
                                surface_method::ColorfieldSurfaceMethod)
    (; cache) = system
    gradient = surface_gradient(cache, surface_method)
    (; colorfield, initial_colorfield) = neighbor_system.boundary_model.cache
    (; boundary_contact_threshold) = surface_method

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    colorfield .= initial_colorfield
    foreach_point_neighbor(neighbor_system, system,
                           neighbor_system_coords, system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        colorfield[particle] += hydrodynamic_mass(system, neighbor) /
                                current_density(v, system, neighbor) *
                                smoothing_kernel(system, distance, particle)
    end

    maximum_colorfield = maximum(colorfield)
    iszero(maximum_colorfield) && return system

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        if colorfield[neighbor] / maximum_colorfield > boundary_contact_threshold
            volume_a = hydrodynamic_mass(system, particle) /
                       current_density(v, system, particle)
            grad_kernel = kernel_grad(system_smoothing_kernel(system), pos_diff, distance,
                                      smoothing_length(system, particle))
            for i in 1:ndims(system)
                gradient[i, particle] += volume_a * grad_kernel[i]
            end
            cache.neighbor_count[particle] += 1
        end
    end

    return system
end

function calc_surface!(system::AbstractFluidSystem,
                       neighbor_system::AbstractBoundarySystem,
                       u_system, v, v_neighbor_system, u_neighbor_system, semi,
                       surface_method::ColorfieldSurfaceMethod,
                       neighbor_surface_method)
    contributes_boundary_colorfield(neighbor_system) || return system

    return calc_boundary_surface!(system, neighbor_system, u_system, v, u_neighbor_system,
                                  semi, surface_method)
end

@inline function invalid_surface_support(neighbor_count, NDIMS, reference_particle_spacing,
                                         support_radius,
                                         surface_method::ColorfieldSurfaceMethod)
    minimum_neighbor_count = 2^NDIMS + 1
    neighbor_count < minimum_neighbor_count && return true

    threshold = surface_method.ideal_density_threshold
    return threshold > 0 &&
           threshold * ideal_neighbor_count(Val(NDIMS), reference_particle_spacing,
                                support_radius) < neighbor_count
end

function invalid_surface_particle(system, surface_method::ColorfieldSurfaceMethod,
                                  particle, support_radius)
    return invalid_surface_support(system.cache.neighbor_count[particle], ndims(system),
                                   system.cache.reference_particle_spacing,
                                   support_radius, surface_method)
end

function finalize_surface!(system::AbstractFluidSystem, surface_tension,
                           surface_method::ColorfieldSurfaceMethod, semi)
    gradient = surface_gradient(system.cache, surface_method)
    support_radius = compact_support(system_smoothing_kernel(system),
                                     initial_smoothing_length(system))

    @threaded semi for particle in each_integrated_particle(system)
        particle_gradient = extract_svector(gradient, system, particle)
        gradient_norm = norm(particle_gradient)
        activity = gradient_surface_activity(gradient_norm, support_radius, surface_method)

        if invalid_surface_particle(system, surface_method, particle, support_radius)
            @inbounds system.cache.surface_activity[particle] = zero(activity)
            for i in 1:ndims(system)
                @inbounds gradient[i, particle] = zero(eltype(gradient))
            end
        else
            @inbounds system.cache.surface_activity[particle] = activity
        end
    end

    return system
end

function finalize_surface!(system::AbstractFluidSystem, surface_tension,
                           surface_method::ColorfieldSurfaceNormal, semi)
    gradient = surface_gradient(system.cache, surface_method)
    support_radius = compact_support(system_smoothing_kernel(system),
                                     initial_smoothing_length(system))
    normal_condition2 = (surface_method.interface_threshold / support_radius)^2
    reset_surface_delta!(system, surface_tension)

    @threaded semi for particle in each_integrated_particle(system)
        particle_gradient = extract_svector(gradient, system, particle)
        norm2 = dot(particle_gradient, particle_gradient)
        gradient_norm = sqrt(norm2)
        activity = gradient_surface_activity(gradient_norm, support_radius, surface_method)

        if invalid_surface_particle(system, surface_method, particle, support_radius)
            @inbounds system.cache.surface_activity[particle] = zero(activity)
            for i in 1:ndims(system)
                @inbounds gradient[i, particle] = zero(eltype(gradient))
            end
        elseif norm2 > normal_condition2
            @inbounds system.cache.surface_activity[particle] = activity
            store_surface_delta!(system, surface_tension, particle, gradient_norm)
            if normalize_surface_normals(surface_tension)
                for i in 1:ndims(system)
                    @inbounds gradient[i, particle] = particle_gradient[i] / gradient_norm
                end
            end
        else
            @inbounds system.cache.surface_activity[particle] = activity
            for i in 1:ndims(system)
                @inbounds gradient[i, particle] = zero(eltype(gradient))
            end
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

@inline store_surface_delta!(system, surface_tension, particle, gradient_norm) = system

@inline function store_surface_delta!(system, ::SurfaceTensionMomentumMorris, particle,
                                      gradient_norm)
    @inbounds system.cache.delta_s[particle] = gradient_norm
    return system
end

@inline function normalized_surface_normal(system, particle)
    normal = surface_normal(system, particle)
    norm2 = dot(normal, normal)
    # Surface-method filtering already uses a dimensionless threshold. Applying machine
    # epsilon to this dimensionful gradient would make normal validity depend on scale.
    return iszero(norm2) ? zero(normal) : normal / sqrt(norm2)
end

function compute_surface!(system, surface_method, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_surface!(system::AbstractFluidSystem,
                          surface_method_::ColorfieldSurfaceMethod,
                          v, u, v_ode, u_ode, semi, t)
    (; cache, surface_tension) = system

    set_zero!(surface_gradient(cache, surface_method_))
    set_zero!(cache.surface_activity)
    set_zero!(cache.neighbor_count)

    @trixi_timeit timer() "compute surface" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            has_system_interaction(system, neighbor_system, semi) || return

            calc_surface!(system, neighbor_system, u, v, v_neighbor_system,
                          u_neighbor_system, semi, surface_method_,
                          surface_method(neighbor_system))
        end
    end
    finalize_surface!(system, surface_tension, surface_method_, semi)

    return system
end

function calc_curvature!(system, neighbor_system, u_system, v,
                         v_neighbor_system, u_neighbor_system, semi, surface_method,
                         neighbor_surface_method)
end

# Section 5 in Morris 2000 "Simulating surface tension with smoothed particle hydrodynamics"
function calc_curvature!(system::AbstractFluidSystem, neighbor_system::AbstractFluidSystem,
                         u_system, v, v_neighbor_system, u_neighbor_system, semi,
                         surface_method_::ColorfieldSurfaceNormal,
                         neighbor_surface_method::ColorfieldSurfaceNormal)
    (; cache) = system
    (; curvature, correction_factor) = cache

    system_coords = current_coordinates(u_system, system)
    neighbor_system_coords = current_coordinates(u_neighbor_system, neighbor_system)

    # `curvature` and `correction_factor` are zeroed once in `compute_curvature!` and
    # normalized once after all neighboring systems have contributed, so the result is
    # independent of system ordering and same-phase partitioning.
    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
        # Normals are phase-local. Reorient unlike-phase normals into the target phase
        # before taking their divergence, so a planar interface has a consistent stencil.
        n_a = normalized_surface_normal(system, particle)
        n_b = normalized_surface_normal(neighbor_system, neighbor)
        system.cache.color == neighbor_system.cache.color || (n_b = -n_b)
        v_b = m_b / rho_b

        if !iszero(dot(n_a, n_a)) && !iszero(dot(n_b, n_b))
            w = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

            for i in 1:ndims(system)
                curvature[particle] += v_b * (n_b[i] - n_a[i]) * grad_kernel[i]
            end
            correction_factor[particle] += v_b * w
        end
    end

    return system
end

function compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_curvature!(system::AbstractFluidSystem,
                            surface_tension::SurfaceTensionMorris,
                            v, u, v_ode, u_ode, semi, t)
    (; cache) = system
    (; curvature, correction_factor) = cache

    set_zero!(curvature)
    set_zero!(correction_factor)

    @trixi_timeit timer() "compute surface curvature" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            has_system_interaction(system, neighbor_system, semi) || return

            calc_curvature!(system, neighbor_system, u, v, v_neighbor_system,
                            u_neighbor_system, semi, surface_method(system),
                            surface_method(neighbor_system))
        end
    end

    # Normalize once after all neighboring systems have contributed.
    threshold = eps(eltype(system))
    @threaded semi for particle in each_integrated_particle(system)
        correction = correction_factor[particle]
        if correction > threshold
            curvature[particle] /= correction
        else
            curvature[particle] = zero(correction)
        end
    end

    return system
end
