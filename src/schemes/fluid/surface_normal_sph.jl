@doc raw"""
    ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                              ideal_density_threshold=0.0, interface_taper_start=0.8,
                              support_taper_width=0.025)

Color field based computation of the interface normals.

# Keywords
- `boundary_contact_threshold=0.1`: If this threshold is reached the fluid is assumed to be in contact with the boundary.
- `interface_threshold=0.01`:       Threshold for normals to be removed as being invalid.
- `ideal_density_threshold=0.0`:    For Morris CSF, assume particles are inside when their
                                    continuous kernel-support moment is above this fraction of
                                    complete support. Zero disables this filter. Other models
                                    retain their existing neighbor-count interpretation.
- `interface_taper_start=0.8`:      Start Morris CSF interface activation at this fraction of
                                    `interface_threshold`.
- `support_taper_width=0.025`:      Width of the Morris CSF support-moment transition above
                                    `ideal_density_threshold`.
"""
struct ColorfieldSurfaceNormal{ELTYPE}
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
    interface_taper_start::ELTYPE
    support_taper_width::ELTYPE
end

@doc raw"""
    CorrectedCSFSurfaceNormal(; contact_angle=nothing)

Interface geometry for the corrected continuous-surface-force (C-CSF) method of Vergnaud
et al. (2022). The outward unit normal is computed from the renormalized gradient of the
smallest eigenvalue of the first-order kernel moment. Curvature uses the corresponding
renormalized divergence with the published thin-jet angular filter, and the surface delta
uses the published Shepard correction.

With `contact_angle=nothing`, this explicit opt-in implements the single-fluid free-surface core
(equations 15--25) with [`SurfaceTensionMorris`](@ref). A finite `contact_angle` in degrees enables
the planar boundary-integral geometry and contact-normal correction from equations 41--50. The
angle is measured between the wall normal pointing into the fluid and the outward interface
normal. Contact boundaries require a stationary `WallBoundarySystem`, a three-dimensional
Wendland C2 kernel, explicit surface measures, and normal offset vectors. These terms correct
interface geometry only; hydrodynamic wall interactions remain those of the configured boundary
model.
"""
struct CorrectedCSFSurfaceNormal{CONTACT_ANGLE}
    contact_angle::CONTACT_ANGLE

    function CorrectedCSFSurfaceNormal(contact_angle::Nothing)
        return new{Nothing}(contact_angle)
    end

    function CorrectedCSFSurfaceNormal(contact_angle::Real)
        if !isfinite(contact_angle) || !(0 <= contact_angle <= 180)
            throw(ArgumentError("`contact_angle` must be a finite real number in [0, 180] degrees"))
        end
        angle = float(contact_angle)
        return new{typeof(angle)}(angle)
    end
end

function CorrectedCSFSurfaceNormal(; contact_angle=nothing)
    return CorrectedCSFSurfaceNormal(contact_angle)
end

@inline validate_corrected_csf(surface_normal_method, surface_tension) = nothing

function validate_corrected_csf(::CorrectedCSFSurfaceNormal, surface_tension)
    surface_tension isa SurfaceTensionMorris ||
        throw(ArgumentError("`CorrectedCSFSurfaceNormal` requires `SurfaceTensionMorris`"))
    return nothing
end

function ColorfieldSurfaceNormal(boundary_contact_threshold, interface_threshold,
                                 ideal_density_threshold)
    return ColorfieldSurfaceNormal(; boundary_contact_threshold, interface_threshold,
                                   ideal_density_threshold)
end

function ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                                 ideal_density_threshold=0.0, interface_taper_start=0.8,
                                 support_taper_width=0.025)
    if !(boundary_contact_threshold isa Real) || isnan(boundary_contact_threshold) ||
       boundary_contact_threshold < 0
        throw(ArgumentError("`boundary_contact_threshold` must be non-negative and not NaN"))
    end
    if !(interface_threshold isa Real) || !isfinite(interface_threshold) ||
       interface_threshold < 0
        throw(ArgumentError("`interface_threshold` must be finite and non-negative"))
    end
    if !(ideal_density_threshold isa Real) || !isfinite(ideal_density_threshold) ||
       ideal_density_threshold < 0
        throw(ArgumentError("`ideal_density_threshold` must be finite and non-negative"))
    end
    if !(interface_taper_start isa Real) || !isfinite(interface_taper_start) ||
       !(0 <= interface_taper_start < 1)
        throw(ArgumentError("`interface_taper_start` must be finite and in [0, 1)"))
    end
    if !(support_taper_width isa Real) || !isfinite(support_taper_width) ||
       support_taper_width <= 0
        throw(ArgumentError("`support_taper_width` must be finite and positive"))
    end

    thresholds = promote(boundary_contact_threshold, interface_threshold,
                         ideal_density_threshold)
    ELTYPE = typeof(first(thresholds))
    if ELTYPE <: Integer
        thresholds = float.(thresholds)
        ELTYPE = typeof(first(thresholds))
    end

    taper_start = convert(ELTYPE, interface_taper_start)
    taper_width = convert(ELTYPE, support_taper_width)
    return ColorfieldSurfaceNormal(thresholds..., taper_start, taper_width)
end

@inline function cubic_smoothstep(value)
    value <= zero(value) && return zero(value)
    value >= one(value) && return one(value)
    return value^2 * (3 - 2value)
end

@inline function gradient_interface_activity(normal_norm, support_radius,
                                             surface_normal_method::ColorfieldSurfaceNormal)
    threshold = surface_normal_method.interface_threshold
    dimensionless_norm = support_radius * normal_norm
    if iszero(threshold)
        return iszero(dimensionless_norm) ? zero(dimensionless_norm) :
               one(dimensionless_norm)
    end

    lower_bound = surface_normal_method.interface_taper_start * threshold
    transition_coordinate = (dimensionless_norm - lower_bound) /
                            (threshold - lower_bound)
    return cubic_smoothstep(transition_coordinate)
end

@inline function support_interface_activity(support_moment,
                                            surface_normal_method::ColorfieldSurfaceNormal)
    threshold = surface_normal_method.ideal_density_threshold
    iszero(threshold) && return one(support_moment)

    transition_coordinate = (support_moment - threshold) /
                            surface_normal_method.support_taper_width
    return one(support_moment) - cubic_smoothstep(transition_coordinate)
end

@inline function surface_interface_activity(system, particle)
    return surface_interface_activity(surface_tension_model(system), system, particle)
end

@inline function surface_interface_activity(::SurfaceTensionMorris, system, particle)
    return @inbounds system.cache.interface_activity[particle]
end

@inline function surface_interface_activity(surface_tension, system, particle)
    normal = surface_normal(system, particle)
    return dot(normal, normal) > eps(eltype(normal)) ? one(eltype(normal)) :
           zero(eltype(normal))
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

function create_cache_surface_normal(::CorrectedCSFSurfaceNormal, ELTYPE, NDIMS, nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    correction_factor = Array{ELTYPE, 1}(undef, nparticles)
    ccsf_correction_matrix = Array{ELTYPE, 3}(undef, NDIMS, NDIMS, nparticles)
    ccsf_minimum_eigenvalue = Array{ELTYPE, 1}(undef, nparticles)
    ccsf_lambda_gradient = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    ccsf_color_gradient = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    ccsf_shepard_sum = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_normal, neighbor_count, correction_factor,
            ccsf_correction_matrix, ccsf_minimum_eigenvalue,
            ccsf_lambda_gradient, ccsf_color_gradient, ccsf_shepard_sum)
end

function create_cache_surface_normal(::CorrectedCSFSurfaceNormal{<:Real}, ELTYPE, NDIMS,
                                     nparticles)
    cache = create_cache_surface_normal(CorrectedCSFSurfaceNormal(), ELTYPE, NDIMS,
                                        nparticles)
    ccsf_boundary_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    ccsf_boundary_distance = Array{ELTYPE, 1}(undef, nparticles)
    return (; cache..., ccsf_boundary_normal, ccsf_boundary_distance)
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
        accumulate_surface_support_moment!(system, surface_tension_model(system), particle,
                                           m_b / density_neighbor, pos_diff, grad_kernel)

        cache.neighbor_count[particle] += 1
    end

    return system
end

@inline function accumulate_surface_support_moment!(system, surface_tension, particle,
                                                    volume, pos_diff, grad_kernel)
    return system
end

@inline function accumulate_surface_support_moment!(system, ::SurfaceTensionMorris,
                                                    particle, volume, pos_diff,
                                                    grad_kernel)
    value = -volume * dot(pos_diff, grad_kernel) / ndims(system)
    @inbounds system.cache.support_moment[particle] += value
    return system
end

@inline function accumulate_boundary_surface_support_moment!(system, surface_tension,
                                                             neighbor_system,
                                                             v_neighbor_system, particle,
                                                             neighbor, pos_diff, distance)
    return system
end

@inline function accumulate_boundary_surface_support_moment!(system,
                                                             ::SurfaceTensionMorris,
                                                             neighbor_system,
                                                             v_neighbor_system, particle,
                                                             neighbor, pos_diff, distance)
    m_b = hydrodynamic_mass(neighbor_system, neighbor)
    density_neighbor = current_density(v_neighbor_system, neighbor_system, neighbor)
    grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
    value = -m_b / density_neighbor * dot(pos_diff, grad_kernel) / ndims(system)
    @inbounds system.cache.support_moment[particle] += value
    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: This is the simplest form of normal approximation commonly used in SPH and comes
# with serious deficits in accuracy especially at corners, small neighborhoods and boundaries
function calc_boundary_normal!(system::AbstractFluidSystem, neighbor_system, u_system, v,
                               v_neighbor_system, u_neighbor_system, semi,
                               surface_normal_method)
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
        accumulate_boundary_surface_support_moment!(system, surface_tension_model(system),
                                                    neighbor_system, v_neighbor_system,
                                                    particle, neighbor, pos_diff, distance)

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
    return calc_boundary_normal!(system, neighbor_system, u_system, v, v_neighbor_system,
                                 u_neighbor_system, semi, surface_normal_method)
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
                                 surface_tension::SurfaceTensionMomentumMorris,
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

function remove_invalid_normals!(system::AbstractFluidSystem,
                                 ::SurfaceTensionMorris,
                                 surface_normal_method::ColorfieldSurfaceNormal)
    (; cache, smoothing_kernel) = system
    support_radius = compact_support(smoothing_kernel, initial_smoothing_length(system))

    for particle in each_integrated_particle(system)
        cache.delta_s[particle] = zero(eltype(system))
        cache.interface_activity[particle] = zero(eltype(system))

        particle_surface_normal = surface_normal(system, particle)
        norm2 = dot(particle_surface_normal, particle_surface_normal)
        if !(norm2 > eps(norm2))
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        normal_norm = sqrt(norm2)
        gradient_activity = gradient_interface_activity(normal_norm, support_radius,
                                                        surface_normal_method)
        support_moment = cache.support_moment[particle]
        support_activity = support_interface_activity(support_moment,
                                                      surface_normal_method)
        activity = gradient_activity * support_activity
        if !(activity > zero(activity))
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        cache.interface_activity[particle] = activity
        # A one-phase free surface samples one half of the kernel-smoothed interface.
        cache.delta_s[particle] = 2 * normal_norm * activity
        cache.surface_normal[1:ndims(system),
                             particle] = particle_surface_normal / normal_norm
    end

    return system
end

@inline reset_surface_interface_data!(system, surface_tension) = system

@inline function reset_surface_interface_data!(system, ::SurfaceTensionMorris)
    set_zero!(system.cache.support_moment)
    set_zero!(system.cache.interface_activity)
    set_zero!(system.cache.delta_s)
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
    reset_surface_interface_data!(system, surface_tension)

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

@inline function ccsf_store_matrix!(matrix_cache, system, particle, matrix)
    for column in 1:ndims(system), row in 1:ndims(system)
        @inbounds matrix_cache[row, column, particle] = matrix[row, column]
    end
    return matrix_cache
end

@inline function ccsf_minimum_eigenvalue(matrix)
    # `eigmin` falls back to an allocating dense eigensolver for static matrices.
    symmetric_matrix = (matrix + transpose(matrix)) / 2
    return minimum(eigvals(Symmetric(symmetric_matrix)))
end

function ccsf_halfspace_outside_fraction(::WendlandC2Kernel{3}, normalized_distance)
    distance = clamp(normalized_distance, zero(normalized_distance),
                     convert(typeof(normalized_distance), 2))
    distance >= 2 && return zero(distance)

    # Integrate the normalized three-dimensional kernel over a spherical cap.
    coefficients = (one(distance), zero(distance), -5one(distance) / 2,
                    5one(distance) / 2, -15one(distance) / 16,
                    one(distance) / 8)
    upper = convert(typeof(distance), 2)
    integral = zero(distance)
    for power in 0:5
        coefficient = coefficients[power + 1]
        integral += coefficient *
                    ((upper^(power + 3) - distance^(power + 3)) / (power + 3) -
                     distance * (upper^(power + 2) - distance^(power + 2)) /
                     (power + 2))
    end
    return 21integral / 8
end

@inline ccsf_eigenvalue_moment(moment, gamma,
                               ::CorrectedCSFSurfaceNormal{Nothing}) = moment

@inline function ccsf_eigenvalue_moment(moment, gamma,
                                        ::CorrectedCSFSurfaceNormal{<:Real})
    gamma > eps(gamma) || return moment
    return moment / gamma
end

@inline function ccsf_corrected_divergence(normal_difference, renormalization,
                                           kernel_direction)
    return dot(renormalization * normal_difference, kernel_direction)
end

@inline function ccsf_lambda_difference(lambda_i, lambda_j)
    return lambda_i >= oftype(lambda_i, 0.7) ? lambda_j - lambda_i : lambda_j
end

@inline ccsf_boundary_gamma(system, particle,
                            ::CorrectedCSFSurfaceNormal{Nothing}) = one(eltype(system))

@inline function ccsf_boundary_gamma(system, particle,
                                     ::CorrectedCSFSurfaceNormal{<:Real})
    distance = @inbounds system.cache.ccsf_boundary_distance[particle]
    isfinite(distance) || return one(eltype(system))
    normalized_distance = distance / smoothing_length(system, particle)
    outside_fraction = ccsf_halfspace_outside_fraction(system_smoothing_kernel(system),
                                                       normalized_distance)
    return one(outside_fraction) - outside_fraction
end

@inline ccsf_has_boundary_geometry(::CorrectedCSFSurfaceNormal{Nothing}) = false
@inline ccsf_has_boundary_geometry(::CorrectedCSFSurfaceNormal{<:Real}) = true

@inline function ccsf_boundary_cache(system)
    hasproperty(system, :boundary_model) || return nothing
    cache = system.boundary_model.cache
    haskey(cache, :surface_measure) || return nothing
    return cache
end

@inline function check_corrected_csf_boundary_configuration!(system,
                                                             surface_normal_method,
                                                             systems)
    return system
end

function check_corrected_csf_boundary_configuration!(system::AbstractFluidSystem,
                                                     ::CorrectedCSFSurfaceNormal{<:Real},
                                                     systems)
    system_smoothing_kernel(system) isa WendlandC2Kernel{3} ||
        throw(ArgumentError("C-CSF planar boundary geometry currently requires `WendlandC2Kernel{3}`"))

    support = compact_support(system_smoothing_kernel(system),
                              initial_smoothing_length(system))
    fluid_count = 0
    boundary_count = 0
    foreach_system(systems) do candidate
        if candidate isa AbstractFluidSystem
            fluid_count += 1
            return
        end

        valid_boundary = candidate isa WallBoundarySystem &&
                         candidate.boundary_model isa BoundaryModelDummyParticles
        valid_boundary ||
            throw(ArgumentError("C-CSF boundary geometry supports only dummy-particle wall boundaries"))
        isnothing(candidate.prescribed_motion) ||
            throw(ArgumentError("C-CSF boundary geometry currently requires stationary walls"))

        boundary_cache = candidate.boundary_model.cache
        haskey(boundary_cache, :surface_measure) ||
            throw(ArgumentError("C-CSF boundary geometry requires per-particle `surface_measure` values"))
        normals = candidate.initial_condition.normals
        isnothing(normals) &&
            throw(ArgumentError("C-CSF boundary geometry requires boundary normal offset vectors"))
        surface_measure = boundary_cache.surface_measure
        any(>(zero(eltype(surface_measure))), surface_measure) ||
            throw(ArgumentError("C-CSF boundary geometry requires a positive surface measure"))

        for particle in eachparticle(candidate)
            surface_measure[particle] > zero(eltype(surface_measure)) || continue
            offset = extract_svector(normals, candidate, particle)
            all(isfinite, offset) ||
                throw(ArgumentError("C-CSF boundary normal offsets must be finite"))
            offset_norm = norm(offset)
            0 < offset_norm < support ||
                throw(ArgumentError("C-CSF boundary normal offsets must place the physical face inside the kernel support"))
        end
        boundary_count += 1
    end

    fluid_count == 1 ||
        throw(ArgumentError("C-CSF boundary geometry requires exactly one fluid system"))
    boundary_count > 0 ||
        throw(ArgumentError("C-CSF boundary geometry requires at least one boundary"))
    return system
end

@inline function ccsf_face_geometry(system, neighbor_system, particle, neighbor,
                                    pos_diff)
    boundary_cache = ccsf_boundary_cache(neighbor_system)
    isnothing(boundary_cache) && return nothing
    surface_measure = @inbounds boundary_cache.surface_measure[neighbor]
    surface_measure > zero(surface_measure) || return nothing
    normals = neighbor_system.initial_condition.normals
    isnothing(normals) && return nothing
    offset = extract_svector(normals, neighbor_system, neighbor)
    offset_norm = norm(offset)
    offset_norm > eps(offset_norm) || return nothing
    # The offset points from the physical face into the dummy-particle layer.
    wall_normal = offset / offset_norm
    face_diff = -pos_diff - offset # Physical face position minus fluid position.
    kernel_distance = norm(face_diff)
    kernel_distance < compact_support(system, neighbor_system) || return nothing
    boundary_distance = abs(dot(face_diff, wall_normal))
    return (; surface_measure, wall_normal, face_diff, kernel_distance,
            boundary_distance)
end

@inline function accumulate_ccsf_boundary_moments!(system, neighbor_system, v, u,
                                                   v_neighbor, u_neighbor, semi, method)
    return system
end

function accumulate_ccsf_boundary_moments!(system::AbstractFluidSystem,
                                           neighbor_system::AbstractBoundarySystem,
                                           v, u, v_neighbor, u_neighbor, semi,
                                           ::CorrectedCSFSurfaceNormal{<:Real})
    cache = system.cache
    system_coordinates = current_coordinates(u, system)
    neighbor_coordinates = current_coordinates(u_neighbor, neighbor_system)
    foreach_point_neighbor(system, neighbor_system, system_coordinates,
                           neighbor_coordinates, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        geometry = ccsf_face_geometry(system, neighbor_system, particle, neighbor,
                                      pos_diff)
        isnothing(geometry) && return
        (; surface_measure, wall_normal, face_diff, kernel_distance,
         boundary_distance) = geometry
        weight = surface_measure * smoothing_kernel(system, kernel_distance, particle)
        moment = weight * face_diff * permutedims(wall_normal)
        for column in 1:ndims(system), row in 1:ndims(system)
            @inbounds cache.ccsf_correction_matrix[row, column,
                                                   particle] += moment[row, column]
        end
        for dimension in 1:ndims(system)
            @inbounds cache.ccsf_color_gradient[dimension,
                                                particle] += weight * wall_normal[dimension]
        end
        if boundary_distance < @inbounds(cache.ccsf_boundary_distance[particle])
            @inbounds cache.ccsf_boundary_distance[particle] = boundary_distance
            for dimension in 1:ndims(system)
                @inbounds cache.ccsf_boundary_normal[dimension,
                                                     particle] = -wall_normal[dimension]
            end
        end
    end
    return system
end

@inline function accumulate_ccsf_boundary_lambda_gradient!(system, neighbor_system, u,
                                                           u_neighbor, semi, method)
    return system
end

function accumulate_ccsf_boundary_lambda_gradient!(system::AbstractFluidSystem,
                                                   neighbor_system::AbstractBoundarySystem,
                                                   u, u_neighbor, semi,
                                                   ::CorrectedCSFSurfaceNormal{<:Real})
    cache = system.cache
    coordinates = current_coordinates(u, system)
    neighbor_coordinates = current_coordinates(u_neighbor, neighbor_system)
    foreach_point_neighbor(system, neighbor_system, coordinates, neighbor_coordinates, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        geometry = ccsf_face_geometry(system, neighbor_system, particle, neighbor,
                                      pos_diff)
        isnothing(geometry) && return
        (; surface_measure, wall_normal, kernel_distance) = geometry
        weight = surface_measure * smoothing_kernel(system, kernel_distance, particle)
        lambda_i = @inbounds cache.ccsf_minimum_eigenvalue[particle]
        renormalization = extract_smatrix(cache.ccsf_correction_matrix, system, particle)
        coefficient = ccsf_lambda_difference(lambda_i, one(lambda_i))
        contribution = coefficient * weight * renormalization * wall_normal
        for dimension in 1:ndims(system)
            @inbounds cache.ccsf_lambda_gradient[dimension,
                                                 particle] += contribution[dimension]
        end
    end
    return system
end

function apply_ccsf_contact_normal!(system, method::CorrectedCSFSurfaceNormal{<:Real})
    cache = system.cache
    support = compact_support(system_smoothing_kernel(system),
                              initial_smoothing_length(system))
    target_angle = deg2rad(convert(eltype(system), method.contact_angle))
    for particle in each_integrated_particle(system)
        distance = @inbounds cache.ccsf_boundary_distance[particle]
        distance < support || continue
        normal = surface_normal(system, particle)
        dot(normal, normal) > eps(eltype(normal)) || continue
        boundary_normal = extract_svector(cache.ccsf_boundary_normal, system, particle)
        tangent = normal - dot(normal, boundary_normal) * boundary_normal
        tangent_norm = norm(tangent)
        tangent_norm > eps(tangent_norm) || continue
        tangent /= tangent_norm
        current_angle = acos(clamp(dot(normal, boundary_normal), -one(eltype(system)),
                                   one(eltype(system))))
        corrected_angle = target_angle +
                          (current_angle - target_angle) * (distance / support)^2
        corrected = cos(corrected_angle) * boundary_normal +
                    sin(corrected_angle) * tangent
        for dimension in 1:ndims(system)
            @inbounds cache.surface_normal[dimension, particle] = corrected[dimension]
        end
    end
    return system
end

@inline apply_ccsf_contact_normal!(system, method) = system

function compute_surface_normal!(system::AbstractFluidSystem,
                                 method::CorrectedCSFSurfaceNormal,
                                 v, u, v_ode, u_ode, semi, t)
    system.surface_tension isa SurfaceTensionMorris ||
        throw(ArgumentError("`CorrectedCSFSurfaceNormal` requires `SurfaceTensionMorris`"))
    cache = system.cache
    matrix_cache = cache.ccsf_correction_matrix
    lambda = cache.ccsf_minimum_eigenvalue
    lambda_gradient = cache.ccsf_lambda_gradient
    color_gradient = cache.ccsf_color_gradient
    shepard_sum = cache.ccsf_shepard_sum
    coordinates = current_coordinates(u, system)

    set_zero!(cache.surface_normal)
    set_zero!(cache.neighbor_count)
    set_zero!(matrix_cache)
    set_zero!(lambda)
    set_zero!(lambda_gradient)
    set_zero!(color_gradient)
    set_zero!(shepard_sum)
    set_zero!(cache.support_moment)
    if ccsf_has_boundary_geometry(method)
        set_zero!(cache.ccsf_boundary_normal)
        fill!(cache.ccsf_boundary_distance,
              typemax(eltype(cache.ccsf_boundary_distance)))
    end

    @trixi_timeit timer() "compute C-CSF moments" begin
        foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            m_b = hydrodynamic_mass(system, neighbor)
            rho_b = current_density(v, system, neighbor)
            volume_b = m_b / rho_b
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            kernel = smoothing_kernel(system, distance, particle)

            moment = -volume_b * grad_kernel * permutedims(pos_diff)
            for column in 1:ndims(system), row in 1:ndims(system)
                @inbounds matrix_cache[row, column, particle] += moment[row, column]
            end
            for dimension in 1:ndims(system)
                @inbounds color_gradient[dimension,
                                         particle] += volume_b * grad_kernel[dimension]
            end
            @inbounds shepard_sum[particle] += volume_b * kernel
            @inbounds cache.neighbor_count[particle] += 1
        end
    end

    if ccsf_has_boundary_geometry(method)
        foreach_system(semi) do neighbor_system
            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)
            accumulate_ccsf_boundary_moments!(system, neighbor_system, v, u,
                                              v_neighbor, u_neighbor, semi, method)
        end
    end

    @threaded semi for particle in each_integrated_particle(system)
        inverse_renormalization = extract_smatrix(matrix_cache, system, particle)
        gamma = ccsf_boundary_gamma(system, particle, method)
        eigenvalue_moment = ccsf_eigenvalue_moment(inverse_renormalization, gamma,
                                                   method)
        @inbounds lambda[particle] = ccsf_minimum_eigenvalue(eigenvalue_moment)
        renormalization = abs(det(inverse_renormalization)) < 1.0f-9 ?
                          one(inverse_renormalization) : inv(inverse_renormalization)
        ccsf_store_matrix!(matrix_cache, system, particle, renormalization)
    end

    if ccsf_has_boundary_geometry(method)
        foreach_system(semi) do neighbor_system
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)
            accumulate_ccsf_boundary_lambda_gradient!(system, neighbor_system, u,
                                                      u_neighbor, semi, method)
        end
    end

    @trixi_timeit timer() "compute C-CSF normal" begin
        foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            rho_b = current_density(v, system, neighbor)
            volume_b = hydrodynamic_mass(system, neighbor) / rho_b
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            renormalization = extract_smatrix(matrix_cache, system, particle)
            lambda_a = @inbounds lambda[particle]
            lambda_b = @inbounds lambda[neighbor]
            coefficient = ccsf_lambda_difference(lambda_a, lambda_b)
            contribution = coefficient * volume_b * renormalization * grad_kernel
            for dimension in 1:ndims(system)
                @inbounds lambda_gradient[dimension, particle] += contribution[dimension]
            end
        end
    end

    set_zero!(cache.interface_activity)
    set_zero!(cache.delta_s)
    for particle in each_integrated_particle(system)
        gradient = extract_svector(lambda_gradient, system, particle)
        gradient_norm = norm(gradient)
        lambda_i = @inbounds lambda[particle]
        threshold = oftype(lambda_i, 0.1) * lambda_i /
                    smoothing_length(system, particle)
        if gradient_norm > threshold
            normal = -gradient / gradient_norm
            for dimension in 1:ndims(system)
                @inbounds cache.surface_normal[dimension, particle] = normal[dimension]
            end
            @inbounds cache.interface_activity[particle] = one(lambda_i)
        end

        raw_gradient = extract_svector(color_gradient, system, particle)
        shepard = @inbounds shepard_sum[particle]
        gamma = ccsf_boundary_gamma(system, particle, method)
        correction = shepard > eps(shepard) ?
                     max(one(shepard), gamma / (2shepard)) : one(shepard)
        @inbounds cache.delta_s[particle] = 2correction / gamma * norm(raw_gradient)
        @inbounds cache.support_moment[particle] = lambda_i
    end
    apply_ccsf_contact_normal!(system, method)

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

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
        n_a = surface_normal(system, particle)
        n_b = surface_normal(neighbor_system, neighbor)
        v_b = m_b / rho_b
        activity_a = surface_interface_activity(system, particle)
        activity_b = surface_interface_activity(neighbor_system, neighbor)

        if activity_a > zero(activity_a) && activity_b > zero(activity_b)
            w = smoothing_kernel(system, distance, particle)
            grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
            weighted_volume = v_b * activity_b

            for i in 1:ndims(system)
                curvature[particle] += weighted_volume * (n_b[i] - n_a[i]) *
                                       grad_kernel[i]
            end
            correction_factor[particle] += weighted_volume * w
        end
    end

    return system
end

function calc_curvature!(system::AbstractFluidSystem,
                         neighbor_system::AbstractFluidSystem,
                         u_system, v, v_neighbor_system, u_neighbor_system, semi,
                         ::CorrectedCSFSurfaceNormal,
                         ::CorrectedCSFSurfaceNormal)
    system === neighbor_system ||
        throw(ArgumentError("`CorrectedCSFSurfaceNormal` currently supports one fluid system"))
    cache = system.cache
    coordinates = current_coordinates(u_system, system)
    cosine_threshold = -inv(convert(eltype(system), ndims(system)))

    foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        n_a = surface_normal(system, particle)
        n_b = surface_normal(system, neighbor)
        dot(n_a, n_a) > eps(eltype(n_a)) || return
        dot(n_b, n_b) > eps(eltype(n_b)) || return
        dot(n_a, n_b) >= cosine_threshold || return

        rho_b = current_density(v, system, neighbor)
        volume_b = hydrodynamic_mass(system, neighbor) / rho_b
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        renormalization = extract_smatrix(cache.ccsf_correction_matrix, system, particle)
        @inbounds cache.curvature[particle] += volume_b *
                                               ccsf_corrected_divergence(n_b - n_a,
                                                                         renormalization,
                                                                         grad_kernel)
    end
    return system
end

function calc_curvature!(system::AbstractFluidSystem,
                         neighbor_system::AbstractBoundarySystem,
                         u_system, v, v_neighbor_system, u_neighbor_system, semi,
                         method::CorrectedCSFSurfaceNormal{<:Real},
                         neighbor_surface_normal_method)
    cache = system.cache
    coordinates = current_coordinates(u_system, system)
    neighbor_coordinates = current_coordinates(u_neighbor_system, neighbor_system)
    target_angle = deg2rad(convert(eltype(system), method.contact_angle))
    cosine_threshold = -inv(convert(eltype(system), ndims(system)))
    foreach_point_neighbor(system, neighbor_system, coordinates, neighbor_coordinates, semi;
                           points=each_integrated_particle(system)) do particle, neighbor,
                                                                       pos_diff, distance
        geometry = ccsf_face_geometry(system, neighbor_system, particle, neighbor,
                                      pos_diff)
        isnothing(geometry) && return
        (; surface_measure, wall_normal, kernel_distance) = geometry
        normal = surface_normal(system, particle)
        dot(normal, normal) > eps(eltype(normal)) || return
        boundary_normal = -wall_normal
        tangent = normal - dot(normal, boundary_normal) * boundary_normal
        tangent_norm = norm(tangent)
        tangent_norm > eps(tangent_norm) || return
        contact_normal = cos(target_angle) * boundary_normal +
                         sin(target_angle) * tangent / tangent_norm
        dot(normal, contact_normal) >= cosine_threshold || return
        renormalization = extract_smatrix(cache.ccsf_correction_matrix, system, particle)
        weight = surface_measure * smoothing_kernel(system, kernel_distance, particle)
        @inbounds cache.curvature[particle] += weight *
                                               ccsf_corrected_divergence(contact_normal -
                                                                         normal,
                                                                         renormalization,
                                                                         wall_normal)
    end
    return system
end

@inline function normalized_surface_curvature(curvature_numerator, denominator)
    denominator > sqrt(eps(typeof(denominator))) || return zero(curvature_numerator)
    return curvature_numerator / denominator
end

@inline function finalize_surface_curvature(curvature_numerator, denominator,
                                            surface_normal_method)
    return normalized_surface_curvature(curvature_numerator, denominator)
end

@inline function finalize_surface_curvature(curvature_numerator, denominator,
                                            ::CorrectedCSFSurfaceNormal)
    return curvature_numerator
end

function compute_curvature!(system, surface_tension, v, u, v_ode, u_ode, semi, t)
    return system
end

function compute_curvature!(system::AbstractFluidSystem,
                            surface_tension::SurfaceTensionMorris,
                            v, u, v_ode, u_ode, semi, t)
    (; cache, surface_tension) = system
    normal_method = surface_normal_method(system)

    # Reset once so contributions from multiple fluid systems accumulate consistently.
    set_zero!(cache.curvature)
    set_zero!(cache.correction_factor)

    @trixi_timeit timer() "compute surface curvature" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_curvature!(system, neighbor_system, u, v, v_neighbor_system,
                        u_neighbor_system, semi, normal_method,
                        surface_normal_method(neighbor_system))
    end

    for particle in each_integrated_particle(system)
        denominator = cache.correction_factor[particle]
        cache.curvature[particle] = finalize_surface_curvature(cache.curvature[particle],
                                                               denominator, normal_method)
    end
    return system
end
