abstract type AbstractContactAngleModel end

function validate_contact_angle(contact_angle)
    if !(contact_angle isa Real) || !isfinite(contact_angle) ||
       !(0 <= contact_angle <= 180)
        throw(ArgumentError("`contact_angle` must be a finite real number in [0, 180] degrees"))
    end

    return contact_angle
end

@doc raw"""
    WettedAreaContactAngle(contact_angle)

Apply Young's wall energy through a corrected wetted-area quadrature. The model is an explicit
opt-in for [`ColorfieldSurfaceNormal`](@ref); constructing `ColorfieldSurfaceNormal()` without a
contact model remains unchanged.

The validated production configuration is currently restricted to three-dimensional
`ContinuityDensity` fluids using `WendlandC2Kernel{3}` with `h/dx=1.4`. Contact boundaries must be
dummy-particle wall or rigid-body systems built with per-particle `surface_measure` values and
`InitialCondition.normals`. Each boundary system represents one connected disk-like wetted patch.
Angles at exactly 0 and 180 degrees are not supported by the canonical edge correction; at
90 degrees the wall energy and force are exactly zero.
"""
struct WettedAreaContactAngle{ELTYPE <: Real} <: AbstractContactAngleModel
    contact_angle::ELTYPE

    function WettedAreaContactAngle(contact_angle)
        angle = validate_contact_angle(contact_angle)
        0 < angle < 180 ||
            throw(ArgumentError("`WettedAreaContactAngle` requires `contact_angle` in (0, 180) degrees"))
        new{typeof(angle)}(angle)
    end
end

@inline convert_contact_model(::Nothing, ELTYPE) = nothing

@inline function convert_contact_model(contact_model::WettedAreaContactAngle, ELTYPE)
    return WettedAreaContactAngle(convert(ELTYPE, contact_model.contact_angle))
end

function convert_contact_model(contact_model, ELTYPE)
    throw(ArgumentError("`contact_model` must be `nothing` or `WettedAreaContactAngle`"))
end

@doc raw"""
    ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                              ideal_density_threshold=0.0, interface_taper_start=0.8,
                              support_taper_width=0.025, contact_model=nothing,
                              normal_smoothing=false)

Color field based computation of the interface normals.

# Keywords
- `boundary_contact_threshold=0.1`: If this threshold is reached the fluid is assumed to be in contact with the boundary.
- `interface_threshold=0.01`:       Threshold for normals to be removed as being invalid.
- `ideal_density_threshold=0.0`:    Assume particles are inside if their continuous kernel-support
                                    moment is above this fraction of complete support. Zero disables
                                    this filter.
- `interface_taper_start=0.8`:      Start the smooth interface activation at this fraction of
                                    `interface_threshold`.
- `support_taper_width=0.025`:      Width of the smooth transition above
                                    `ideal_density_threshold`.
- `contact_model=nothing`:          Optional contact-angle model. The validated explicit choice is
                                    [`WettedAreaContactAngle`](@ref).
- `normal_smoothing=false`:         Apply one activity-weighted Shepard smoothing pass to unit
                                    normals before curvature or CSS stress evaluation.
"""
struct ColorfieldSurfaceNormal{ELTYPE, CONTACT_MODEL}
    boundary_contact_threshold::ELTYPE
    interface_threshold::ELTYPE
    ideal_density_threshold::ELTYPE
    interface_taper_start::ELTYPE
    support_taper_width::ELTYPE
    contact_model::CONTACT_MODEL
    normal_smoothing::Bool
end

@doc raw"""
    CorrectedCSFSurfaceNormal(; contact_angle=nothing)

Interface geometry for the corrected continuous-surface-force (C-CSF) method of Vergnaud
et al. (2022). The outward unit normal is computed from the renormalized gradient of the
smallest eigenvalue of the first-order kernel moment. Curvature uses the corresponding
renormalized divergence with the published thin-jet angular filter, and the surface delta
uses the published Shepard correction.

With `contact_angle=nothing`, this explicit opt-in implements the single-fluid free-surface core
(equations 15--25) with [`SurfaceTensionMorris`](@ref). A finite `contact_angle` enables the
planar boundary-integral geometry terms and contact-normal correction from equations 41--50.
Contact boundaries require a three-dimensional Wendland C2 kernel, explicit surface measures,
and normal offset vectors. These terms correct interface geometry only; hydrodynamic wall
interactions remain those of the configured boundary model.
"""
struct CorrectedCSFSurfaceNormal{CONTACT_ANGLE}
    contact_angle::CONTACT_ANGLE
end

function CorrectedCSFSurfaceNormal(; contact_angle=nothing)
    isnothing(contact_angle) && return CorrectedCSFSurfaceNormal(nothing)
    return CorrectedCSFSurfaceNormal(float(validate_contact_angle(contact_angle)))
end

@inline function supports_free_surface_shifting(::ColorfieldSurfaceNormal,
                                                ::Union{SurfaceTensionMorris,
                                                        SurfaceTensionMomentumMorris})
    return true
end

@inline function supports_free_surface_shifting(::CorrectedCSFSurfaceNormal,
                                                ::SurfaceTensionMorris)
    return true
end

@inline validate_corrected_csf(surface_normal_method, surface_tension) = nothing

function validate_corrected_csf(::CorrectedCSFSurfaceNormal, surface_tension)
    surface_tension isa SurfaceTensionMorris ||
        throw(ArgumentError("`CorrectedCSFSurfaceNormal` requires `SurfaceTensionMorris`"))
    return nothing
end

# Interface-aware TIC needs interface activity from one of these surface-normal methods;
# anything else cannot gate the tensile correction.
@inline supports_interface_aware_tic(surface_normal_method, surface_tension) = false

@inline function supports_interface_aware_tic(::ColorfieldSurfaceNormal,
                                              ::Union{SurfaceTensionMorris,
                                                      SurfaceTensionMomentumMorris})
    return true
end

@inline function supports_interface_aware_tic(::CorrectedCSFSurfaceNormal,
                                              ::SurfaceTensionMorris)
    return true
end

function ColorfieldSurfaceNormal(boundary_contact_threshold, interface_threshold,
                                 ideal_density_threshold)
    return ColorfieldSurfaceNormal(; boundary_contact_threshold, interface_threshold,
                                   ideal_density_threshold)
end

function ColorfieldSurfaceNormal(; boundary_contact_threshold=0.1, interface_threshold=0.01,
                                 ideal_density_threshold=0.0, interface_taper_start=0.8,
                                 support_taper_width=0.025, contact_model=nothing,
                                 normal_smoothing=false)
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
    normal_smoothing isa Bool ||
        throw(ArgumentError("`normal_smoothing` must be `true` or `false`"))

    thresholds = promote(boundary_contact_threshold, interface_threshold,
                         ideal_density_threshold)
    ELTYPE = typeof(first(thresholds))
    if ELTYPE <: Integer
        thresholds = float.(thresholds)
        ELTYPE = typeof(first(thresholds))
    end
    taper_start = convert(ELTYPE, interface_taper_start)
    taper_width = convert(ELTYPE, support_taper_width)

    contact_model_ = convert_contact_model(contact_model, ELTYPE)
    return ColorfieldSurfaceNormal(thresholds..., taper_start, taper_width, contact_model_,
                                   normal_smoothing)
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

@inline function surface_support_moment(system, ::SurfaceTensionMorris, particle)
    return @inbounds system.cache.support_moment[particle]
end

@inline function surface_support_moment(system, ::SurfaceTensionMomentumMorris, particle)
    return @inbounds system.cache.divergence_correction[particle]
end

@inline function surface_interface_activity(system, particle)
    return surface_interface_activity(surface_tension_model(system), system, particle)
end

@inline function surface_interface_activity(::Union{SurfaceTensionMorris,
                                                    SurfaceTensionMomentumMorris},
                                            system, particle)
    return @inbounds system.cache.interface_activity[particle]
end

@inline function surface_interface_activity(surface_tension, system, particle)
    normal = surface_normal(system, particle)
    return dot(normal, normal) > eps(eltype(normal)) ? one(eltype(normal)) :
           zero(eltype(normal))
end

@inline function default_surface_normal_method(surface_tension, surface_normal_method)
    if isnothing(surface_normal_method) && requires_surface_normal(surface_tension)
        if surface_tension isa SurfaceTensionAkinci
            # Akinci et al. (2013), Equation 2, sums fluid neighbors only. Wall adhesion is
            # a separate pair force and must not alter the surface-area normal implicitly.
            return ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf)
        end
        return ColorfieldSurfaceNormal()
    end

    return surface_normal_method
end

function create_cache_surface_normal(surface_normal_method, ELTYPE, NDIMS, nparticles)
    return (;)
end

function create_cache_surface_normal(method::ColorfieldSurfaceNormal, ELTYPE, NDIMS,
                                     nparticles)
    surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    neighbor_count = Array{ELTYPE, 1}(undef, nparticles)
    colorfield = Array{ELTYPE, 1}(undef, nparticles)
    correction_factor = Array{ELTYPE, 1}(undef, nparticles)
    cache = (; surface_normal, neighbor_count, colorfield, correction_factor)
    method.normal_smoothing || return cache
    smoothed_surface_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    normal_smoothing_weight = Array{ELTYPE, 1}(undef, nparticles)
    return (; cache..., smoothed_surface_normal, normal_smoothing_weight)
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
    ccsf_boundary_normal = Array{ELTYPE, 2}(undef, NDIMS, nparticles)
    ccsf_boundary_distance = Array{ELTYPE, 1}(undef, nparticles)
    return (; surface_normal, neighbor_count, correction_factor,
            ccsf_correction_matrix, ccsf_minimum_eigenvalue,
            ccsf_lambda_gradient, ccsf_color_gradient, ccsf_shepard_sum,
            ccsf_boundary_normal, ccsf_boundary_distance)
end

function create_cache_surface_normal(method::ColorfieldSurfaceNormal{<:Any,
                                                                     <:WettedAreaContactAngle},
                                     ELTYPE, NDIMS, nparticles)
    cache = create_cache_surface_normal(ColorfieldSurfaceNormal(;
                                                                normal_smoothing=method.normal_smoothing),
                                        ELTYPE, NDIMS, nparticles)
    wetted_area_density_conjugate = zeros(ELTYPE, nparticles)
    wetted_area_energy = Ref(zero(ELTYPE))
    wetted_area_raw_area = Ref(zero(ELTYPE))
    wetted_area = Ref(zero(ELTYPE))
    wetted_area_normalized_edge_shift = Ref(convert(ELTYPE, NaN))
    wetted_area_evaluations = Ref(0)
    return (; cache..., wetted_area_density_conjugate,
            wetted_area_energy, wetted_area_raw_area, wetted_area,
            wetted_area_normalized_edge_shift, wetted_area_evaluations)
end

@inline wetted_area_smoothstep_derivative(value) = 6value * (1 - value)

@inline function wetted_area_contact_cosine(contact_model::WettedAreaContactAngle)
    contact_model.contact_angle == 90 && return zero(contact_model.contact_angle)
    return cosd(contact_model.contact_angle)
end

@inline function wetted_area_coefficient(surface_tension,
                                         contact_model::WettedAreaContactAngle)
    contact_cosine = wetted_area_contact_cosine(contact_model)
    iszero(contact_cosine) && return zero(surface_tension.surface_tension_coefficient)
    return surface_tension.surface_tension_coefficient * contact_cosine
end

function wetted_area_halfspace_reference(::WendlandC2Kernel{3}, normalized_distance)
    distance = clamp(normalized_distance, zero(normalized_distance),
                     convert(typeof(normalized_distance), 2))
    distance >= 2 && return zero(distance)

    # Integrate the normalized three-dimensional kernel over a spherical cap:
    # 2pi * integral_d^2 W(r) * r * (r - d) dr.
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

function canonical_wetted_area_edge_shift(smoothing_kernel, cells_per_h, contact_angle;
                                          quadrature_cells_per_h=64)
    contact_sine = sind(contact_angle)
    abs(contact_sine) > sqrt(eps(typeof(contact_sine))) ||
        return zero(contact_sine)
    contact_cotangent = cosd(contact_angle) / contact_sine
    lattice_spacing = inv(convert(typeof(cells_per_h), quadrature_cells_per_h))
    support = compact_support(smoothing_kernel, one(cells_per_h))
    search_radius = ceil(Int, support / lattice_spacing)
    boundary_distance = inv(2cells_per_h)
    thresholds = typeof(cells_per_h)[]
    weights = typeof(cells_per_h)[]

    for z_offset in (-search_radius):search_radius,
        x_offset in (-search_radius):search_radius
        planar_distance2 = lattice_spacing^2 * (x_offset^2 + z_offset^2)
        planar_distance2 < support^2 || continue
        reduced_kernel = zero(cells_per_h)
        for tangent_offset in (-search_radius):search_radius
            distance = lattice_spacing *
                       sqrt(x_offset^2 + tangent_offset^2 + z_offset^2)
            distance < support || continue
            reduced_kernel += lattice_spacing * kernel(smoothing_kernel, distance,
                                     one(cells_per_h))
        end
        source_z = -boundary_distance - z_offset * lattice_spacing
        source_z > 0 || continue
        push!(thresholds, x_offset * lattice_spacing + contact_cotangent * source_z)
        push!(weights, lattice_spacing^2 * reduced_kernel)
    end

    order = sortperm(thresholds)
    thresholds = thresholds[order]
    weights = weights[order]
    reference = sum(weights)
    reference > eps(reference) || return zero(reference)
    breaks = sort!(unique!([thresholds; zero(cells_per_h)]))
    cumulative = zero(reference)
    event = 1
    shift = zero(reference)
    for interval in 1:(length(breaks) - 1)
        left = breaks[interval]
        right = breaks[interval + 1]
        while event <= length(thresholds) && thresholds[event] <= left
            cumulative += weights[event]
            event += 1
        end
        fraction = clamp(cumulative / reference, 0, 1)
        step = (left + right) / 2 > 0 ? one(reference) : zero(reference)
        shift += (right - left) * (cubic_smoothstep(fraction) - step)
    end
    return shift
end

@inline function wetted_area_boundary_cache(system)
    hasproperty(system, :boundary_model) || return nothing
    model = system.boundary_model
    hasproperty(model, :cache) || return nothing
    haskey(model.cache, :wetted_area_surface_measure) || return nothing
    return model.cache
end

@inline wetted_area_supported_fluid(system) = false

@inline function check_corrected_csf_boundary_configuration!(system,
                                                             surface_normal_method,
                                                             systems)
    return system
end

function check_corrected_csf_boundary_configuration!(system::AbstractFluidSystem,
                                                     method::CorrectedCSFSurfaceNormal{<:Real},
                                                     systems)
    system_smoothing_kernel(system) isa WendlandC2Kernel{3} ||
        throw(ArgumentError("C-CSF planar BIM geometry currently requires `WendlandC2Kernel{3}`"))
    fluid_count = 0
    boundary_count = 0
    foreach_system(systems) do candidate
        if candidate isa AbstractFluidSystem
            fluid_count += 1
            return
        end
        valid_boundary = candidate isa AbstractBoundarySystem &&
                         hasproperty(candidate, :boundary_model) &&
                         candidate.boundary_model isa BoundaryModelDummyParticles
        valid_boundary ||
            throw(ArgumentError("C-CSF boundary geometry supports only dummy-particle boundaries"))
        cache = candidate.boundary_model.cache
        haskey(cache, :wetted_area_surface_measure) ||
            throw(ArgumentError("C-CSF boundary geometry requires per-particle `surface_measure` values"))
        isnothing(candidate.initial_condition.normals) &&
            throw(ArgumentError("C-CSF boundary geometry requires boundary normal offset vectors"))
        any(>(zero(eltype(cache.wetted_area_surface_measure))),
            cache.wetted_area_surface_measure) ||
            throw(ArgumentError("C-CSF boundary geometry requires a positive surface measure"))
        boundary_count += 1
    end
    fluid_count == 1 ||
        throw(ArgumentError("C-CSF boundary geometry requires exactly one fluid system"))
    boundary_count > 0 ||
        throw(ArgumentError("C-CSF boundary geometry requires at least one boundary"))
    return system
end

@inline function check_wetted_area_configuration!(system, surface_normal_method, systems)
    return system
end

function check_wetted_area_configuration!(system::AbstractFluidSystem,
                                          surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                         <:WettedAreaContactAngle},
                                          systems)
    ndims(system) == 3 ||
        throw(ArgumentError("`WettedAreaContactAngle` currently supports only 3D fluids"))
    wetted_area_supported_fluid(system) ||
        throw(ArgumentError("`WettedAreaContactAngle` currently supports only WCSPH and EDAC fluids"))
    density_calculator(system) isa ContinuityDensity ||
        throw(ArgumentError("`WettedAreaContactAngle` requires `ContinuityDensity`"))
    system.smoothing_kernel isa WendlandC2Kernel{3} ||
        throw(ArgumentError("`WettedAreaContactAngle` requires `WendlandC2Kernel{3}`"))
    system.surface_tension isa SurfaceTensionMomentumMorris ||
        throw(ArgumentError("`WettedAreaContactAngle` requires `SurfaceTensionMomentumMorris`"))
    isfinite(surface_normal_method.boundary_contact_threshold) ||
        throw(ArgumentError("`WettedAreaContactAngle` requires a finite `boundary_contact_threshold`"))
    system.cache.color == 1 ||
        throw(ArgumentError("`WettedAreaContactAngle` requires the fluid `color_value` to be 1"))

    particle_spacing = system.cache.reference_particle_spacing
    cells_per_h = initial_smoothing_length(system) / particle_spacing
    isapprox(cells_per_h, convert(typeof(cells_per_h), 1.4);
             rtol=100eps(typeof(cells_per_h)), atol=zero(cells_per_h)) ||
        throw(ArgumentError("`WettedAreaContactAngle` requires `smoothing_length / reference_particle_spacing == 1.4`"))

    fluid_count = 0
    boundary_count = 0
    foreach_system(systems) do candidate
        if candidate isa AbstractFluidSystem
            fluid_count += 1
            return
        end

        valid_boundary = (candidate isa WallBoundarySystem ||
                          candidate isa RigidBodySystem) &&
                         hasproperty(candidate, :boundary_model) &&
                         candidate.boundary_model isa BoundaryModelDummyParticles
        valid_boundary ||
            throw(ArgumentError("`WettedAreaContactAngle` supports only dummy-particle wall and rigid-body neighbors"))
        candidate.cache.color == 0 ||
            throw(ArgumentError("`WettedAreaContactAngle` requires contact-boundary `color_value` to be 0"))
        boundary_count += 1
        initialize_wetted_area_boundary!(system, candidate)
    end
    fluid_count == 1 ||
        throw(ArgumentError("`WettedAreaContactAngle` requires exactly one fluid system"))
    boundary_count > 0 ||
        throw(ArgumentError("`WettedAreaContactAngle` requires at least one contact boundary"))

    cache = system.cache
    if isnan(cache.wetted_area_normalized_edge_shift[])
        cache.wetted_area_normalized_edge_shift[] = canonical_wetted_area_edge_shift(system.smoothing_kernel,
                                                                                     cells_per_h,
                                                                                     surface_normal_method.contact_model.contact_angle)
    end
    return system
end

function initialize_wetted_area_boundary!(fluid_system, boundary_system)
    cache = wetted_area_boundary_cache(boundary_system)
    isnothing(cache) &&
        throw(ArgumentError("contact boundaries require explicit per-particle `surface_measure` values"))
    haskey(cache, :initial_colorfield) ||
        throw(ArgumentError("contact boundaries require a positive `reference_particle_spacing`"))

    normals = boundary_system.initial_condition.normals
    isnothing(normals) &&
        throw(ArgumentError("contact boundaries require `InitialCondition.normals`"))
    surface_measure = cache.wetted_area_surface_measure
    contact_model = fluid_system.surface_normal_method.contact_model
    cache.wetted_area_active[] = !iszero(wetted_area_coefficient(fluid_system.surface_tension,
                                                                 contact_model))
    active_particles = findall(>(zero(eltype(surface_measure))), surface_measure)
    isempty(active_particles) &&
        throw(ArgumentError("each contact boundary requires at least one positive `surface_measure`"))
    validate_wetted_area_patch_connectivity(boundary_system.initial_condition,
                                            surface_measure, active_particles)

    particle_spacing = fluid_system.cache.reference_particle_spacing
    particle_volume = fluid_system.initial_condition.mass[first(eachparticle(fluid_system))] /
                      fluid_system.initial_condition.density[first(eachparticle(fluid_system))]
    volume_scale = particle_volume / particle_spacing^3
    for particle in eachparticle(fluid_system)
        volume = fluid_system.initial_condition.mass[particle] /
                 fluid_system.initial_condition.density[particle]
        isapprox(volume / particle_spacing^3, volume_scale;
                 rtol=100eps(typeof(volume_scale)), atol=zero(volume_scale)) ||
            throw(ArgumentError("`WettedAreaContactAngle` requires uniform reference fluid particle volumes"))
    end

    smoothing_length = initial_smoothing_length(fluid_system)
    support = compact_support(fluid_system.smoothing_kernel, one(smoothing_length))
    set_zero!(cache.wetted_area_flooded_reference)
    for particle in active_particles
        normal = extract_svector(normals, boundary_system, particle)
        all(isfinite, normal) ||
            throw(ArgumentError("contact-boundary normals must be finite"))
        normalized_offset = norm(normal) / smoothing_length
        0 < normalized_offset < support ||
            throw(ArgumentError("the magnitude of each active contact-boundary normal must place the physical surface inside the kernel support"))
        reference = volume_scale *
                    wetted_area_halfspace_reference(fluid_system.smoothing_kernel,
                                                    normalized_offset)
        reference > eps(reference) ||
            throw(ArgumentError("contact-boundary flooded colorfield references must be positive"))
        cache.wetted_area_flooded_reference[particle] = reference
    end
    return boundary_system
end

function validate_wetted_area_patch_connectivity(initial_condition, surface_measure,
                                                 active_particles)
    length(active_particles) == 1 && return initial_condition
    spacing = initial_condition.particle_spacing
    area_spacing = sqrt(maximum(surface_measure))
    length_scale = max(spacing > 0 ? spacing : zero(spacing), area_spacing)
    length_scale > 0 ||
        throw(ArgumentError("positive contact surface measures must define a finite patch scale"))
    connection_radius2 = (1.75length_scale)^2
    coordinates = initial_condition.coordinates
    visited = falses(length(surface_measure))
    queue = [first(active_particles)]
    visited[first(queue)] = true
    next_particle = 1
    while next_particle <= length(queue)
        particle = queue[next_particle]
        next_particle += 1
        for neighbor in active_particles
            visited[neighbor] && continue
            distance2 = zero(eltype(coordinates))
            for dim in axes(coordinates, 1)
                distance2 += (coordinates[dim, particle] -
                              coordinates[dim, neighbor])^2
            end
            distance2 <= connection_radius2 || continue
            visited[neighbor] = true
            push!(queue, neighbor)
        end
    end
    all(visited[active_particles]) ||
        throw(ArgumentError("each contact boundary must contain one connected wetted-area patch"))
    return initial_condition
end

@inline function prepare_wetted_area_boundary!(system, neighbor_system,
                                               surface_normal_method)
    return system
end

function prepare_wetted_area_boundary!(system::AbstractFluidSystem, neighbor_system,
                                       surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                      <:WettedAreaContactAngle})
    boundary_cache = wetted_area_boundary_cache(neighbor_system)
    isnothing(boundary_cache) && return system
    (; wetted_area_surface_measure, wetted_area_flooded_reference,
     wetted_area_weight, colorfield) = boundary_cache
    set_zero!(wetted_area_weight)

    raw_area = zero(eltype(system))
    for particle in eachparticle(neighbor_system)
        surface_measure = wetted_area_surface_measure[particle]
        iszero(surface_measure) && continue
        reference = wetted_area_flooded_reference[particle]
        fraction = clamp(colorfield[particle] / reference, zero(reference), one(reference))
        raw_area += surface_measure * cubic_smoothstep(fraction)
    end

    pi_ = convert(eltype(system), pi)
    raw_radius = sqrt(raw_area / pi_)
    edge_shift = system.cache.wetted_area_normalized_edge_shift[] *
                 initial_smoothing_length(system)
    corrected_radius = max(raw_radius - edge_shift, zero(raw_radius))
    corrected_area = pi_ * corrected_radius^2
    area_derivative = raw_radius > eps(raw_radius) ? corrected_radius / raw_radius :
                      zero(raw_radius)
    system.cache.wetted_area_raw_area[] += raw_area
    system.cache.wetted_area[] += corrected_area

    coefficient = wetted_area_coefficient(surface_tension_model(system),
                                          surface_normal_method.contact_model)
    iszero(coefficient) && return system
    for particle in eachparticle(neighbor_system)
        surface_measure = wetted_area_surface_measure[particle]
        iszero(surface_measure) && continue
        reference = wetted_area_flooded_reference[particle]
        fraction = colorfield[particle] / reference
        0 < fraction < 1 || continue
        wetted_area_weight[particle] = area_derivative * surface_measure / reference *
                                       wetted_area_smoothstep_derivative(fraction)
    end
    return system
end

@inline function accumulate_wetted_area_density_conjugate!(system, neighbor_system,
                                                           surface_normal_method,
                                                           particle, neighbor, distance)
    return system
end

@inline function accumulate_wetted_area_density_conjugate!(system::AbstractFluidSystem,
                                                           neighbor_system,
                                                           ::ColorfieldSurfaceNormal{<:Any,
                                                                                     <:WettedAreaContactAngle},
                                                           particle, neighbor, distance)
    boundary_cache = wetted_area_boundary_cache(neighbor_system)
    isnothing(boundary_cache) && return system
    weight = @inbounds boundary_cache.wetted_area_weight[neighbor]
    iszero(weight) && return system
    kernel_value = smoothing_kernel(system, distance, particle)
    @inbounds system.cache.wetted_area_density_conjugate[particle] += weight *
                                                                      kernel_value
    return system
end

@inline function finalize_wetted_area_contact!(system, surface_normal_method, v)
    return system
end

function finalize_wetted_area_contact!(system::AbstractFluidSystem,
                                       surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                      <:WettedAreaContactAngle},
                                       v)
    coefficient = wetted_area_coefficient(surface_tension_model(system),
                                          surface_normal_method.contact_model)
    area = system.cache.wetted_area[]
    system.cache.wetted_area_energy[] = iszero(coefficient) ? zero(coefficient) :
                                        -coefficient * area
    if iszero(coefficient)
        set_zero!(system.cache.wetted_area_density_conjugate)
    else
        for particle in each_integrated_particle(system)
            density = current_density(v, system, particle)
            @inbounds system.cache.wetted_area_density_conjugate[particle] *= coefficient /
                                                                              density^2
        end
    end
    system.cache.wetted_area_evaluations[] += 1
    return system
end

@inline function wetted_area_density_acceleration(surface_normal_method, particle_system,
                                                  neighbor_system, particle, neighbor,
                                                  rho_a, rho_b, m_b, grad_kernel)
    return zero(grad_kernel)
end

@inline function wetted_area_density_acceleration(surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                                 <:WettedAreaContactAngle},
                                                  particle_system::AbstractFluidSystem,
                                                  neighbor_system::AbstractFluidSystem,
                                                  particle, neighbor, rho_a, rho_b, m_b,
                                                  grad_kernel)
    particle_system === neighbor_system || return zero(grad_kernel)
    conjugate_a = @inbounds particle_system.cache.wetted_area_density_conjugate[particle]
    conjugate_b = @inbounds neighbor_system.cache.wetted_area_density_conjugate[neighbor]
    pair_coefficient = conjugate_a * rho_a / rho_b + conjugate_b * rho_b / rho_a
    iszero(pair_coefficient) && return zero(grad_kernel)
    return -m_b * pair_coefficient * grad_kernel
end

@inline function wetted_area_explicit_acceleration(surface_tension,
                                                   surface_normal_method,
                                                   particle_system, neighbor_system,
                                                   particle, neighbor, m_a, rho_a,
                                                   grad_kernel)
    return zero(grad_kernel)
end

@inline function wetted_area_explicit_acceleration(surface_tension,
                                                   surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                                  <:WettedAreaContactAngle},
                                                   particle_system::AbstractFluidSystem,
                                                   neighbor_system, particle, neighbor,
                                                   m_a, rho_a, grad_kernel)
    boundary_cache = wetted_area_boundary_cache(neighbor_system)
    isnothing(boundary_cache) && return zero(grad_kernel)
    weight = @inbounds boundary_cache.wetted_area_weight[neighbor]
    iszero(weight) && return zero(grad_kernel)
    coefficient = wetted_area_coefficient(surface_tension,
                                          surface_normal_method.contact_model)
    iszero(coefficient) && return zero(grad_kernel)
    acceleration = coefficient / rho_a * weight * grad_kernel
    if neighbor_system isa WallBoundarySystem
        thread = Threads.threadid()
        reaction_buffer = boundary_cache.wetted_area_reaction_buffer
        for dim in eachindex(acceleration)
            @inbounds reaction_buffer[dim, neighbor, thread] -= m_a * acceleration[dim]
        end
    end
    return acceleration
end

@inline function surface_normal(particle_system::AbstractFluidSystem, particle)
    (; cache) = particle_system
    return extract_svector(cache.surface_normal, particle_system, particle)
end

@inline function surface_tension_normal(particle_system::AbstractFluidSystem, particle)
    return surface_tension_normal(surface_normal_method(particle_system), particle_system,
                                  particle)
end

@inline function surface_tension_normal(surface_normal_method, particle_system, particle)
    return surface_normal(particle_system, particle)
end

@inline function surface_tension_normal(method::ColorfieldSurfaceNormal, particle_system,
                                        particle)
    method.normal_smoothing || return surface_normal(particle_system, particle)
    return extract_svector(particle_system.cache.smoothed_surface_normal,
                           particle_system, particle)
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
        accumulate_surface_divergence_correction!(system, surface_tension_model(system),
                                                  particle, m_b / density_neighbor,
                                                  pos_diff, grad_kernel)

        cache.neighbor_count[particle] += 1
    end

    return system
end

@inline function accumulate_surface_divergence_correction!(system, surface_tension,
                                                           particle, volume, pos_diff,
                                                           grad_kernel)
    return system
end

@inline function accumulate_surface_divergence_correction!(system,
                                                           ::SurfaceTensionMomentumMorris,
                                                           particle, volume, pos_diff,
                                                           grad_kernel)
    value = -volume * dot(pos_diff, grad_kernel) / ndims(system)
    @inbounds system.cache.divergence_correction[particle] += value
    return system
end

@inline function accumulate_surface_divergence_correction!(system,
                                                           ::SurfaceTensionMorris,
                                                           particle, volume, pos_diff,
                                                           grad_kernel)
    value = -volume * dot(pos_diff, grad_kernel) / ndims(system)
    @inbounds system.cache.support_moment[particle] += value
    return system
end

# Section 2.2 in Akinci et al. 2013 "Versatile Surface Tension and Adhesion for SPH Fluids"
# Note: This is the simplest form of normal approximation commonly used in SPH and comes
# with serious deficits in accuracy especially at corners, small neighborhoods and boundaries
function calc_boundary_normal!(system::AbstractFluidSystem, neighbor_system, u_system, v,
                               v_neighbor_system, u_neighbor_system, semi,
                               surface_normal_method)
    surface_normal_method.boundary_contact_threshold == Inf && return system

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
    prepare_wetted_area_boundary!(system, neighbor_system, surface_normal_method)

    foreach_point_neighbor(system, neighbor_system,
                           system_coords, neighbor_system_coords,
                           semi) do particle, neighbor, pos_diff, distance
        m_b = hydrodynamic_mass(neighbor_system, neighbor)
        density_neighbor = current_density(v_neighbor_system, neighbor_system, neighbor)
        fluid_volume = hydrodynamic_mass(system, particle) /
                       current_density(v, system, particle)
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        accumulate_wetted_area_density_conjugate!(system, neighbor_system,
                                                  surface_normal_method, particle,
                                                  neighbor, distance)

        # Boundary particles complete the quadrature stencil for the continuum-stress
        # divergence, even though the solid itself carries no capillary stress.
        accumulate_surface_divergence_correction!(system, surface_tension_model(system),
                                                  particle, m_b / density_neighbor,
                                                  pos_diff, grad_kernel)

        # We assume that we are in contact with the boundary if the color of the boundary particle
        # is larger than the threshold
        if colorfield[neighbor] / maximum_colorfield > boundary_contact_threshold
            for i in 1:ndims(system)
                cache.surface_normal[i, particle] += fluid_volume * grad_kernel[i]
            end
            accumulate_boundary_normal!(system, surface_normal_method, particle,
                                        fluid_volume, grad_kernel)
            cache.neighbor_count[particle] += 1
        end
    end

    return system
end

@inline function accumulate_boundary_normal!(system, surface_normal_method, particle,
                                             volume, grad_kernel)
    return system
end

@inline function accumulate_boundary_normal!(system,
                                             ::ColorfieldSurfaceNormal{<:Any,
                                                                       <:AbstractContactAngleModel},
                                             particle, volume, grad_kernel)
    for i in 1:ndims(system)
        @inbounds system.cache.boundary_normal[i, particle] += volume * grad_kernel[i]
    end
    return system
end

@inline function accumulate_boundary_normal!(system,
                                             ::ColorfieldSurfaceNormal{<:Any,
                                                                       <:WettedAreaContactAngle},
                                             particle, volume, grad_kernel)
    return system
end

@inline reset_boundary_normal!(system, surface_normal_method) = system

@inline function reset_boundary_normal!(system,
                                        surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                       <:AbstractContactAngleModel})
    set_zero!(system.cache.boundary_normal)
    reset_contact_angle_data!(system, surface_normal_method.contact_model)
    return system
end

@inline function reset_boundary_normal!(system,
                                        surface_normal_method::ColorfieldSurfaceNormal{<:Any,
                                                                                       <:WettedAreaContactAngle})
    reset_contact_angle_data!(system, surface_normal_method.contact_model)
    return system
end

@inline reset_contact_angle_data!(system, contact_model) = system

@inline function reset_contact_angle_data!(system, ::WettedAreaContactAngle)
    set_zero!(system.cache.wetted_area_density_conjugate)
    system.cache.wetted_area_energy[] = zero(eltype(system))
    system.cache.wetted_area_raw_area[] = zero(eltype(system))
    system.cache.wetted_area[] = zero(eltype(system))
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
                                 surface_tension::Union{SurfaceTensionMorris,
                                                        SurfaceTensionMomentumMorris},
                                 surface_normal_method::ColorfieldSurfaceNormal)
    (; cache, smoothing_kernel) = system
    support_radius = compact_support(smoothing_kernel, initial_smoothing_length(system))

    for particle in each_integrated_particle(system)
        store_surface_delta!(system, surface_tension, particle, zero(eltype(system)))
        store_surface_activity!(system, surface_tension, particle, zero(eltype(system)))

        particle_surface_normal = surface_normal(system, particle)
        norm2 = dot(particle_surface_normal, particle_surface_normal)
        if norm2 <= eps(norm2)
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        normal_norm = sqrt(norm2)
        gradient_activity = gradient_interface_activity(normal_norm, support_radius,
                                                        surface_normal_method)
        support_moment = surface_support_moment(system, surface_tension, particle)
        support_activity = support_interface_activity(support_moment,
                                                      surface_normal_method)
        activity = gradient_activity * support_activity
        if iszero(activity)
            cache.surface_normal[1:ndims(system), particle] .= 0
            continue
        end

        store_surface_activity!(system, surface_tension, particle, activity)
        store_surface_delta!(system, surface_tension, particle, normal_norm * activity)
        cache.surface_normal[1:ndims(system),
                             particle] = particle_surface_normal / normal_norm
    end

    return system
end

@inline function smooth_surface_normals!(system, surface_normal_method, v, u, semi)
    return system
end

function smooth_surface_normals!(system::AbstractFluidSystem,
                                 surface_normal_method::ColorfieldSurfaceNormal,
                                 v, u, semi)
    surface_normal_method.normal_smoothing || return system
    cache = system.cache
    normal_sum = cache.smoothed_surface_normal
    weight_sum = cache.normal_smoothing_weight
    coordinates = current_coordinates(u, system)
    set_zero!(normal_sum)
    set_zero!(weight_sum)

    @trixi_timeit timer() "smooth surface normals" begin
        foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                               points=each_integrated_particle(system)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            activity = surface_interface_activity(system, neighbor)
            activity > zero(activity) || return
            volume = hydrodynamic_mass(system, neighbor) /
                     current_density(v, system, neighbor)
            weight = activity * volume * smoothing_kernel(system, distance, particle)
            normal = surface_normal(system, neighbor)
            for dimension in 1:ndims(system)
                @inbounds normal_sum[dimension, particle] += weight * normal[dimension]
            end
            @inbounds weight_sum[particle] += weight
        end
    end

    for particle in each_integrated_particle(system)
        surface_interface_activity(system, particle) > zero(eltype(system)) || continue
        weight = @inbounds weight_sum[particle]
        normal = extract_svector(normal_sum, system, particle)
        normal_norm = norm(normal)
        raw_normal = surface_normal(system, particle)
        use_smoothed_normal = weight > eps(weight) && normal_norm > eps(normal_norm)
        for dimension in 1:ndims(system)
            @inbounds normal_sum[dimension,
                                 particle] = use_smoothed_normal ?
                                             normal[dimension] / normal_norm :
                                             raw_normal[dimension]
        end
    end
    return system
end

@inline store_surface_delta!(system, surface_tension, particle, value) = system

@inline function store_surface_delta!(system,
                                      ::Union{SurfaceTensionMorris,
                                              SurfaceTensionMomentumMorris},
                                      particle, value)
    # Only the fluid half of the symmetric, kernel-smoothed interface is sampled in a
    # one-phase free-surface simulation. Multiplying by two gives a surface delta whose
    # integral through the represented half-interface is one.
    @inbounds system.cache.delta_s[particle] = 2 * value
    return system
end

@inline store_surface_activity!(system, surface_tension, particle, value) = system

@inline function store_surface_activity!(system,
                                         ::Union{SurfaceTensionMorris,
                                                 SurfaceTensionMomentumMorris},
                                         particle, value)
    @inbounds system.cache.interface_activity[particle] = value
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
    reset_surface_divergence_correction!(system, surface_tension)
    reset_boundary_normal!(system, surface_normal_method_)

    # TODO: if color values are set only different systems need to be called
    @trixi_timeit timer() "compute surface normal" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        calc_normal!(system, neighbor_system, u, v, v_neighbor_system,
                     u_neighbor_system, semi, surface_normal_method_,
                     surface_normal_method(neighbor_system))
    end
    finalize_wetted_area_contact!(system, surface_normal_method_, v)
    remove_invalid_normals!(system, surface_tension, surface_normal_method_)
    smooth_surface_normals!(system, surface_normal_method_, v, u, semi)
    compute_contact_angle_cache!(system, surface_normal_method_, v, u, v_ode, u_ode,
                                 semi)

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
    outside_fraction = wetted_area_halfspace_reference(system_smoothing_kernel(system),
                                                       normalized_distance)
    return one(outside_fraction) - outside_fraction
end

@inline ccsf_has_boundary_geometry(::CorrectedCSFSurfaceNormal{Nothing}) = false
@inline ccsf_has_boundary_geometry(::CorrectedCSFSurfaceNormal{<:Real}) = true

@inline function ccsf_boundary_cache(system)
    hasproperty(system, :boundary_model) || return nothing
    cache = system.boundary_model.cache
    haskey(cache, :wetted_area_surface_measure) || return nothing
    return cache
end

@inline function ccsf_face_geometry(system, neighbor_system, particle, neighbor,
                                    pos_diff)
    boundary_cache = ccsf_boundary_cache(neighbor_system)
    isnothing(boundary_cache) && return nothing
    surface_measure = @inbounds boundary_cache.wetted_area_surface_measure[neighbor]
    surface_measure > zero(surface_measure) || return nothing
    normals = neighbor_system.initial_condition.normals
    isnothing(normals) && return nothing
    offset = extract_svector(normals, neighbor_system, neighbor)
    offset_norm = norm(offset)
    offset_norm > eps(offset_norm) || return nothing
    wall_normal = offset / offset_norm # Points from the physical face into the wall.
    face_diff = -pos_diff - offset     # x_face - x_fluid
    kernel_distance = norm(face_diff)
    kernel_distance < compact_support(system, neighbor_system) || return nothing
    boundary_distance = abs(dot(face_diff, wall_normal))
    return (; surface_measure, wall_normal, face_diff, kernel_distance,
            boundary_distance)
end

@inline function accumulate_ccsf_boundary_moments!(system, neighbor_system, v, u,
                                                   v_neighbor, u_neighbor, semi,
                                                   method)
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
        kernel = smoothing_kernel(system, kernel_distance, particle)
        weight = surface_measure * kernel
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
                          (current_angle - target_angle) *
                          (distance / support)^2
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
    set_zero!(lambda_gradient)
    set_zero!(color_gradient)
    set_zero!(shepard_sum)
    set_zero!(cache.ccsf_boundary_normal)
    fill!(cache.ccsf_boundary_distance, typemax(eltype(cache.ccsf_boundary_distance)))

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
                                         particle] += volume_b *
                                                      grad_kernel[dimension]
            end
            @inbounds shepard_sum[particle] += volume_b * kernel
            @inbounds cache.neighbor_count[particle] += 1
        end
    end

    if ccsf_has_boundary_geometry(surface_normal_method(system))
        foreach_system(semi) do neighbor_system
            v_neighbor = wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)
            accumulate_ccsf_boundary_moments!(system, neighbor_system, v, u,
                                              v_neighbor, u_neighbor, semi,
                                              surface_normal_method(system))
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

    if ccsf_has_boundary_geometry(surface_normal_method(system))
        foreach_system(semi) do neighbor_system
            u_neighbor = wrap_u(u_ode, neighbor_system, semi)
            accumulate_ccsf_boundary_lambda_gradient!(system, neighbor_system, u,
                                                      u_neighbor, semi,
                                                      surface_normal_method(system))
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
    apply_ccsf_contact_normal!(system, surface_normal_method(system))

    return system
end

@inline function compute_contact_angle_cache!(system, surface_normal_method, v, u,
                                              v_ode, u_ode, semi)
    return system
end

@inline reset_surface_divergence_correction!(system, surface_tension) = system

@inline function reset_surface_divergence_correction!(system,
                                                      ::SurfaceTensionMomentumMorris)
    set_zero!(system.cache.divergence_correction)
    return system
end

@inline function reset_surface_divergence_correction!(system, ::SurfaceTensionMorris)
    set_zero!(system.cache.support_moment)
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
        n_a = surface_tension_normal(system, particle)
        n_b = surface_tension_normal(neighbor_system, neighbor)
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
