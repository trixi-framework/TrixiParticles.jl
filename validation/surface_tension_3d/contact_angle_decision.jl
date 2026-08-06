using CSV
using DataFrames
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "compare_akinci", "css_sessile_drop.jl"))

const PHASE2_MATRIX = joinpath(@__DIR__, "sessile_drop_matrix.csv")
const PHASE2_SENSITIVITY = joinpath(@__DIR__, "sessile_drop_sensitivity.csv")
const SCORECARD_PATH = joinpath(@__DIR__, "contact_angle_scorecard.csv")
const THRESHOLD_PATH = joinpath(@__DIR__, "contact_angle_threshold_replay.csv")
const PERTURBATION_PATH = joinpath(@__DIR__, "contact_angle_perturbation.csv")
const TIMESTEP_PATH = joinpath(@__DIR__, "contact_angle_timestep.csv")
const COST_PATH = joinpath(@__DIR__, "contact_angle_cost.csv")
const SELECTED_PATH = joinpath(@__DIR__, "contact_angle_selected_matrix.csv")
const NORMAL_COMPONENTS_PATH = joinpath(@__DIR__, "contact_angle_normal_components.csv")
const FORCE_SIGN_PATH = joinpath(@__DIR__, "contact_angle_force_sign.csv")
const GHOST_FORCE_SIGN_PATH = joinpath(@__DIR__,
                                       "contact_angle_force_sign_ghost_geometric.csv")
const WALL_ENERGY_FORCE_SIGN_PATH = joinpath(@__DIR__,
                                             "contact_angle_force_sign_wall_energy.csv")
const LINE_NORMALIZATION_PATH = joinpath(@__DIR__, "contact_line_normalization.csv")
const CAP_TRANSFER_PATH = joinpath(@__DIR__, "contact_line_cap_transfer.csv")
const WETTED_AREA_PATH = joinpath(@__DIR__, "wetted_area_measure.csv")
const WETTED_AREA_FORCE_SIGN_PATH = joinpath(@__DIR__,
                                             "contact_angle_force_sign_wetted_area.csv")
const RECOVERY_COMPARISON_PATH = joinpath(@__DIR__,
                                          "contact_angle_recovery_comparison.csv")
const MEASURE_PROTOCOL_PATH = joinpath(@__DIR__, "contact_measure_protocol.csv")
const EXTENDED_CAP_PATH = joinpath(@__DIR__, "contact_angle_recovery_extended.csv")
const CORRECTED_WETTED_AREA_PATH = joinpath(@__DIR__, "wetted_area_corrected.csv")
const EXTENDED_FORCE_SIGN_PATH = joinpath(@__DIR__,
                                          "contact_angle_force_sign_extended.csv")
const EXTENDED_COMPARISON_PATH = joinpath(@__DIR__,
                                          "contact_angle_recovery_extended_comparison.csv")
const R4_STATIC_PATH = joinpath(@__DIR__, "contact_angle_static_r4_wetted_area.csv")
const R4_PERTURBATION_PATH = joinpath(@__DIR__,
                                      "contact_angle_perturbation_r4_wetted_area.csv")
const R4_PERTURBATION_CLASSIFIED_PATH = joinpath(@__DIR__,
                                                 "contact_angle_perturbation_r4_wetted_area_classified.csv")
const R4_PERTURBATION_EXTENDED_PATH = joinpath(@__DIR__,
                                               "contact_angle_perturbation_r4_wetted_area_extended.csv")
const R4_THRESHOLD_PATH = joinpath(@__DIR__,
                                   "contact_angle_threshold_r4_wetted_area.csv")
const R4_TIMESTEP_PATH = joinpath(@__DIR__,
                                  "contact_angle_timestep_r4_wetted_area.csv")
const R4_COST_PATH = joinpath(@__DIR__, "contact_angle_cost_r4_wetted_area.csv")
const R4_ACTIVE_COST_PATH = joinpath(@__DIR__,
                                     "contact_angle_cost_r4_wetted_area_active.csv")
const R4_SELECTED_PATH = joinpath(@__DIR__,
                                  "contact_angle_selected_matrix_r4_wetted_area.csv")
const R4_SENSITIVITY_PATH = joinpath(@__DIR__,
                                     "contact_angle_sensitivity_r4_wetted_area.csv")
const PRODUCTION_STATIC_PATH = joinpath(@__DIR__,
                                        "contact_angle_static_r4_wetted_area_production.csv")
const PRODUCTION_PERTURBATION_PATH = joinpath(@__DIR__,
                                              "contact_angle_perturbation_r4_wetted_area_production.csv")
const PRODUCTION_PERTURBATION_EXTENDED_PATH = joinpath(@__DIR__,
                                                       "contact_angle_perturbation_r4_wetted_area_extended_production.csv")
const PRODUCTION_THRESHOLD_PATH = joinpath(@__DIR__,
                                           "contact_angle_threshold_r4_wetted_area_production.csv")
const PRODUCTION_TIMESTEP_PATH = joinpath(@__DIR__,
                                          "contact_angle_timestep_r4_wetted_area_production.csv")
const PRODUCTION_COST_PATH = joinpath(@__DIR__,
                                      "contact_angle_cost_r4_wetted_area_production.csv")
const PRODUCTION_ACTIVE_COST_PATH = joinpath(@__DIR__,
                                             "contact_angle_cost_r4_wetted_area_active_production.csv")
const PRODUCTION_SELECTED_PATH = joinpath(@__DIR__,
                                          "contact_angle_selected_matrix_r4_wetted_area_production.csv")
const PRODUCTION_SENSITIVITY_PATH = joinpath(@__DIR__,
                                             "contact_angle_sensitivity_r4_wetted_area_production.csv")

# These labels identify committed pre-D5 evidence. They are not runnable model selectors.
const REJECTED_MECHANISMS = (:geometric, :contact_line_force)
const HISTORICAL_MECHANISMS = (:none, REJECTED_MECHANISMS...)
const TARGET_ANGLES = (30.0, 60.0, 90.0, 120.0, 150.0)
const RESOLUTIONS = (750, 1500, 3000)
const LINE_STUDY_RESOLUTIONS = (2, 4, 8)
const CAP_PHASES = ((-0.4375, -0.4375), (-0.3125, -0.0625),
                    (-0.1875, 0.3125), (-0.0625, -0.3125),
                    (0.0625, 0.0625), (0.1875, 0.4375),
                    (0.3125, -0.1875), (0.4375, 0.1875))
const FIXED_H_CELLS_PER_H = (1.4, 2.8, 4.2)
const LINE_STUDY_KERNELS = (("gaussian", TrixiParticles.GaussianKernel{3}()),
                            ("schoenberg_cubic",
                             TrixiParticles.SchoenbergCubicSplineKernel{3}()),
                            ("schoenberg_quartic",
                             TrixiParticles.SchoenbergQuarticSplineKernel{3}()),
                            ("schoenberg_quintic",
                             TrixiParticles.SchoenbergQuinticSplineKernel{3}()),
                            ("wendland_c2", TrixiParticles.WendlandC2Kernel{3}()),
                            ("wendland_c4", TrixiParticles.WendlandC4Kernel{3}()),
                            ("wendland_c6", TrixiParticles.WendlandC6Kernel{3}()),
                            ("poly6", TrixiParticles.Poly6Kernel{3}()),
                            ("spiky", TrixiParticles.SpikyKernel{3}()),
                            ("laguerre_gauss",
                             TrixiParticles.LaguerreGaussKernel{3}()))

require(condition, message) = condition || error(message)

function quiet_css_sessile_drop(args...; kwargs...)
    return redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            css_sessile_drop(args...; kwargs...)
        end
    end
end

function weighted_quantile(values, weights, probability)
    isempty(values) && return NaN
    order = sortperm(values)
    sorted_values = values[order]
    cumulative_weight = cumsum(weights[order])
    total_weight = last(cumulative_weight)
    total_weight > 0 || return NaN
    index = searchsortedfirst(cumulative_weight, probability * total_weight)
    return sorted_values[clamp(index, 1, length(sorted_values))]
end

function raw_normal_components(result)
    semi = result.solution.prob.p.semi
    system = semi.systems[1]
    v_ode, u_ode = result.solution.prob.u0.x

    # First retain the production contact-line weights at the initial state.
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    line_delta = haskey(system.cache, :contact_line_delta) ?
                 copy(system.cache.contact_line_delta) :
                 zeros(eltype(system), TrixiParticles.nparticles(system))
    surface_delta = copy(system.cache.delta_s)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    density = collect(TrixiParticles.current_density(v, system))
    volume = system.mass ./ density

    # Repeat only the raw normal accumulation and stop before contact-angle application and
    # normalization. The boundary contribution is accumulated identically in both arrays.
    method = TrixiParticles.surface_normal_method(system)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    GC.@preserve v_ode u_ode begin
        TrixiParticles.set_zero!(system.cache.surface_normal)
        TrixiParticles.set_zero!(system.cache.neighbor_count)
        TrixiParticles.reset_surface_divergence_correction!(system,
                                                            system.surface_tension)
        TrixiParticles.reset_boundary_normal!(system, method)
        TrixiParticles.foreach_system(semi) do neighbor_system
            u_neighbor = TrixiParticles.wrap_u(u_ode, neighbor_system, semi)
            v_neighbor = TrixiParticles.wrap_v(v_ode, neighbor_system, semi)
            TrixiParticles.calc_normal!(system, neighbor_system, u, v,
                                        v_neighbor, u_neighbor, semi, method,
                                        TrixiParticles.surface_normal_method(neighbor_system))
        end
    end

    total_normal = copy(system.cache.surface_normal)
    wall_normal = copy(system.cache.boundary_normal)
    fluid_normal = total_normal - wall_normal
    support_moment = copy(system.cache.divergence_correction)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    return (; total_normal, wall_normal, fluid_normal, line_delta, surface_delta,
            support_moment, volume, coordinates)
end

function shape_acceleration_from_caches(result)
    semi = result.solution.prob.p.semi
    system = semi.systems[1]
    v_ode, u_ode = result.solution.prob.u0.x
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    dv = zeros(eltype(v), size(v))
    TrixiParticles.reset_interaction_caches!(semi)
    GC.@preserve v_ode u_ode begin
        TrixiParticles.foreach_system(semi) do neighbor_system
            v_neighbor = TrixiParticles.wrap_v(v_ode, neighbor_system, semi)
            u_neighbor = TrixiParticles.wrap_u(u_ode, neighbor_system, semi)
            TrixiParticles.interact!(dv, v, u, v_neighbor, u_neighbor,
                                     system, neighbor_system, semi)
        end
    end
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    return cap_shape_acceleration(Array(dv[1:3, :]), coordinates)
end

function ghost_geometric_shape_acceleration(result, components, target)
    system = result.solution.prob.p.semi.systems[1]
    (; total_normal, wall_normal, surface_delta) = components
    contact_sine = sind(target)
    abs(contact_sine) > sqrt(eps()) || return NaN
    contact_cotangent = cosd(target) / contact_sine

    for particle in TrixiParticles.eachparticle(system)
        raw_total = total_normal[:, particle]
        total_norm = norm(raw_total)
        if total_norm <= eps(total_norm)
            system.cache.surface_normal[:, particle] .= 0
            system.cache.delta_s[particle] = 0
            continue
        end
        activity = surface_delta[particle] / (2total_norm)
        raw_wall = wall_normal[:, particle]
        wall_norm = norm(raw_wall)
        desired_normal = raw_total
        if wall_norm > eps(wall_norm)
            wall = raw_wall / wall_norm
            tangent = raw_total - dot(raw_total, wall) * wall
            tangent_norm = norm(tangent)
            if tangent_norm > eps(tangent_norm)
                desired_normal = tangent + tangent_norm * contact_cotangent * wall
            end
        end
        desired_norm = norm(desired_normal)
        if desired_norm > eps(desired_norm)
            system.cache.surface_normal[:, particle] .= desired_normal / desired_norm
            system.cache.delta_s[particle] = 2desired_norm * activity
        else
            system.cache.surface_normal[:, particle] .= 0
            system.cache.delta_s[particle] = 0
        end
    end
    return shape_acceleration_from_caches(result)
end

function wall_energy_shape_acceleration(result, target; line_delta_scale=1.0)
    semi = result.solution.prob.p.semi
    system = semi.systems[1]
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    acceleration = zeros(eltype(system), 3, TrixiParticles.nparticles(system))
    sigma = system.surface_tension.surface_tension_coefficient
    for particle in TrixiParticles.eachparticle(system)
        delta = system.cache.contact_line_delta[particle]
        delta > eps(delta) || continue
        wall = system.cache.boundary_normal[:, particle]
        interface = system.cache.surface_normal[:, particle]
        dynamic_cosine = dot(wall, interface)
        tangent = interface - dynamic_cosine * wall
        tangent_norm = norm(tangent)
        tangent_norm > eps(tangent_norm) || continue
        direction = -tangent / tangent_norm
        density = TrixiParticles.current_density(v, system, particle)
        acceleration[:,
                     particle] .= sigma / density * cosd(target) *
                                  line_delta_scale * delta * direction
    end
    return cap_shape_acceleration(acceleration, coordinates)
end

function kernel_plane_profile(kernel; samples=20_000)
    smoothing_length = 1.0
    support = TrixiParticles.compact_support(kernel, smoothing_length)
    spacing = support / samples
    radii = collect(range(0.0, support; length=samples + 1))
    values = TrixiParticles.kernel.(Ref(kernel), radii, smoothing_length)

    radial_integrand = 2pi .* radii .* values
    convolution_profile = zeros(length(radii))
    for index in samples:-1:1
        convolution_profile[index] = convolution_profile[index + 1] +
                                     spacing * (radial_integrand[index] +
                                      radial_integrand[index + 1]) / 2
    end
    halfspace_color_profile = zeros(length(radii))
    for index in samples:-1:1
        halfspace_color_profile[index] = halfspace_color_profile[index + 1] +
                                         spacing * (convolution_profile[index] +
                                          convolution_profile[index + 1]) / 2
    end
    kernel_half_profile_integral = spacing *
                                   (sum(convolution_profile) -
                                    (convolution_profile[1] +
                                     convolution_profile[end]) / 2)
    mass_integrand = 4pi .* radii .^ 2 .* values
    kernel_mass = spacing *
                  (sum(mass_integrand) -
                   (mass_integrand[1] + mass_integrand[end]) / 2)

    # `kernel_grad` differentiates only inside the strict compact support. Account for a
    # nonzero value at the truncation radius (notably Laguerre-Gauss) when deriving the
    # half-space gradient represented by that operator.
    cutoff_value = TrixiParticles.kernel(kernel, prevfloat(support), smoothing_length)
    gradient_profile = convolution_profile .-
                       pi .* (support^2 .- radii .^ 2) .* cutoff_value
    half_profile_integral = spacing *
                            (sum(gradient_profile) -
                             (gradient_profile[1] + gradient_profile[end]) / 2)
    return (; support, spacing, profile=gradient_profile,
            halfspace_color_profile, half_profile_integral,
            kernel_half_profile_integral, kernel_mass, cutoff_value)
end

@inline function plane_profile_value(profile_data, distance)
    distance >= profile_data.support && return 0.0
    coordinate = max(distance, 0.0) / profile_data.spacing
    lower = floor(Int, coordinate) + 1
    fraction = coordinate - floor(coordinate)
    return (1 - fraction) * profile_data.profile[lower] +
           fraction * profile_data.profile[lower + 1]
end

@inline function halfspace_color_value(profile_data, distance)
    distance >= profile_data.support && return 0.0
    coordinate = max(distance, 0.0) / profile_data.spacing
    lower = floor(Int, coordinate) + 1
    fraction = coordinate - floor(coordinate)
    return (1 - fraction) * profile_data.halfspace_color_profile[lower] +
           fraction * profile_data.halfspace_color_profile[lower + 1]
end

function kernel_lattice_offsets(kernel, cells_per_h)
    lattice_spacing = 1 / cells_per_h
    support = TrixiParticles.compact_support(kernel, 1.0)
    search_radius = ceil(Int, support / lattice_spacing)
    offsets = NamedTuple[]
    for z_offset in (-search_radius):search_radius,
        x_offset in (-search_radius):search_radius
        planar_distance2 = lattice_spacing^2 * (x_offset^2 + z_offset^2)
        planar_distance2 < support^2 || continue
        gradient_x = 0.0
        gradient_z = 0.0
        kernel_value = 0.0
        for tangent_offset in (-search_radius):search_radius
            pos_diff = lattice_spacing *
                       SVector(x_offset, tangent_offset, z_offset)
            distance = norm(pos_diff)
            distance < support || continue
            gradient = TrixiParticles.kernel_grad(kernel, pos_diff, distance, 1.0)
            gradient_x += lattice_spacing * gradient[1]
            gradient_z += lattice_spacing * gradient[3]
            kernel_value += lattice_spacing *
                            TrixiParticles.kernel(kernel, distance, 1.0)
        end
        push!(offsets, (; x_offset, z_offset, gradient_x, gradient_z, kernel_value))
    end
    return offsets
end

function planar_line_measure_row(kernel_name, profile_data, offsets, angle, cells_per_h)
    contact_sine = sind(angle)
    contact_cosine = cosd(angle)
    lattice_spacing = 1 / cells_per_h
    support = profile_data.support
    horizontal_radius = support * (1 + abs(contact_cosine)) / contact_sine +
                        2lattice_spacing
    horizontal_cells = ceil(Int, horizontal_radius / lattice_spacing)
    vertical_cells = ceil(Int, support / lattice_spacing)
    profile_coarea_measure = 0.0
    coarea_measure = 0.0
    wedge_coarea_measure = 0.0
    gated_coarea_measure = 0.0
    divergence_measure = 0.0
    phase_fractions = (0.0, 0.25, 0.5, 0.75)

    for phase_fraction in phase_fractions
        interface_shift = phase_fraction * lattice_spacing * contact_sine
        boundary_colorfield = Dict{Tuple{Int, Int}, Float64}()
        wall_layers = ceil(Int, support / lattice_spacing)
        flooded_colorfield = 0.0
        for wall_index in (-wall_layers):-1
            wall_position = (wall_index + 0.5) * lattice_spacing
            value = 0.0
            for offset in offsets
                fluid_wall_position = wall_position -
                                      offset.z_offset * lattice_spacing
                fluid_wall_position > 0 || continue
                value += lattice_spacing^2 * offset.kernel_value
            end
            flooded_colorfield = max(flooded_colorfield, value)
        end

        function boundary_colorfield_value(x_index, wall_index)
            return get!(boundary_colorfield, (x_index, wall_index)) do
                boundary_x = (x_index + 0.5) * lattice_spacing
                boundary_wall_position = (wall_index + 0.5) * lattice_spacing
                value = 0.0
                for source_offset in offsets
                    fluid_x = boundary_x -
                              source_offset.x_offset * lattice_spacing
                    fluid_wall_position = boundary_wall_position -
                                          source_offset.z_offset * lattice_spacing
                    fluid_interface_position = contact_sine * fluid_x +
                                               contact_cosine * fluid_wall_position -
                                               interface_shift
                    fluid_wall_position > 0 && fluid_interface_position > 0 ||
                        continue
                    value += lattice_spacing^2 * source_offset.kernel_value
                end
                value
            end
        end

        for j in 0:(vertical_cells - 1)
            wall_distance = (j + 0.5) * lattice_spacing
            wall_distance < support || continue
            wall_profile = plane_profile_value(profile_data, wall_distance)
            for i in (-horizontal_cells):(horizontal_cells - 1)
                x = (i + 0.5) * lattice_spacing
                interface_distance = contact_sine * x +
                                     contact_cosine * wall_distance - interface_shift
                0 < interface_distance < support || continue
                interface_profile = plane_profile_value(profile_data, interface_distance)
                profile_coarea_measure += lattice_spacing^2 * contact_sine *
                                          interface_profile * wall_profile

                interface_gradient_x = 0.0
                interface_gradient_z = 0.0
                wedge_gradient_x = 0.0
                wedge_gradient_z = 0.0
                wall_gradient_x = 0.0
                wall_gradient_z = 0.0
                gated_wall_gradient_x = 0.0
                gated_wall_gradient_z = 0.0
                divergence_gradient = 0.0
                delta_prime_a = contact_sine * interface_profile
                for offset in offsets
                    neighbor_x = x - offset.x_offset * lattice_spacing
                    neighbor_wall_distance = wall_distance -
                                             offset.z_offset * lattice_spacing
                    neighbor_interface_distance = contact_sine * neighbor_x +
                                                  contact_cosine *
                                                  neighbor_wall_distance -
                                                  interface_shift
                    if neighbor_interface_distance > 0
                        interface_gradient_x += lattice_spacing^2 * offset.gradient_x
                        interface_gradient_z += lattice_spacing^2 * offset.gradient_z
                        if neighbor_wall_distance > 0
                            wedge_gradient_x += lattice_spacing^2 * offset.gradient_x
                            wedge_gradient_z += lattice_spacing^2 * offset.gradient_z
                        end
                    end
                    if neighbor_wall_distance < 0
                        wall_gradient_x += lattice_spacing^2 * offset.gradient_x
                        wall_gradient_z += lattice_spacing^2 * offset.gradient_z
                        neighbor_x_index = i - offset.x_offset
                        neighbor_wall_index = j - offset.z_offset
                        color_fraction = boundary_colorfield_value(neighbor_x_index,
                                                                   neighbor_wall_index) /
                                         flooded_colorfield
                        if color_fraction > 0.1
                            gated_wall_gradient_x += lattice_spacing^2 *
                                                     offset.gradient_x
                            gated_wall_gradient_z += lattice_spacing^2 *
                                                     offset.gradient_z
                        end
                        delta_prime_b = 0.0
                    elseif neighbor_interface_distance > 0
                        delta_prime_b = contact_sine *
                                        plane_profile_value(profile_data,
                                                            neighbor_interface_distance)
                    else
                        # The one-phase operator has no gas neighbors.
                        continue
                    end
                    divergence_gradient += lattice_spacing^2 *
                                           (delta_prime_b - delta_prime_a) *
                                           offset.gradient_z
                end
                coarea_measure += lattice_spacing^2 *
                                  abs(interface_gradient_x * wall_gradient_z -
                                      interface_gradient_z * wall_gradient_x)
                wedge_coarea_measure += lattice_spacing^2 *
                                        abs(wedge_gradient_x * wall_gradient_z -
                                            wedge_gradient_z * wall_gradient_x)
                gated_coarea_measure += lattice_spacing^2 *
                                        abs(wedge_gradient_x * gated_wall_gradient_z -
                                            wedge_gradient_z * gated_wall_gradient_x)
                divergence_measure += lattice_spacing^2 *
                                      max(2divergence_gradient, 0.0)
            end
        end
    end
    profile_coarea_measure /= length(phase_fractions)
    coarea_measure /= length(phase_fractions)
    wedge_coarea_measure /= length(phase_fractions)
    gated_coarea_measure /= length(phase_fractions)
    divergence_measure /= length(phase_fractions)

    coarea_normalization = inv(profile_data.half_profile_integral^2)
    divergence_normalization = inv(profile_data.half_profile_integral)
    normalized_coarea = coarea_normalization * coarea_measure
    normalized_wedge_coarea = coarea_normalization * wedge_coarea_measure
    normalized_gated_coarea = coarea_normalization * gated_coarea_measure
    normalized_divergence = divergence_normalization * divergence_measure
    return (; kernel=kernel_name, angle, cells_per_h, lattice_spacing,
            phase_count=length(phase_fractions),
            support, kernel_mass=profile_data.kernel_mass,
            cutoff_value=profile_data.cutoff_value,
            kernel_half_profile_integral=profile_data.kernel_half_profile_integral,
            half_profile_integral=profile_data.half_profile_integral,
            half_profile_mass_error=abs(2profile_data.half_profile_integral /
                                        profile_data.kernel_mass - 1),
            coarea_normalization, divergence_normalization,
            continuous_coarea=coarea_normalization *
                              profile_data.half_profile_integral^2,
            profile_coarea_measure,
            normalized_profile_coarea=coarea_normalization * profile_coarea_measure,
            profile_coarea_error=abs(coarea_normalization * profile_coarea_measure - 1),
            coarea_measure, normalized_coarea,
            coarea_error=abs(normalized_coarea - 1),
            wedge_coarea_measure, normalized_wedge_coarea,
            wedge_coarea_error=abs(normalized_wedge_coarea - 1),
            gated_coarea_measure, normalized_gated_coarea,
            gated_coarea_error=abs(normalized_gated_coarea - 1),
            divergence_measure, normalized_divergence,
            divergence_error=abs(normalized_divergence - 1))
end

function run_line_normalization_study(; output_path=LINE_NORMALIZATION_PATH)
    rows = NamedTuple[]
    for (kernel_name, kernel) in LINE_STUDY_KERNELS
        profile_data = kernel_plane_profile(kernel)
        for cells_per_h in LINE_STUDY_RESOLUTIONS
            offsets = kernel_lattice_offsets(kernel, cells_per_h)
            for angle in TARGET_ANGLES
                push!(rows,
                      planar_line_measure_row(kernel_name, profile_data, offsets,
                                              angle, cells_per_h))
            end
        end
    end
    data = DataFrame(rows)
    gate_columns = ((:coarea_error, :coarea_middle_pass,
                     :coarea_endpoint_decreasing),
                    (:wedge_coarea_error, :wedge_middle_pass,
                     :wedge_endpoint_decreasing),
                    (:gated_coarea_error, :gated_middle_pass,
                     :gated_endpoint_decreasing),
                    (:divergence_error, :divergence_middle_pass,
                     :divergence_endpoint_decreasing))
    for (_, middle_column, endpoint_column) in gate_columns
        data[!, middle_column] = falses(nrow(data))
        data[!, endpoint_column] = falses(nrow(data))
    end
    middle_resolution = LINE_STUDY_RESOLUTIONS[2]
    for indices in groupby(data, [:kernel, :angle]; sort=true)
        order = sortperm(indices.cells_per_h)
        first_index = parentindices(indices)[1][order[1]]
        middle_index = parentindices(indices)[1][order[2]]
        last_index = parentindices(indices)[1][order[3]]
        parent_rows = parentindices(indices)[1]
        for (error_column, middle_column, endpoint_column) in gate_columns
            data[parent_rows, middle_column] .= data[middle_index, error_column] <= 0.2
            data[parent_rows,
                 endpoint_column] .= data[last_index, error_column] <
                                     data[first_index, error_column] ||
                                     data[last_index, error_column] <= 1.0e-12
        end
        require(data.cells_per_h[middle_index] == middle_resolution,
                "unexpected line-study resolution order")
    end

    CSV.write(output_path, data)
    middle = data[data.cells_per_h .== middle_resolution, :]
    @printf("kernel-derived coarea: %d/%d middle-resolution and %d/%d endpoint gates\n",
            count(middle.coarea_middle_pass), nrow(middle),
            count(middle.coarea_endpoint_decreasing), nrow(middle))
    for (label, middle_column, endpoint_column) in
        (("wedge coarea", :wedge_middle_pass, :wedge_endpoint_decreasing),
         ("gated coarea", :gated_middle_pass, :gated_endpoint_decreasing),
         ("kernel-derived divergence", :divergence_middle_pass,
          :divergence_endpoint_decreasing))
        @printf("%s: %d/%d middle-resolution and %d/%d endpoint gates\n",
                label, count(middle[!, middle_column]), nrow(middle),
                count(middle[!, endpoint_column]), nrow(middle))
    end
    println("Wrote kernel line-normalization study to ", output_path)
    return data
end

@inline function scaled_plane_profile(profile_data, distance, smoothing_length)
    0 <= distance < profile_data.support * smoothing_length || return 0.0
    return plane_profile_value(profile_data, distance / smoothing_length) /
           smoothing_length
end

function analytic_cap_line_measure(setup, profile_data, smoothing_length)
    coordinates = setup.initial_condition.coordinates
    volume = setup.initial_condition.mass ./ setup.initial_condition.density
    center = SVector(0.0, 0.0, setup.sphere_center_z)
    measure = 0.0
    for particle in axes(coordinates, 2)
        position = SVector{3}(coordinates[:, particle])
        radial = position - center
        radius = norm(radial)
        radius > eps(radius) || continue
        interface_profile = scaled_plane_profile(profile_data,
                                                 setup.sphere_radius - radius,
                                                 smoothing_length)
        wall_profile = scaled_plane_profile(profile_data, position[3], smoothing_length)
        measure += volume[particle] * interface_profile * wall_profile *
                   hypot(radial[1], radial[2]) / radius
    end
    coarea_normalization = inv(profile_data.half_profile_integral^2)
    return (; measure, line_length=coarea_normalization * measure,
            particle_count=size(coordinates, 2), coarea_normalization)
end

function phase_measure_summary(values, analytic_value)
    length(values) == length(CAP_PHASES) ||
        error("phase summary requires $(length(CAP_PHASES)) values")
    mean_value = mean(values)
    relative_standard_error = std(values) /
                              (sqrt(length(values)) * analytic_value)
    phase_values = join((@sprintf("%.17g", value) for value in values), ";")
    return (; phase_values, phase_min=minimum(values), phase_max=maximum(values),
            mean_value,
            signed_error=mean_value / analytic_value - 1,
            error=abs(mean_value / analytic_value - 1),
            relative_standard_error)
end

function apply_phase_measure_gates!(data; error_column=:line_length_error,
                                    standard_error_column=:phase_standard_error,
                                    series_column=nothing,
                                    middle_column=:middle_pass,
                                    endpoint_column=:endpoint_pass)
    data[!, middle_column] = falses(nrow(data))
    data[!, endpoint_column] = falses(nrow(data))
    group_columns = isnothing(series_column) ? [:variant, :target] :
                    [series_column, :target]
    for group in groupby(data, group_columns; sort=true)
        order = sortperm(group.requested_particles)
        length(order) == 3 || error("phase-measure gate requires three resolutions")
        parent_rows = parentindices(group)[1]
        coarse = group[order[1], error_column]
        middle = group[order[2], error_column]
        fine = group[order[3], error_column]
        coarse_se = group[order[1], standard_error_column]
        fine_se = group[order[3], standard_error_column]
        series = isnothing(series_column) ? "production_resolution" :
                 group[order[1], series_column]
        endpoint_pass = if series == "fixed_h_quadrature"
            middle_se = group[order[2], standard_error_column]
            abs(fine - middle) <= abs(middle - coarse) +
                                  2hypot(middle_se, fine_se) && fine <= 0.2
        else
            fine <= 0.2 && fine <= coarse + 2hypot(coarse_se, fine_se)
        end
        data[parent_rows, middle_column] .= middle <= 0.2
        data[parent_rows, endpoint_column] .= endpoint_pass
    end
    return data
end

function measure_protocol_row(series, target, requested_particles, smoothing_length,
                              profile_data)
    line_lengths = Float64[]
    particle_counts = Int[]
    actual_cells_per_h = Float64[]
    analytic_line_length = NaN
    coarea_normalization = NaN
    for lattice_phase in CAP_PHASES
        setup = spherical_cap_initial_condition(target;
                                                target_particle_count=requested_particles,
                                                lattice_phase)
        result = analytic_cap_line_measure(setup, profile_data, smoothing_length)
        push!(line_lengths, result.line_length)
        push!(particle_counts, result.particle_count)
        particle_spacing = setup.initial_condition.particle_spacing
        push!(actual_cells_per_h, smoothing_length / particle_spacing)
        analytic_line_length = 2pi * setup.cap_radius
        coarea_normalization = result.coarea_normalization
    end
    summary = phase_measure_summary(line_lengths, analytic_line_length)
    return (; series, variant="analytic_both_control", target, requested_particles,
            particle_count_min=minimum(particle_counts),
            particle_count_max=maximum(particle_counts),
            smoothing_length, cells_per_h=mean(actual_cells_per_h),
            phase_count=length(CAP_PHASES), coarea_normalization,
            line_length_phases=summary.phase_values,
            line_length_phase_min=summary.phase_min,
            line_length_phase_max=summary.phase_max,
            line_length=summary.mean_value, analytic_line_length,
            signed_error=summary.signed_error,
            line_length_error=summary.error,
            phase_standard_error=summary.relative_standard_error)
end

function run_measure_protocol(; output_path=MEASURE_PROTOCOL_PATH)
    kernel = TrixiParticles.WendlandC2Kernel{3}()
    profile_data = kernel_plane_profile(kernel)
    rows = NamedTuple[]
    drop_volume = 1.0e-6
    for target in TARGET_ANGLES, resolution in RESOLUTIONS
        particle_spacing = cbrt(drop_volume / resolution)
        push!(rows,
              measure_protocol_row("production_resolution", target, resolution,
                                   1.4particle_spacing, profile_data))
    end

    fixed_smoothing_length = 1.4cbrt(drop_volume / RESOLUTIONS[2])
    for target in TARGET_ANGLES, cells_per_h in FIXED_H_CELLS_PER_H
        resolution = round(Int, RESOLUTIONS[2] * (cells_per_h / 1.4)^3)
        push!(rows,
              measure_protocol_row("fixed_h_quadrature", target, resolution,
                                   fixed_smoothing_length, profile_data))
    end

    data = DataFrame(rows)
    apply_phase_measure_gates!(data; series_column=:series)
    data.protocol_pass = falses(nrow(data))
    for target in TARGET_ANGLES
        target_rows = data[data.target .== target, :]
        pass = all(target_rows.middle_pass .& target_rows.endpoint_pass)
        data[data.target .== target, :protocol_pass] .= pass
    end
    sort!(data, [:series, :requested_particles, :target])
    CSV.write(output_path, data)

    production_middle = data[(data.series .== "production_resolution") .& (data.requested_particles .== RESOLUTIONS[2]),
                             :]
    quadrature_middle_resolution = round(Int,
                                         RESOLUTIONS[2] *
                                         (FIXED_H_CELLS_PER_H[2] / 1.4)^3)
    quadrature_middle = data[(data.series .== "fixed_h_quadrature") .& (data.requested_particles .== quadrature_middle_resolution),
                             :]
    @printf("phase-averaged exact control: production middle %d/5, endpoints %d/5; fixed-h middle %d/5, endpoints %d/5\n",
            count(production_middle.middle_pass),
            count(production_middle.endpoint_pass),
            count(quadrature_middle.middle_pass),
            count(quadrature_middle.endpoint_pass))
    println("Wrote amended contact-measure protocol to ", output_path)
    return data
end

function analytic_cap_gradients(result, components, setup, profile_data)
    system = result.solution.prob.p.semi.systems[1]
    smoothing_length = TrixiParticles.initial_smoothing_length(system)
    coordinates = components.coordinates
    interface_gradient = zeros(eltype(coordinates), size(coordinates))
    wall_gradient = zeros(eltype(coordinates), size(coordinates))
    center = SVector(0.0, 0.0, setup.sphere_center_z)

    for particle in axes(coordinates, 2)
        position = SVector{3}(coordinates[:, particle])
        radial = position - center
        radius = norm(radial)
        interface_distance = setup.sphere_radius - radius
        interface_profile = scaled_plane_profile(profile_data, interface_distance,
                                                 smoothing_length)
        if radius > eps(radius) && interface_profile > 0
            interface_gradient[:, particle] .= -interface_profile * radial / radius
        end

        wall_profile = scaled_plane_profile(profile_data, position[3], smoothing_length)
        wall_gradient[3, particle] = -wall_profile
    end
    return (; interface_gradient, wall_gradient)
end

function compatible_indicator_gradients(result, components)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)
    boundary_coordinates = TrixiParticles.current_coordinates(u_boundary, boundary_system)
    colorfield = boundary_system.boundary_model.cache.colorfield
    colorfield_reference = maximum(colorfield)
    wall_gradient = zeros(eltype(system), size(components.fluid_normal))
    continuation_gradient = similar(wall_gradient)
    fill!(continuation_gradient, 0)

    TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                          boundary_coordinates,
                                          semi) do particle, neighbor,
                                                   pos_diff, distance
        density = TrixiParticles.current_density(v, system, particle)
        volume = TrixiParticles.hydrodynamic_mass(system, particle) / density
        gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                        particle)
        wetness = clamp(colorfield[neighbor] / colorfield_reference, 0, 1)
        wall_gradient[:, particle] .+= volume * gradient
        continuation_gradient[:, particle] .+= volume * wetness * gradient
    end
    interface_gradient = components.fluid_normal + continuation_gradient
    return (; interface_gradient, wall_gradient, colorfield_reference)
end

function geometry_wall_gradient(result, components, profile_data)
    system = result.solution.prob.p.semi.systems[1]
    smoothing_length = TrixiParticles.initial_smoothing_length(system)
    wall_gradient = zeros(eltype(system), size(components.fluid_normal))
    for particle in axes(components.coordinates, 2)
        wall_profile = scaled_plane_profile(profile_data,
                                            components.coordinates[3, particle],
                                            smoothing_length)
        wall_gradient[3, particle] = -wall_profile
    end
    return wall_gradient
end

@inline function young_ghost_fraction(surface_fraction, tangent_gradient, depth, angle)
    tangent_gradient > eps(tangent_gradient) || return surface_fraction
    contact_sine = sind(angle)
    if abs(contact_sine) <= sqrt(eps(contact_sine))
        return cosd(angle) > 0 ? 1.0 : 0.0
    end
    return clamp(surface_fraction +
                 depth * tangent_gradient * cosd(angle) / contact_sine, 0, 1)
end

function young_boundary_gradients(result, components, target, profile_data)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)
    boundary_coordinates = Array(TrixiParticles.current_coordinates(u_boundary,
                                                                    boundary_system))
    particle_spacing = system.cache.reference_particle_spacing
    colorfield = boundary_system.boundary_model.cache.colorfield
    colorfield_reference = canonical_flooded_reference(result, profile_data).reference
    color_fraction = clamp.(colorfield ./ colorfield_reference, 0, 1)
    exposed_height = maximum(boundary_coordinates[3, :])
    exposed = isapprox.(boundary_coordinates[3, :], exposed_height;
                        atol=10eps(abs(exposed_height) + particle_spacing))
    minimum_x = minimum(boundary_coordinates[1, exposed])
    minimum_y = minimum(boundary_coordinates[2, exposed])
    surface_fraction = Dict{Tuple{Int, Int}, Float64}()
    for particle in findall(exposed)
        key = (round(Int,
                     (boundary_coordinates[1, particle] - minimum_x) /
                     particle_spacing),
               round(Int,
                     (boundary_coordinates[2, particle] - minimum_y) /
                     particle_spacing))
        surface_fraction[key] = color_fraction[particle]
    end
    tangent_gradient = Dict{Tuple{Int, Int}, Float64}()
    for (key, center) in surface_fraction
        i, j = key
        gradient_x = (get(surface_fraction, (i + 1, j), center) -
                      get(surface_fraction, (i - 1, j), center)) /
                     (2particle_spacing)
        gradient_y = (get(surface_fraction, (i, j + 1), center) -
                      get(surface_fraction, (i, j - 1), center)) /
                     (2particle_spacing)
        tangent_gradient[key] = hypot(gradient_x, gradient_y)
    end

    # Impose Young's normal derivative on scalar ghost values before assembling the gradient.
    ghost_fraction = similar(color_fraction)
    for particle in eachindex(ghost_fraction)
        key = (round(Int,
                     (boundary_coordinates[1, particle] - minimum_x) /
                     particle_spacing),
               round(Int,
                     (boundary_coordinates[2, particle] - minimum_y) /
                     particle_spacing))
        surface_value = surface_fraction[key]
        depth = -boundary_coordinates[3, particle]
        ghost_fraction[particle] = young_ghost_fraction(surface_value,
                                                        tangent_gradient[key], depth,
                                                        target)
    end

    continuation_gradient = zeros(eltype(system), size(components.fluid_normal))
    TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                          boundary_coordinates,
                                          semi) do particle, neighbor,
                                                   pos_diff, distance
        density = TrixiParticles.current_density(v, system, particle)
        volume = TrixiParticles.hydrodynamic_mass(system, particle) / density
        gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                        particle)
        continuation_gradient[:, particle] .+= volume * ghost_fraction[neighbor] *
                                               gradient
    end
    interface_gradient = components.fluid_normal + continuation_gradient
    wall_gradient = geometry_wall_gradient(result, components, profile_data)
    return (; interface_gradient, wall_gradient, colorfield_reference,
            ghost_fraction_min=minimum(ghost_fraction),
            ghost_fraction_max=maximum(ghost_fraction))
end

function gradient_measure_metrics(components, interface_gradient, wall_gradient,
                                  coarea_normalization)
    measure = 0.0
    weighted_angle = 0.0
    valid_particles = 0
    for particle in eachindex(components.volume)
        interface = SVector{3}(interface_gradient[:, particle])
        wall = SVector{3}(wall_gradient[:, particle])
        interface_norm = norm(interface)
        wall_norm = norm(wall)
        interface_norm > eps(interface_norm) && wall_norm > eps(wall_norm) || continue
        weight = components.volume[particle] * norm(cross(interface, wall))
        weight > eps(weight) || continue
        angle = acosd(clamp(dot(interface, wall) / (interface_norm * wall_norm), -1, 1))
        measure += weight
        weighted_angle += weight * angle
        valid_particles += 1
    end
    line_length = coarea_normalization * measure
    angle = measure > eps(measure) ? weighted_angle / measure : NaN
    return (; measure, line_length, angle, valid_particles)
end

function run_extended_cap_recovery(; output_path=EXTENDED_CAP_PATH)
    kernel = TrixiParticles.WendlandC2Kernel{3}()
    profile_data = kernel_plane_profile(kernel)
    coarea_normalization = inv(profile_data.half_profile_integral^2)
    variants = ("analytic_both_control", "compatible_indicator",
                "compatible_geometry_wall", "young_color_boundary")
    rows = NamedTuple[]
    for resolution in RESOLUTIONS, target in TARGET_ANGLES
        line_lengths = Dict(variant => Float64[] for variant in variants)
        angles = Dict(variant => Float64[] for variant in variants)
        valid_particles = Dict(variant => Int[] for variant in variants)
        particle_counts = Int[]
        ghost_minimum = Float64[]
        ghost_maximum = Float64[]
        analytic_line_length = NaN
        for lattice_phase in CAP_PHASES
            result = quiet_css_sessile_drop(target, 0.0, nothing;
                                            target_particle_count=resolution,
                                            mechanism=:contact_line_force,
                                            initial_contact_angle=target,
                                            boundary_contact_threshold=0.1,
                                            damping_coefficient=4000.0,
                                            lattice_phase)
            components = raw_normal_components(result)
            setup = spherical_cap_initial_condition(target;
                                                    target_particle_count=resolution,
                                                    lattice_phase)
            analytic = analytic_cap_gradients(result, components, setup, profile_data)
            compatible = compatible_indicator_gradients(result, components)
            geometry_wall = geometry_wall_gradient(result, components, profile_data)
            young = young_boundary_gradients(result, components, target, profile_data)
            candidate_gradients = (("analytic_both_control",
                                    analytic.interface_gradient,
                                    analytic.wall_gradient),
                                   ("compatible_indicator",
                                    compatible.interface_gradient,
                                    compatible.wall_gradient),
                                   ("compatible_geometry_wall",
                                    compatible.interface_gradient,
                                    geometry_wall),
                                   ("young_color_boundary",
                                    young.interface_gradient,
                                    young.wall_gradient))
            for (variant, interface_gradient, wall_gradient) in candidate_gradients
                metrics = gradient_measure_metrics(components, interface_gradient,
                                                   wall_gradient,
                                                   coarea_normalization)
                push!(line_lengths[variant], metrics.line_length)
                push!(angles[variant], metrics.angle)
                push!(valid_particles[variant], metrics.valid_particles)
            end
            push!(particle_counts, result.particle_count)
            push!(ghost_minimum, young.ghost_fraction_min)
            push!(ghost_maximum, young.ghost_fraction_max)
            analytic_line_length = 2pi * setup.cap_radius
        end
        for variant in variants
            summary = phase_measure_summary(line_lengths[variant], analytic_line_length)
            angle = sum(line_lengths[variant] .* angles[variant]) /
                    sum(line_lengths[variant])
            angle_standard_error = std(angles[variant]) / sqrt(length(CAP_PHASES))
            push!(rows,
                  (; variant, target, requested_particles=resolution,
                   particle_count_min=minimum(particle_counts),
                   particle_count_max=maximum(particle_counts),
                   phase_count=length(CAP_PHASES), coarea_normalization,
                   line_length_phases=summary.phase_values,
                   line_length=summary.mean_value, analytic_line_length,
                   signed_error=summary.signed_error,
                   line_length_error=summary.error,
                   phase_standard_error=summary.relative_standard_error,
                   angle_phases=join((@sprintf("%.17g", value)
                                      for value in angles[variant]), ";"),
                   angle, angle_error=abs(angle - target), angle_standard_error,
                   valid_particles=round(Int, mean(valid_particles[variant])),
                   ghost_fraction_min=variant == "young_color_boundary" ?
                                      minimum(ghost_minimum) : NaN,
                   ghost_fraction_max=variant == "young_color_boundary" ?
                                      maximum(ghost_maximum) : NaN))
        end
    end
    data = DataFrame(rows)
    apply_phase_measure_gates!(data)
    data.angle_middle_pass = falses(nrow(data))
    data.angle_endpoint_pass = falses(nrow(data))
    for group in groupby(data, [:variant, :target]; sort=true)
        order = sortperm(group.requested_particles)
        parent_rows = parentindices(group)[1]
        coarse_error = group.angle_error[order[1]]
        fine_error = group.angle_error[order[3]]
        coarse_se = group.angle_standard_error[order[1]]
        fine_se = group.angle_standard_error[order[3]]
        data[parent_rows, :angle_middle_pass] .= group.angle_error[order[2]] <= 5
        data[parent_rows,
             :angle_endpoint_pass] .= fine_error <= 5 &&
                                      fine_error <=
                                      coarse_error +
                                      2hypot(coarse_se, fine_se)
    end
    protocol = CSV.read(MEASURE_PROTOCOL_PATH, DataFrame)
    protocol_valid = all(protocol.protocol_pass)
    data.protocol_valid = fill(protocol_valid, nrow(data))
    data.measure_eligible = protocol_valid .& data.middle_pass .& data.endpoint_pass
    data.static_eligible = data.measure_eligible .& data.angle_middle_pass .&
                           data.angle_endpoint_pass
    sort!(data, [:variant, :requested_particles, :target])
    CSV.write(output_path, data)
    for variant in variants
        middle = data[(data.variant .== variant) .& (data.requested_particles .== RESOLUTIONS[2]),
                      :]
        @printf("%-28s measure %d/5 middle, %d/5 endpoint; angle %d/5 middle, %d/5 endpoint\n",
                variant, count(middle.middle_pass), count(middle.endpoint_pass),
                count(middle.angle_middle_pass), count(middle.angle_endpoint_pass))
    end
    println("Wrote extended cap recovery evidence to ", output_path)
    return data
end

function coarea_wall_energy_shape_acceleration(result, target, components,
                                               interface_gradient, wall_gradient,
                                               coarea_normalization)
    semi = result.solution.prob.p.semi
    system = semi.systems[1]
    v_ode, _ = result.solution.prob.u0.x
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    acceleration = zeros(eltype(system), ndims(system), TrixiParticles.nparticles(system))
    sigma = system.surface_tension.surface_tension_coefficient
    for particle in TrixiParticles.eachparticle(system)
        interface = SVector{3}(interface_gradient[:, particle])
        wall = SVector{3}(wall_gradient[:, particle])
        wall_norm = norm(wall)
        wall_norm > eps(wall_norm) || continue
        wall /= wall_norm
        tangent = interface - dot(interface, wall) * wall
        tangent_norm = norm(tangent)
        tangent_norm > eps(tangent_norm) || continue
        line_delta = coarea_normalization *
                     norm(cross(interface, SVector{3}(wall_gradient[:, particle])))
        density = TrixiParticles.current_density(v, system, particle)
        acceleration[:,
                     particle] .= -sigma * cosd(target) * line_delta /
                                  density * tangent / tangent_norm
    end
    return cap_shape_acceleration(acceleration, components.coordinates)
end

function young_boundary_shape_acceleration(result, gradients)
    system = result.solution.prob.p.semi.systems[1]
    system.cache.surface_normal .= gradients.interface_gradient
    system.cache.boundary_normal .= 0
    for particle in TrixiParticles.eachparticle(system)
        wall = SVector{3}(gradients.wall_gradient[:, particle])
        wall_norm = norm(wall)
        wall_norm > eps(wall_norm) || continue
        system.cache.boundary_normal[:, particle] .= wall / wall_norm
    end
    method = TrixiParticles.surface_normal_method(system)
    TrixiParticles.remove_invalid_normals!(system, system.surface_tension, method)
    system.cache.contact_line_delta .= 0
    system.cache.contact_line_delta_prime .= 0
    return shape_acceleration_from_caches(result)
end

function run_extended_force_sign(; area_path=CORRECTED_WETTED_AREA_PATH,
                                 cap_path=EXTENDED_CAP_PATH,
                                 output_path=EXTENDED_FORCE_SIGN_PATH)
    area = CSV.read(area_path, DataFrame)
    cap = CSV.read(cap_path, DataFrame)
    area_middle = area[area.requested_particles .== RESOLUTIONS[2], :]
    area_eligible = all(area_middle.measure_eligible)
    compatible_middle = cap[(cap.variant .== "compatible_geometry_wall") .& (cap.requested_particles .== RESOLUTIONS[2]),
                            :]
    compatible_eligible = all(compatible_middle.measure_eligible)
    young_middle = cap[(cap.variant .== "young_color_boundary") .& (cap.requested_particles .== RESOLUTIONS[2]),
                       :]
    young_eligible = all(young_middle.static_eligible)

    kernel = TrixiParticles.WendlandC2Kernel{3}()
    profile_data = kernel_plane_profile(kernel)
    coarea_normalization = inv(profile_data.half_profile_integral^2)
    offset_data = kernel_lattice_color_offsets(kernel, 64)
    edge_data = Dict(target => canonical_wetted_edge_data(profile_data, offset_data,
                                                          target)
                     for target in TARGET_ANGLES)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0), (90.0, 120.0), (120.0, 90.0))
    rows = NamedTuple[]
    for (target, initial_angle) in angle_pairs
        control = quiet_css_sessile_drop(target, 0.0, nothing;
                                         target_particle_count=1500, mechanism=:none,
                                         initial_contact_angle=initial_angle,
                                         boundary_contact_threshold=0.1,
                                         damping_coefficient=4000.0)
        scaffold = quiet_css_sessile_drop(target, 0.0, nothing;
                                          target_particle_count=1500,
                                          mechanism=:contact_line_force,
                                          initial_contact_angle=initial_angle,
                                          boundary_contact_threshold=0.1,
                                          damping_coefficient=4000.0)
        components = raw_normal_components(scaffold)
        compatible = compatible_indicator_gradients(scaffold, components)
        geometry_wall = geometry_wall_gradient(scaffold, components, profile_data)
        young = young_boundary_gradients(scaffold, components, target, profile_data)
        expected_direction = sign(target - initial_angle)

        corrected_area_acceleration = corrected_wetted_area_shape_acceleration(control,
                                                                               target,
                                                                               profile_data,
                                                                               edge_data[target])
        compatible_acceleration = coarea_wall_energy_shape_acceleration(scaffold, target,
                                                                        components,
                                                                        compatible.interface_gradient,
                                                                        geometry_wall,
                                                                        coarea_normalization)
        young_acceleration = young_boundary_shape_acceleration(scaffold, young)
        candidates = (("corrected_wetted_area",
                       control.shape_acceleration + corrected_area_acceleration,
                       corrected_area_acceleration, area_eligible, true,
                       target == 90 ? iszero(corrected_area_acceleration) : missing),
                      ("compatible_geometry_wall",
                       control.shape_acceleration + compatible_acceleration,
                       compatible_acceleration, compatible_eligible, true,
                       target == 90 ? iszero(compatible_acceleration) : missing),
                      ("young_color_boundary", young_acceleration,
                       young_acceleration - control.shape_acceleration,
                       all(young_middle.measure_eligible), young_eligible, missing))
        for (variant, acceleration, contact_acceleration, measure_eligible,
             static_eligible, zero_at_90) in candidates

            total_sign_pass = expected_direction * acceleration > 0
            contact_sign_pass = expected_direction * contact_acceleration > 0
            push!(rows,
                  (; kind="force_sign", variant, target, initial_angle,
                   requested_particles=1500, particle_count=control.particle_count,
                   control_acceleration=control.shape_acceleration,
                   contact_shape_acceleration=contact_acceleration, acceleration,
                   expected_direction, total_sign_pass, contact_sign_pass,
                   wall_zero_at_90=zero_at_90, measure_eligible, static_eligible,
                   sign_pass=total_sign_pass))
        end
    end
    data = DataFrame(rows)
    sort!(data, [:variant, :target, :initial_angle])
    CSV.write(output_path, data)
    for variant in unique(data.variant)
        variant_rows = data[data.variant .== variant, :]
        println(variant, " fixed total/contact signs: ",
                count(variant_rows.total_sign_pass), "/", nrow(variant_rows), " / ",
                count(variant_rows.contact_sign_pass), "/", nrow(variant_rows),
                "; measure eligible: ", all(variant_rows.measure_eligible),
                "; static eligible: ", all(variant_rows.static_eligible))
    end
    println("Wrote extended fixed-particle signs to ", output_path)
    return data
end

@inline smoothstep01(value) = value^2 * (3 - 2value)
@inline smoothstep01_derivative(value) = 6value * (1 - value)

function kernel_lattice_color_offsets(kernel, cells_per_h)
    lattice_spacing = 1 / cells_per_h
    support = TrixiParticles.compact_support(kernel, 1.0)
    search_radius = ceil(Int, support / lattice_spacing)
    offsets = NamedTuple[]
    for z_offset in (-search_radius):search_radius,
        x_offset in (-search_radius):search_radius
        planar_distance2 = lattice_spacing^2 * (x_offset^2 + z_offset^2)
        planar_distance2 < support^2 || continue
        kernel_value = 0.0
        for tangent_offset in (-search_radius):search_radius
            distance = lattice_spacing *
                       sqrt(x_offset^2 + tangent_offset^2 + z_offset^2)
            distance < support || continue
            kernel_value += lattice_spacing *
                            TrixiParticles.kernel(kernel, distance, 1.0)
        end
        push!(offsets, (; x_offset, z_offset, kernel_value))
    end
    return (; offsets, lattice_spacing, support)
end

function canonical_wetted_edge_data(profile_data, offset_data, angle;
                                    production_cells_per_h=1.4)
    contact_sine = sind(angle)
    abs(contact_sine) > sqrt(eps()) ||
        return (; normalized_shift=0.0, lattice_reference=0.0,
                continuum_reference=0.0)
    contact_cotangent = cosd(angle) / contact_sine
    boundary_distance = inv(2production_cells_per_h)
    # Each reduced kernel sample enters the wetted wedge at one horizontal threshold, so the
    # smoothed edge profile is a cumulative sum rather than a sampled convolution.
    thresholds = Float64[]
    weights = Float64[]
    for offset in offset_data.offsets
        source_z = -boundary_distance -
                   offset.z_offset * offset_data.lattice_spacing
        source_z > 0 || continue
        push!(thresholds,
              offset.x_offset * offset_data.lattice_spacing +
              contact_cotangent * source_z)
        push!(weights,
              offset_data.lattice_spacing^2 * offset.kernel_value)
    end
    order = sortperm(thresholds)
    thresholds = thresholds[order]
    weights = weights[order]
    lattice_reference = sum(weights)
    breaks = sort!(unique!([thresholds; 0.0]))
    cumulative = 0.0
    event = 1
    normalized_shift = 0.0
    for interval in 1:(length(breaks) - 1)
        left = breaks[interval]
        right = breaks[interval + 1]
        while event <= length(thresholds) && thresholds[event] <= left
            cumulative += weights[event]
            event += 1
        end
        fraction = clamp(cumulative / lattice_reference, 0, 1)
        step = (left + right) / 2 > 0 ? 1.0 : 0.0
        normalized_shift += (right - left) * (smoothstep01(fraction) - step)
    end
    continuum_reference = halfspace_color_value(profile_data, boundary_distance)
    return (; normalized_shift, lattice_reference, continuum_reference)
end

function canonical_flooded_reference(result, profile_data)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    boundary_coordinates = TrixiParticles.current_coordinates(u_boundary, boundary_system)
    particle_spacing = system.cache.reference_particle_spacing
    smoothing_length = TrixiParticles.initial_smoothing_length(system)
    exposed_height = maximum(boundary_coordinates[3, :])
    wall_distance = -exposed_height
    particle = first(TrixiParticles.eachparticle(system))
    particle_volume = TrixiParticles.hydrodynamic_mass(system, particle) /
                      TrixiParticles.current_density(v, system, particle)
    volume_scale = particle_volume / particle_spacing^3
    reference = volume_scale *
                halfspace_color_value(profile_data, wall_distance / smoothing_length)
    return (; reference, volume_scale, wall_distance, smoothing_length)
end

function canonical_wetted_area_data(result, target, profile_data, edge_data)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    boundary_coordinates = Array(TrixiParticles.current_coordinates(u_boundary,
                                                                    boundary_system))
    colorfield = boundary_system.boundary_model.cache.colorfield
    reference_data = canonical_flooded_reference(result, profile_data)
    particle_spacing = system.cache.reference_particle_spacing
    exposed_height = maximum(boundary_coordinates[3, :])
    exposed = isapprox.(boundary_coordinates[3, :], exposed_height;
                        atol=10eps(abs(exposed_height) + particle_spacing))
    color_fraction = clamp.(colorfield ./ reference_data.reference, 0, 1)
    particle_area = particle_spacing^2
    raw_area = particle_area * sum(smoothstep01, color_fraction[exposed])
    raw_radius = sqrt(raw_area / pi)
    edge_shift = edge_data.normalized_shift * reference_data.smoothing_length
    # Remove the canonical planar edge displacement from the disk's effective radius.
    corrected_radius = max(raw_radius - edge_shift, 0.0)
    corrected_area = pi * corrected_radius^2
    area_derivative = raw_radius > eps(raw_radius) ? corrected_radius / raw_radius : 0.0
    return (; raw_area, corrected_area, raw_radius, corrected_radius,
            edge_shift, area_derivative, particle_area, exposed, color_fraction,
            colorfield_reference=reference_data.reference,
            observed_maximum=maximum(colorfield), reference_data.volume_scale,
            reference_data.wall_distance, reference_data.smoothing_length,
            boundary_coordinates, target)
end

function corrected_wetted_area_shape_acceleration(result, target, profile_data, edge_data)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)
    boundary_coordinates = TrixiParticles.current_coordinates(u_boundary, boundary_system)
    area_data = canonical_wetted_area_data(result, target, profile_data, edge_data)
    acceleration = zeros(eltype(system), ndims(system), TrixiParticles.nparticles(system))
    coefficient = system.surface_tension.surface_tension_coefficient * cosd(target) *
                  area_data.particle_area * area_data.area_derivative /
                  area_data.colorfield_reference

    TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                          boundary_coordinates,
                                          semi) do particle, neighbor,
                                                   pos_diff, distance
        area_data.exposed[neighbor] || return
        fraction = area_data.color_fraction[neighbor]
        derivative = smoothstep01_derivative(fraction)
        derivative > eps(derivative) || return
        density = TrixiParticles.current_density(v, system, particle)
        gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                        particle)
        acceleration[:, particle] .+= coefficient / density * derivative * gradient
    end
    return cap_shape_acceleration(acceleration, Array(coordinates))
end

function run_corrected_wetted_area(; output_path=CORRECTED_WETTED_AREA_PATH)
    kernel = TrixiParticles.WendlandC2Kernel{3}()
    profile_data = kernel_plane_profile(kernel)
    offset_data = kernel_lattice_color_offsets(kernel, 64)
    edge_data = Dict(target => canonical_wetted_edge_data(profile_data, offset_data,
                                                          target)
                     for target in TARGET_ANGLES)
    rows = NamedTuple[]
    for resolution in RESOLUTIONS, target in TARGET_ANGLES
        raw_areas = Float64[]
        corrected_areas = Float64[]
        references = Float64[]
        observed_maxima = Float64[]
        particle_counts = Int[]
        exposed_counts = Int[]
        edge_shifts = Float64[]
        area_derivatives = Float64[]
        analytic_area = NaN
        for lattice_phase in CAP_PHASES
            result = quiet_css_sessile_drop(target, 0.0, nothing;
                                            target_particle_count=resolution,
                                            mechanism=:none,
                                            initial_contact_angle=target,
                                            boundary_contact_threshold=0.1,
                                            damping_coefficient=4000.0,
                                            lattice_phase)
            area_data = canonical_wetted_area_data(result, target, profile_data,
                                                   edge_data[target])
            setup = spherical_cap_initial_condition(target;
                                                    target_particle_count=resolution,
                                                    lattice_phase)
            analytic_area = pi * setup.cap_radius^2
            push!(raw_areas, area_data.raw_area)
            push!(corrected_areas, area_data.corrected_area)
            push!(references, area_data.colorfield_reference)
            push!(observed_maxima, area_data.observed_maximum)
            push!(particle_counts, result.particle_count)
            push!(exposed_counts, count(area_data.exposed))
            push!(edge_shifts, area_data.edge_shift)
            push!(area_derivatives, area_data.area_derivative)
        end
        raw = phase_measure_summary(raw_areas, analytic_area)
        corrected = phase_measure_summary(corrected_areas, analytic_area)
        push!(rows,
              (; variant="corrected_wetted_area", target,
               requested_particles=resolution,
               particle_count_min=minimum(particle_counts),
               particle_count_max=maximum(particle_counts),
               exposed_particles=round(Int, mean(exposed_counts)),
               phase_count=length(CAP_PHASES),
               colorfield_reference=mean(references),
               observed_maximum=mean(observed_maxima),
               reference_ratio=mean(references ./ observed_maxima),
               normalized_edge_shift=edge_data[target].normalized_shift,
               edge_shift=mean(edge_shifts),
               area_derivative=mean(area_derivatives),
               raw_area_phases=raw.phase_values,
               raw_area=raw.mean_value,
               raw_area_error=raw.error,
               raw_phase_standard_error=raw.relative_standard_error,
               corrected_area_phases=corrected.phase_values,
               corrected_area=corrected.mean_value, analytic_area,
               corrected_area_error=corrected.error,
               phase_standard_error=corrected.relative_standard_error))
    end
    data = DataFrame(rows)
    apply_phase_measure_gates!(data; error_column=:corrected_area_error)
    apply_phase_measure_gates!(data; error_column=:raw_area_error,
                               standard_error_column=:raw_phase_standard_error,
                               middle_column=:raw_middle_pass,
                               endpoint_column=:raw_endpoint_pass)
    protocol = CSV.read(MEASURE_PROTOCOL_PATH, DataFrame)
    protocol_valid = all(protocol.protocol_pass)
    data.protocol_valid = fill(protocol_valid, nrow(data))
    data.measure_eligible = protocol_valid .& data.middle_pass .& data.endpoint_pass
    sort!(data, [:requested_particles, :target])
    CSV.write(output_path, data)
    middle = data[data.requested_particles .== RESOLUTIONS[2], :]
    @printf("corrected wetted area: middle %d/5, endpoints %d/5, max middle error %.3f; protocol %s\n",
            count(middle.middle_pass), count(middle.endpoint_pass),
            maximum(middle.corrected_area_error), protocol_valid ? "valid" : "invalid")
    println("Wrote corrected wetted-area evidence to ", output_path)
    return data
end

function wetted_area_data(result)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    boundary_coordinates = Array(TrixiParticles.current_coordinates(u_boundary,
                                                                    boundary_system))
    colorfield = boundary_system.boundary_model.cache.colorfield
    colorfield_reference = maximum(colorfield)
    particle_spacing = system.cache.reference_particle_spacing
    exposed_height = maximum(boundary_coordinates[3, :])
    exposed = isapprox.(boundary_coordinates[3, :], exposed_height;
                        atol=10eps(abs(exposed_height) + particle_spacing))
    color_fraction = clamp.(colorfield ./ colorfield_reference, 0, 1)
    particle_area = particle_spacing^2
    wetted_area = particle_area * sum(smoothstep01, color_fraction[exposed])
    return (; wetted_area, particle_area, exposed, color_fraction,
            colorfield_reference, boundary_coordinates)
end

function wetted_area_shape_acceleration(result, target)
    semi = result.solution.prob.p.semi
    system, boundary_system = semi.systems
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = TrixiParticles.current_coordinates(u, system)
    boundary_coordinates = TrixiParticles.current_coordinates(u_boundary, boundary_system)
    area_data = wetted_area_data(result)
    acceleration = zeros(eltype(system), ndims(system), TrixiParticles.nparticles(system))
    coefficient = system.surface_tension.surface_tension_coefficient * cosd(target) *
                  area_data.particle_area / area_data.colorfield_reference

    TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                          boundary_coordinates,
                                          semi) do particle, neighbor,
                                                   pos_diff, distance
        area_data.exposed[neighbor] || return
        fraction = area_data.color_fraction[neighbor]
        derivative = smoothstep01_derivative(fraction)
        derivative > eps(derivative) || return
        density = TrixiParticles.current_density(v, system, particle)
        gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                        particle)
        acceleration[:, particle] .+= coefficient / density * derivative * gradient
    end
    return cap_shape_acceleration(acceleration, Array(coordinates))
end

function run_wetted_area_measure(; output_path=WETTED_AREA_PATH)
    rows = NamedTuple[]
    for resolution in RESOLUTIONS, target in TARGET_ANGLES
        result = css_sessile_drop(target, 0.0, nothing;
                                  target_particle_count=resolution,
                                  mechanism=:none,
                                  initial_contact_angle=target,
                                  boundary_contact_threshold=0.1,
                                  damping_coefficient=4000.0)
        area_data = wetted_area_data(result)
        setup = spherical_cap_initial_condition(target;
                                                target_particle_count=resolution)
        analytic_area = pi * setup.cap_radius^2
        push!(rows,
              (; target, requested_particles=resolution,
               particle_count=result.particle_count,
               exposed_particles=count(area_data.exposed),
               colorfield_reference=area_data.colorfield_reference,
               wetted_area=area_data.wetted_area, analytic_area,
               area_error=abs(area_data.wetted_area / analytic_area - 1)))
    end
    data = DataFrame(rows)
    data.middle_pass = data.area_error .<= 0.2
    data.endpoint_decreasing = falses(nrow(data))
    for group in groupby(data, :target)
        coarse = only(group[group.requested_particles .== first(RESOLUTIONS), :area_error])
        fine = only(group[group.requested_particles .== last(RESOLUTIONS), :area_error])
        data[parentindices(group)[1],
             :endpoint_decreasing] .= fine < coarse || fine <= 1.0e-12
    end
    sort!(data, [:requested_particles, :target])
    CSV.write(output_path, data)
    middle = data[data.requested_particles .== RESOLUTIONS[2], :]
    @printf("wetted area: middle %d/5, endpoints %d/5, max middle error %.3f\n",
            count(middle.middle_pass), count(middle.endpoint_decreasing),
            maximum(middle.area_error))
    println("Wrote wetted-area measure to ", output_path)
    return data
end

function run_wetted_area_force_sign(; measure_path=WETTED_AREA_PATH,
                                    output_path=WETTED_AREA_FORCE_SIGN_PATH)
    measure = isfile(measure_path) ? CSV.read(measure_path, DataFrame) :
              run_wetted_area_measure(; output_path=measure_path)
    measure_middle = measure[measure.requested_particles .== RESOLUTIONS[2], :]
    measure_eligible = all(measure_middle.middle_pass .&
                           measure_middle.endpoint_decreasing)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0), (90.0, 120.0), (120.0, 90.0))
    rows = NamedTuple[]
    for (target, initial_angle) in angle_pairs
        control = css_sessile_drop(target, 0.0, nothing;
                                   target_particle_count=1500, mechanism=:none,
                                   initial_contact_angle=initial_angle,
                                   boundary_contact_threshold=0.1,
                                   damping_coefficient=4000.0)
        wall_acceleration = wetted_area_shape_acceleration(control, target)
        acceleration = control.shape_acceleration + wall_acceleration
        expected_direction = sign(target - initial_angle)
        push!(rows,
              (; kind="force_sign", variant="wetted_area", target, initial_angle,
               requested_particles=1500, particle_count=control.particle_count,
               control_acceleration=control.shape_acceleration, wall_acceleration,
               acceleration, expected_direction,
               wall_zero_at_90=target == 90 ? iszero(wall_acceleration) : missing,
               sign_pass=expected_direction * acceleration > 0,
               measure_eligible))
    end
    data = DataFrame(rows)
    CSV.write(output_path, data)
    println("wetted-area total fixed-particle signs: ", count(data.sign_pass), "/",
            nrow(data), "; measure eligible: ", measure_eligible)
    println("Wrote wetted-area force signs to ", output_path)
    return data
end

function run_recovery_comparison(; cap_path=CAP_TRANSFER_PATH,
                                 area_path=WETTED_AREA_PATH,
                                 area_sign_path=WETTED_AREA_FORCE_SIGN_PATH,
                                 output_path=RECOVERY_COMPARISON_PATH)
    cap = CSV.read(cap_path, DataFrame)
    area = CSV.read(area_path, DataFrame)
    area_sign = CSV.read(area_sign_path, DataFrame)
    rows = NamedTuple[]
    cap_methods = (("R6-D production discrete", "attribution", "production_discrete"),
                   ("R6-D analytic wall", "attribution", "analytic_wall"),
                   ("R6-D analytic interface", "attribution", "analytic_interface"),
                   ("R6-D analytic both", "attribution", "analytic_both"),
                   ("R6-C compatible indicator", "candidate", "compatible_indicator"),
                   ("support-moment diagnostic", "attribution", "support_moment"))
    for (method, role, variant) in cap_methods
        middle = cap[(cap.variant .== variant) .& (cap.requested_particles .== RESOLUTIONS[2]),
                     :]
        middle_passes = count(middle.middle_pass)
        endpoint_passes = count(middle.endpoint_decreasing)
        measure_gate = middle_passes == length(TARGET_ANGLES) &&
                       endpoint_passes == length(TARGET_ANGLES)
        push!(rows,
              (; method, role, measure="contact_line", middle_passes,
               endpoint_passes, measure_gate, sign_cases=0, sign_passes=0,
               zero_at_90=missing, eligible=false,
               max_middle_error=maximum(middle.line_length_error)))
    end

    area_middle = area[area.requested_particles .== RESOLUTIONS[2], :]
    middle_passes = count(area_middle.middle_pass)
    endpoint_passes = count(area_middle.endpoint_decreasing)
    measure_gate = middle_passes == length(TARGET_ANGLES) &&
                   endpoint_passes == length(TARGET_ANGLES)
    zero_values = collect(skipmissing(area_sign.wall_zero_at_90))
    zero_at_90 = length(zero_values) == 2 && all(zero_values)
    sign_passes = count(area_sign.sign_pass)
    push!(rows,
          (; method="R6-W wetted-area energy", role="candidate", measure="wetted_area",
           middle_passes, endpoint_passes, measure_gate,
           sign_cases=nrow(area_sign), sign_passes, zero_at_90,
           eligible=measure_gate && sign_passes == nrow(area_sign),
           max_middle_error=maximum(area_middle.area_error)))

    data = DataFrame(rows)
    CSV.write(output_path, data)
    println(data)
    println("Wrote R6 recovery comparison to ", output_path)
    return data
end

function run_extended_recovery_comparison(; cap_path=EXTENDED_CAP_PATH,
                                          area_path=CORRECTED_WETTED_AREA_PATH,
                                          sign_path=EXTENDED_FORCE_SIGN_PATH,
                                          output_path=EXTENDED_COMPARISON_PATH)
    cap = CSV.read(cap_path, DataFrame)
    area = CSV.read(area_path, DataFrame)
    signs = CSV.read(sign_path, DataFrame)
    rows = NamedTuple[]

    cap_methods = (("exact-profile protocol control", "control",
                    "analytic_both_control"),
                   ("R7-CG compatible + geometry wall", "candidate",
                    "compatible_geometry_wall"),
                   ("R7-Y Young color boundary", "candidate",
                    "young_color_boundary"))
    for (method, role, variant) in cap_methods
        middle = cap[(cap.variant .== variant) .& (cap.requested_particles .== RESOLUTIONS[2]),
                     :]
        sign_rows = signs[signs.variant .== variant, :]
        middle_passes = count(middle.middle_pass)
        endpoint_passes = count(middle.endpoint_pass)
        measure_gate = middle_passes == length(TARGET_ANGLES) &&
                       endpoint_passes == length(TARGET_ANGLES)
        angle_middle_passes = count(middle.angle_middle_pass)
        angle_endpoint_passes = count(middle.angle_endpoint_pass)
        angle_gate = angle_middle_passes == length(TARGET_ANGLES) &&
                     angle_endpoint_passes == length(TARGET_ANGLES)
        sign_cases = nrow(sign_rows)
        sign_passes = count(sign_rows.sign_pass)
        zero_values = collect(skipmissing(sign_rows.wall_zero_at_90))
        zero_at_90 = isempty(zero_values) ? missing : all(zero_values)
        eligible_for_dynamics = role == "candidate" && measure_gate && angle_gate &&
                                sign_cases == 4 && sign_passes == 4
        push!(rows,
              (; method, role, measure="contact_line", middle_passes,
               endpoint_passes, measure_gate, angle_middle_passes,
               angle_endpoint_passes, angle_gate, sign_cases, sign_passes,
               zero_at_90, eligible_for_dynamics,
               max_middle_error=maximum(middle.line_length_error)))
    end

    middle = area[area.requested_particles .== RESOLUTIONS[2], :]
    sign_rows = signs[signs.variant .== "corrected_wetted_area", :]
    middle_passes = count(middle.middle_pass)
    endpoint_passes = count(middle.endpoint_pass)
    measure_gate = middle_passes == length(TARGET_ANGLES) &&
                   endpoint_passes == length(TARGET_ANGLES)
    sign_cases = nrow(sign_rows)
    sign_passes = count(sign_rows.sign_pass)
    zero_values = collect(skipmissing(sign_rows.wall_zero_at_90))
    zero_at_90 = length(zero_values) == 2 && all(zero_values)
    push!(rows,
          (; method="R7-W corrected wetted-area energy", role="candidate",
           measure="wetted_area", middle_passes, endpoint_passes, measure_gate,
           angle_middle_passes=0, angle_endpoint_passes=0, angle_gate=true,
           sign_cases, sign_passes, zero_at_90,
           eligible_for_dynamics=measure_gate && sign_cases == 4 &&
                                 sign_passes == 4 && zero_at_90,
           max_middle_error=maximum(middle.corrected_area_error)))

    data = DataFrame(rows)
    CSV.write(output_path, data)
    println(data)
    println("Wrote extended R7 recovery comparison to ", output_path)
    return data
end

function cap_transfer_row(variant, target, resolution, setup, components,
                          interface_gradient, wall_gradient, coarea_normalization;
                          particle_scale=ones(length(components.volume)))
    measure = 0.0
    for particle in eachindex(components.volume)
        measure += components.volume[particle] * particle_scale[particle] *
                   norm(cross(interface_gradient[:, particle],
                              wall_gradient[:, particle]))
    end
    line_length = coarea_normalization * measure
    analytic_line_length = 2pi * setup.cap_radius
    return (; variant=String(variant), target, requested_particles=resolution,
            particle_count=length(components.volume), measure, coarea_normalization,
            line_length, analytic_line_length,
            line_length_error=abs(line_length / analytic_line_length - 1))
end

function run_cap_transfer_diagnostics(; output_path=CAP_TRANSFER_PATH)
    kernel = TrixiParticles.WendlandC2Kernel{3}()
    profile_data = kernel_plane_profile(kernel)
    coarea_normalization = inv(profile_data.half_profile_integral^2)
    rows = NamedTuple[]
    for resolution in RESOLUTIONS, target in TARGET_ANGLES
        result = css_sessile_drop(target, 0.0, nothing;
                                  target_particle_count=resolution,
                                  mechanism=:contact_line_force,
                                  initial_contact_angle=target,
                                  boundary_contact_threshold=0.1,
                                  damping_coefficient=4000.0)
        components = raw_normal_components(result)
        setup = spherical_cap_initial_condition(target;
                                                target_particle_count=resolution)
        analytic = analytic_cap_gradients(result, components, setup, profile_data)
        compatible = compatible_indicator_gradients(result, components)
        push!(rows,
              cap_transfer_row(:production_discrete, target, resolution, setup,
                               components, components.fluid_normal,
                               components.wall_normal, coarea_normalization))
        push!(rows,
              cap_transfer_row(:analytic_wall, target, resolution, setup,
                               components, components.fluid_normal,
                               analytic.wall_gradient, coarea_normalization))
        push!(rows,
              cap_transfer_row(:analytic_interface, target, resolution, setup,
                               components, analytic.interface_gradient,
                               components.wall_normal, coarea_normalization))
        push!(rows,
              cap_transfer_row(:analytic_both, target, resolution, setup,
                               components, analytic.interface_gradient,
                               analytic.wall_gradient, coarea_normalization))
        push!(rows,
              cap_transfer_row(:compatible_indicator, target, resolution, setup,
                               components, compatible.interface_gradient,
                               compatible.wall_gradient, coarea_normalization))
        inverse_support = map(components.support_moment) do moment
            moment > sqrt(eps(moment)) ? inv(moment) : 0.0
        end
        push!(rows,
              cap_transfer_row(:support_moment, target, resolution, setup,
                               components, components.fluid_normal,
                               components.wall_normal, coarea_normalization;
                               particle_scale=inverse_support))
    end

    data = DataFrame(rows)
    data.middle_pass = data.line_length_error .<= 0.2
    data.endpoint_decreasing = falses(nrow(data))
    for group in groupby(data, [:variant, :target])
        coarse = only(group[group.requested_particles .== first(RESOLUTIONS),
                            :line_length_error])
        fine = only(group[group.requested_particles .== last(RESOLUTIONS),
                          :line_length_error])
        parent_rows = parentindices(group)[1]
        data[parent_rows, :endpoint_decreasing] .= fine < coarse || fine <= 1.0e-12
    end
    sort!(data, [:variant, :requested_particles, :target])
    CSV.write(output_path, data)
    for variant in unique(data.variant)
        middle = data[(data.variant .== variant) .& (data.requested_particles .== RESOLUTIONS[2]),
                      :]
        @printf("%-22s middle %d/5, endpoints %d/5, max middle error %.3f\n",
                variant, count(middle.middle_pass),
                count(middle.endpoint_decreasing), maximum(middle.line_length_error))
    end
    println("Wrote cap-transfer diagnostics to ", output_path)
    return data
end

function normal_component_row(target, initial_angle, resolution, variant, candidate_normal,
                              components, local_angle)
    (; total_normal, wall_normal, line_delta, surface_delta, support_moment,
     volume) = components
    active_particles = findall(>(0), line_delta)
    valid_particles = filter(active_particles) do particle
        dot(wall_normal[:, particle], wall_normal[:, particle]) > eps() &&
            dot(candidate_normal[:, particle], candidate_normal[:, particle]) > eps()
    end
    total_weight = sum(line_delta[active_particles])
    valid_weight = sum(line_delta[valid_particles])
    angles = Float64[]
    weights = Float64[]
    for particle in valid_particles
        wall = wall_normal[:, particle]
        candidate = candidate_normal[:, particle]
        cosine = dot(wall, candidate) / sqrt(dot(wall, wall) * dot(candidate, candidate))
        push!(angles, acosd(clamp(cosine, -1, 1)))
        push!(weights, line_delta[particle])
    end

    weight_sum = sum(weights)
    angle_mean = weight_sum > 0 ? sum(weights .* angles) / weight_sum : NaN
    angle_median = weighted_quantile(angles, weights, 0.5)
    angle_p10 = weighted_quantile(angles, weights, 0.1)
    angle_p90 = weighted_quantile(angles, weights, 0.9)
    reference_cosine_error = cosd(target) - cosd(local_angle)
    wrong_sign_weight = if isapprox(reference_cosine_error, 0.0; atol=1.0e-12)
        NaN
    else
        wrong_weight = 0.0
        for (weight, angle) in zip(weights, angles)
            if reference_cosine_error * (cosd(target) - cosd(angle)) <= 0
                wrong_weight += weight
            end
        end
        weight_sum > 0 ? wrong_weight / weight_sum : NaN
    end
    setup = spherical_cap_initial_condition(initial_angle;
                                            target_particle_count=resolution)
    analytic_line_length = 2pi * setup.cap_radius
    discrete_line_length = sum(volume .* line_delta)
    cross_line_length = 0.0
    cross_line_length_raw = 0.0
    corrected_cross_line_length = 0.0
    for particle in eachindex(volume)
        total_norm = norm(total_normal[:, particle])
        total_norm > eps(total_norm) || continue
        cross_gradient = norm(cross(total_normal[:, particle], wall_normal[:, particle]))
        activity = surface_delta[particle] / (2total_norm)
        cross_line_length += volume[particle] * activity * cross_gradient
        cross_line_length_raw += volume[particle] * cross_gradient
        correction = support_moment[particle]
        if correction > sqrt(eps(correction))
            corrected_cross_line_length += volume[particle] * cross_gradient / correction
        end
    end

    return (; kind="normal", variant=String(variant), target, initial_angle,
            requested_particles=resolution, particle_count=length(volume), local_angle,
            angle_mean, angle_median, angle_p10, angle_p90,
            mean_error=abs(angle_mean - local_angle),
            median_error=abs(angle_median - local_angle), wrong_sign_weight,
            valid_weight_fraction=total_weight > 0 ? valid_weight / total_weight : 0.0,
            line_particles=length(active_particles),
            valid_particles=length(valid_particles),
            line_weight=total_weight, discrete_line_length, analytic_line_length,
            line_length_error=abs(discrete_line_length / analytic_line_length - 1),
            cross_line_length,
            cross_error_1x=abs(cross_line_length / analytic_line_length - 1),
            cross_error_2x=abs(2cross_line_length / analytic_line_length - 1),
            cross_error_4x=abs(4cross_line_length / analytic_line_length - 1),
            cross_line_length_raw,
            raw_cross_error_4x=abs(4cross_line_length_raw / analytic_line_length - 1),
            corrected_cross_line_length,
            corrected_cross_error_4x=abs(4corrected_cross_line_length /
                                         analytic_line_length - 1))
end

function run_decision_case(kind, target, initial_angle, mechanism;
                           resolution=1500, threshold=0.1,
                           damping=4000.0, final_time=0.01, repeat=1,
                           variant=:baseline)
    result = nothing
    total_runtime = @elapsed result = css_sessile_drop(target, final_time, nothing;
                                                       target_particle_count=resolution,
                                                       mechanism,
                                                       initial_contact_angle=initial_angle,
                                                       boundary_contact_threshold=threshold,
                                                       damping_coefficient=damping)
    initial_error = result.initial_circle.angle - target
    final_error = result.circle.angle - target
    error_reduction = abs(initial_error) > eps() ?
                      1 - abs(final_error) / abs(initial_error) : NaN
    cache = result.solution.prob.p.semi.systems[1].cache
    scalar_size = sizeof(eltype(result.solution.prob.p.semi.systems[1]))
    contact_scalars = mechanism == :none ? 0 : mechanism == :geometric ? 3 : 5

    return (; kind=String(kind), variant=String(variant), target, initial_angle,
            mechanism=String(mechanism), requested_particles=resolution,
            particle_count=result.particle_count, threshold, damping, final_time, repeat,
            initial_circle=result.initial_circle.angle,
            final_circle=result.circle.angle, initial_error, final_error, error_reduction,
            cap_angle=result.final.angle, normal_angle=result.measured_contact_angle,
            line_angle=result.line_contact_angle,
            shape_acceleration=result.shape_acceleration,
            boundary_shape_acceleration=result.boundary_shape_acceleration,
            below_wall=result.below_wall, density_min=result.density_range[1],
            density_max=result.density_range[2], rms_speed=result.rms_speed,
            settled=result.settled, accepted_steps=result.accepted_steps,
            rejected_steps=result.rejected_steps,
            rejected_fraction=result.rejected_fraction,
            minimum_dt=result.minimum_dt, dt_reference=result.dt_reference,
            eta_p01=result.eta_p01, eta_median=result.eta_median,
            eta_tail_head=result.eta_tail_head,
            solver_runtime=result.runtime, total_runtime,
            cache_bytes=Base.summarysize(cache),
            contact_cache_bytes=contact_scalars * result.particle_count * scalar_size)
end

function run_normal_diagnostics(; output_path=NORMAL_COMPONENTS_PATH)
    off_target = Dict(60.0 => (90.0,), 90.0 => (60.0, 120.0), 120.0 => (90.0,))
    rows = NamedTuple[]
    for resolution in RESOLUTIONS, initial_angle in TARGET_ANGLES
        result = css_sessile_drop(initial_angle, 0.0, nothing;
                                  target_particle_count=resolution,
                                  mechanism=:contact_line_force,
                                  initial_contact_angle=initial_angle,
                                  boundary_contact_threshold=0.1,
                                  damping_coefficient=4000.0)
        components = raw_normal_components(result)
        targets = Float64[initial_angle]
        if resolution == 1500
            for (target, initial_angles) in off_target
                initial_angle in initial_angles && push!(targets, target)
            end
        end
        for target in unique(targets)
            push!(rows,
                  normal_component_row(target, initial_angle, resolution,
                                       :baseline_total, components.total_normal,
                                       components, result.initial_circle.angle))
            push!(rows,
                  normal_component_row(target, initial_angle, resolution,
                                       :fluid_only, components.fluid_normal,
                                       components, result.initial_circle.angle))
        end
    end
    data = DataFrame(rows)
    sort!(data, [:variant, :requested_particles, :initial_angle, :target])
    CSV.write(output_path, data)

    for variant in ("baseline_total", "fluid_only")
        static_middle = data[(data.variant .== variant) .& (data.requested_particles .== 1500) .& (data.target .== data.initial_angle),
                             :]
        off_target_rows = data[(data.variant .== variant) .& (data.target .!= data.initial_angle),
                               :]
        @printf("%-15s static max mean error %.3f deg, off-target max wrong weight %.3f, min coverage %.3f\n",
                variant, maximum(static_middle.mean_error),
                maximum(off_target_rows.wrong_sign_weight),
                minimum(data[data.variant .== variant, :valid_weight_fraction]))
    end
    println("Wrote raw normal diagnostics to ", output_path)
    return data
end

function run_force_sign_cases(; variant=:baseline, output_path=FORCE_SIGN_PATH)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0), (90.0, 120.0), (120.0, 90.0))
    rows = [run_decision_case(:force_sign, target, initial, mechanism;
                              final_time=0.0, variant)
            for (target, initial) in angle_pairs for mechanism in HISTORICAL_MECHANISMS]
    data = DataFrame(rows)
    data.contact_shape_acceleration = fill(NaN, nrow(data))
    data.expected_direction = sign.(data.target .- data.initial_angle)
    data.sign_pass = falses(nrow(data))

    for (target, initial) in angle_pairs
        group = findall((data.target .== target) .& (data.initial_angle .== initial))
        control = only(filter(index -> data.mechanism[index] == "none", group))
        for index in group
            data.contact_shape_acceleration[index] = data.shape_acceleration[index] -
                                                     data.shape_acceleration[control]
            data.sign_pass[index] = data.mechanism[index] == "none" ||
                                    data.expected_direction[index] *
                                    data.contact_shape_acceleration[index] > 0
        end
    end
    CSV.write(output_path, data)
    for mechanism in String.(REJECTED_MECHANISMS)
        mechanism_rows = data[data.mechanism .== mechanism, :]
        println(mechanism, " fixed-particle signs: ", count(mechanism_rows.sign_pass),
                "/", nrow(mechanism_rows))
    end
    println("Wrote fixed-particle force signs to ", output_path)
    return data
end

function run_ghost_geometric_force_sign(; output_path=GHOST_FORCE_SIGN_PATH)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0), (90.0, 120.0), (120.0, 90.0))
    rows = NamedTuple[]
    for (target, initial_angle) in angle_pairs
        control = css_sessile_drop(target, 0.0, nothing;
                                   target_particle_count=1500, mechanism=:none,
                                   initial_contact_angle=initial_angle,
                                   boundary_contact_threshold=0.1,
                                   damping_coefficient=4000.0)
        geometric = css_sessile_drop(target, 0.0, nothing;
                                     target_particle_count=1500, mechanism=:geometric,
                                     initial_contact_angle=initial_angle,
                                     boundary_contact_threshold=0.1,
                                     damping_coefficient=4000.0)
        components = raw_normal_components(geometric)
        acceleration = ghost_geometric_shape_acceleration(geometric, components, target)
        contact_acceleration = acceleration - control.shape_acceleration
        expected_direction = sign(target - initial_angle)
        push!(rows,
              (; kind="force_sign", variant="ghost_geometric", target, initial_angle,
               requested_particles=1500, particle_count=geometric.particle_count,
               control_acceleration=control.shape_acceleration, acceleration,
               contact_shape_acceleration=contact_acceleration, expected_direction,
               sign_pass=expected_direction * contact_acceleration > 0))
    end
    data = DataFrame(rows)
    CSV.write(output_path, data)
    println("ghost geometric fixed-particle signs: ", count(data.sign_pass), "/",
            nrow(data))
    println("Wrote ghost-geometric force signs to ", output_path)
    return data
end

function run_wall_energy_force_sign(; output_path=WALL_ENERGY_FORCE_SIGN_PATH)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0), (90.0, 120.0), (120.0, 90.0))
    rows = NamedTuple[]
    for (target, initial_angle) in angle_pairs
        control = css_sessile_drop(target, 0.0, nothing;
                                   target_particle_count=1500, mechanism=:none,
                                   initial_contact_angle=initial_angle,
                                   boundary_contact_threshold=0.1,
                                   damping_coefficient=4000.0)
        clf = css_sessile_drop(target, 0.0, nothing;
                               target_particle_count=1500,
                               mechanism=:contact_line_force,
                               initial_contact_angle=initial_angle,
                               boundary_contact_threshold=0.1,
                               damping_coefficient=4000.0)
        expected_direction = sign(target - initial_angle)
        for line_delta_scale in (1.0, 2.0)
            wall_acceleration = wall_energy_shape_acceleration(clf, target;
                                                               line_delta_scale)
            acceleration = control.shape_acceleration + wall_acceleration
            push!(rows,
                  (; kind="force_sign", variant="wall_energy_$(Int(line_delta_scale))x",
                   target, initial_angle, requested_particles=1500,
                   particle_count=clf.particle_count,
                   control_acceleration=control.shape_acceleration,
                   wall_acceleration, acceleration,
                   contact_shape_acceleration=wall_acceleration, expected_direction,
                   sign_pass=expected_direction * acceleration > 0))
        end
    end
    data = DataFrame(rows)
    CSV.write(output_path, data)
    for variant in unique(data.variant)
        rows_variant = data[data.variant .== variant, :]
        println(variant, " total fixed-particle signs: ", count(rows_variant.sign_pass),
                "/", nrow(rows_variant))
    end
    println("Wrote wall-energy force signs to ", output_path)
    return data
end

function validate_phase2_inputs(matrix, sensitivity)
    require(nrow(matrix) == 30, "Phase 2 matrix must contain exactly 30 rows")
    require(nrow(sensitivity) == 8,
            "Phase 2 sensitivity matrix must contain exactly eight rows")

    for mechanism in String.(REJECTED_MECHANISMS), target in TARGET_ANGLES,
        resolution in RESOLUTIONS
        rows = matrix[(matrix.mechanism .== mechanism) .& (matrix.target .== target) .& (matrix.requested_particles .== resolution),
                      :]
        require(nrow(rows) == 1,
                "missing or duplicate Phase 2 row for $mechanism/$target/$resolution")
    end
    for mechanism in String.(REJECTED_MECHANISMS), threshold in (0.0, 0.1),
        damping in (2000.0, 8000.0)
        rows = sensitivity[(sensitivity.mechanism .== mechanism) .& (sensitivity.boundary_contact_threshold .== threshold) .& (sensitivity.damping_coefficient .== damping),
                           :]
        require(nrow(rows) == 1,
                "missing or duplicate sensitivity row for $mechanism/$threshold/$damping")
    end

    for column in (:circle_angle, :circle_error, :rms_speed, :density_min,
         :density_max, :minimum_dt, :runtime)
        require(all(isfinite, matrix[!, column]), "non-finite Phase 2 field: $column")
    end
    require(all(matrix.circle_error .<= 5), "a Phase 2 angle row exceeds 5 degrees")
    require(all(matrix.below_wall .== 0), "a Phase 2 row penetrates the wall")
    require(all((matrix.density_min .>= 980) .& (matrix.density_max .<= 1020)),
            "a Phase 2 row violates density bounds")
    require(all(matrix.rms_speed .< 5.0e-3), "a Phase 2 row is not settled")
    require(all(matrix.accepted_steps .< 2000), "a Phase 2 row exceeds the step budget")
    rejected_fraction = matrix.rejected_steps ./
                        (matrix.accepted_steps .+ matrix.rejected_steps)
    require(all(rejected_fraction .<= 0.25),
            "a Phase 2 row exceeds the rejection budget")
    require(all(sensitivity.pass), "a Phase 2 sensitivity row failed")

    return nothing
end

function score_phase2(; output_path=SCORECARD_PATH)
    matrix = CSV.read(PHASE2_MATRIX, DataFrame)
    sensitivity = CSV.read(PHASE2_SENSITIVITY, DataFrame)
    validate_phase2_inputs(matrix, sensitivity)
    perturbation = isfile(PERTURBATION_PATH) ? CSV.read(PERTURBATION_PATH, DataFrame) :
                   DataFrame()
    threshold_replay = isfile(THRESHOLD_PATH) ? CSV.read(THRESHOLD_PATH, DataFrame) :
                       DataFrame()
    timestep = isfile(TIMESTEP_PATH) ? CSV.read(TIMESTEP_PATH, DataFrame) : DataFrame()
    cost = isfile(COST_PATH) ? CSV.read(COST_PATH, DataFrame) : DataFrame()
    control_runtime = isempty(cost) ? NaN :
                      median(cost[cost.mechanism .== "none", :solver_runtime])

    rows = NamedTuple[]
    for mechanism in String.(REJECTED_MECHANISMS)
        mechanism_rows = matrix[matrix.mechanism .== mechanism, :]
        resolution_metrics = Dict{Int, NamedTuple}()
        monotone_targets = true
        for resolution in RESOLUTIONS
            data = mechanism_rows[mechanism_rows.requested_particles .== resolution, :]
            sort!(data, :target)
            monotone_targets &= all(diff(data.circle_angle) .> 0)
            resolution_metrics[resolution] = (; mae=mean(data.circle_error),
                                              maximum=maximum(data.circle_error))
        end
        endpoint_regressions = 0
        for target in TARGET_ANGLES
            data = mechanism_rows[mechanism_rows.target .== target, :]
            coarse = only(data[data.requested_particles .== 750, :circle_error])
            fine = only(data[data.requested_particles .== 3000, :circle_error])
            endpoint_regressions += fine > coarse
        end
        sensitivity_rows = sensitivity[sensitivity.mechanism .== mechanism, :]
        sensitivity_span = maximum(sensitivity_rows.circle_angle) -
                           minimum(sensitivity_rows.circle_angle)
        rejected_fraction = mechanism_rows.rejected_steps ./
                            (mechanism_rows.accepted_steps .+
                             mechanism_rows.rejected_steps)
        coarse = resolution_metrics[750]
        medium = resolution_metrics[1500]
        fine = resolution_metrics[3000]
        static_eligible = monotone_targets && coarse.mae >= fine.mae &&
                          coarse.maximum >= fine.maximum && sensitivity_span <= 1 &&
                          all(rejected_fraction .<= 0.25)
        response_rows = isempty(perturbation) ? DataFrame() :
                        perturbation[perturbation.mechanism .== mechanism, :]
        response_cases = nrow(response_rows)
        response_passes = response_cases == 0 ? 0 : count(response_rows.response_pass)
        acceleration_passes = response_cases == 0 ? 0 :
                              count(response_rows.acceleration_toward_target)
        motion_passes = response_cases == 0 ? 0 :
                        count(response_rows.motion_toward_target)
        mean_error_reduction = response_cases == 0 ? NaN :
                               mean(response_rows.error_reduction)
        replay_rows = isempty(threshold_replay) ? DataFrame() :
                      threshold_replay[threshold_replay.mechanism .== mechanism, :]
        threshold_pass = nrow(replay_rows) == length(TARGET_ANGLES) && all(replay_rows.pass)
        timestep_rows = isempty(timestep) ? DataFrame() :
                        timestep[timestep.mechanism .== mechanism, :]
        timestep_pass = nrow(timestep_rows) == 2 && all(timestep_rows.pass)
        cost_rows = isempty(cost) ? DataFrame() : cost[cost.mechanism .== mechanism, :]
        median_runtime = isempty(cost_rows) ? NaN : median(cost_rows.solver_runtime)
        runtime_mad = isempty(cost_rows) ? NaN :
                      median(abs.(cost_rows.solver_runtime .- median_runtime))
        runtime_overhead = median_runtime / control_runtime
        contact_cache_bytes = isempty(cost_rows) ? 0 :
                              round(Int, median(cost_rows.contact_cache_bytes))
        eligible = static_eligible && threshold_pass && timestep_pass &&
                   response_cases == 4 && response_passes == 4
        push!(rows,
              (; mechanism, eligible, static_cells=nrow(mechanism_rows), static_eligible,
               monotone_targets, mae_750=coarse.mae, max_750=coarse.maximum,
               mae_1500=medium.mae, max_1500=medium.maximum,
               mae_3000=fine.mae, max_3000=fine.maximum,
               endpoint_regressions, sensitivity_span,
               threshold_pass, timestep_pass, response_cases, response_passes,
               acceleration_passes, motion_passes, mean_error_reduction,
               median_runtime, runtime_mad, runtime_overhead, contact_cache_bytes,
               worst_rms_speed=maximum(mechanism_rows.rms_speed),
               density_min=minimum(mechanism_rows.density_min),
               density_max=maximum(mechanism_rows.density_max),
               max_rejected_fraction=maximum(rejected_fraction),
               minimum_dt=minimum(mechanism_rows.minimum_dt),
               mean_runtime=mean(mechanism_rows.runtime),
               total_runtime=sum(mechanism_rows.runtime)))
    end
    scorecard = DataFrame(rows)
    require(all(scorecard.static_eligible),
            "at least one mechanism fails static eligibility")
    CSV.write(output_path, scorecard)
    println(scorecard)
    println("Wrote Phase 3 scorecard to ", output_path)
    return scorecard
end

function run_threshold_replay(; output_path=THRESHOLD_PATH)
    rows = [run_decision_case(:threshold, target, target, mechanism)
            for mechanism in REJECTED_MECHANISMS for target in TARGET_ANGLES]
    data = DataFrame(rows)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25)
    require(all(data.pass), "recommended-threshold replay failed")
    CSV.write(output_path, data)
    println("Wrote threshold replay to ", output_path)
    return data
end

function run_perturbation_matrix(; output_path=PERTURBATION_PATH)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0), (90.0, 120.0), (120.0, 90.0))
    rows = [run_decision_case(:perturbation, target, initial, mechanism)
            for (target, initial) in angle_pairs for mechanism in HISTORICAL_MECHANISMS]
    data = DataFrame(rows)
    data.control_error_reduction = fill(NaN, nrow(data))
    data.contact_shape_acceleration = fill(NaN, nrow(data))
    data.motion_toward_target = falses(nrow(data))
    data.beats_control = falses(nrow(data))
    data.acceleration_toward_target = falses(nrow(data))
    data.response_pass = falses(nrow(data))

    for (target, initial) in angle_pairs
        group = findall((data.target .== target) .& (data.initial_angle .== initial))
        control = only(filter(index -> data.mechanism[index] == "none", group))
        direction = sign(target - initial)
        for index in group
            data.control_error_reduction[index] = data.error_reduction[control]
            data.contact_shape_acceleration[index] = data.shape_acceleration[index] -
                                                     data.shape_acceleration[control]
            data.motion_toward_target[index] = direction * (data.final_circle[index] -
                                                data.initial_circle[index]) > 0
            if data.mechanism[index] == "none"
                data.beats_control[index] = true
                data.acceleration_toward_target[index] = true
                data.response_pass[index] = true
            else
                data.beats_control[index] = data.error_reduction[index] >
                                            data.error_reduction[control]
                data.acceleration_toward_target[index] = direction *
                                                         data.contact_shape_acceleration[index] >
                                                         0
                data.response_pass[index] = data.error_reduction[index] > 0 &&
                                            data.motion_toward_target[index] &&
                                            data.beats_control[index] &&
                                            data.acceleration_toward_target[index] &&
                                            data.below_wall[index] == 0 &&
                                            980 <= data.density_min[index] <= 1020 &&
                                            980 <= data.density_max[index] <= 1020 &&
                                            data.rejected_fraction[index] <= 0.25
            end
        end
    end
    CSV.write(output_path, data)
    println("Wrote perturbation matrix to ", output_path)
    return data
end

function run_timestep_cases(; output_path=TIMESTEP_PATH)
    rows = NamedTuple[]
    for mechanism in REJECTED_MECHANISMS,
        (target, resolution) in ((90.0, 1500),
                                 (30.0, 3000))
        push!(rows, run_decision_case(:timestep, target, target, mechanism; resolution))
    end
    data = DataFrame(rows)
    data.pass = (data.eta_p01 .>= 0.05) .& (data.eta_tail_head .>= 0.5) .&
                (data.rejected_fraction .<= 0.25) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020)
    require(all(data.pass), "a representative timestep case failed")
    CSV.write(output_path, data)
    println("Wrote timestep diagnostics to ", output_path)
    return data
end

function run_cost_cases(; output_path=COST_PATH)
    for mechanism in HISTORICAL_MECHANISMS
        run_decision_case(:warmup, 90.0, 90.0, mechanism;
                          resolution=200, final_time=0.001)
    end

    orders = ((:none, :geometric, :contact_line_force),
              (:geometric, :contact_line_force, :none),
              (:contact_line_force, :none, :geometric))
    rows = NamedTuple[]
    for (repeat, order) in enumerate(orders), mechanism in order
        push!(rows, run_decision_case(:cost, 90.0, 90.0, mechanism; repeat))
    end
    data = DataFrame(rows)
    control_median = median(data[data.mechanism .== "none", :solver_runtime])
    data.normalized_runtime = data.solver_runtime ./ control_median
    CSV.write(output_path, data)

    for mechanism in String.(HISTORICAL_MECHANISMS)
        values = data[data.mechanism .== mechanism, :solver_runtime]
        center = median(values)
        @printf("%-18s median %.3f s MAD %.3f s overhead %.3f\n",
                mechanism, center, median(abs.(values .- center)), center / control_median)
    end
    println("Wrote repeated cost measurements to ", output_path)
    return data
end

function run_selected_matrix(mechanism; output_path=SELECTED_PATH)
    require(mechanism in REJECTED_MECHANISMS,
            "historical mechanism must be geometric or CLF")
    rows = [run_decision_case(:selected, target, target, mechanism; resolution)
            for resolution in RESOLUTIONS for target in TARGET_ANGLES]
    data = DataFrame(rows)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                (data.eta_p01 .>= 0.05) .& (data.eta_tail_head .>= 0.5)
    require(all(data.pass), "selected-mechanism replay failed")
    CSV.write(output_path, data)
    println("Wrote selected-mechanism matrix to ", output_path)
    return data
end

function r4_wetted_area_parameters(; smoothing_length_ratio=1.4)
    kernel = TrixiParticles.WendlandC2Kernel{3}()
    profile_data = kernel_plane_profile(kernel)
    offset_data = kernel_lattice_color_offsets(kernel, 64)
    edge_data = Dict(target => canonical_wetted_edge_data(profile_data, offset_data,
                                                          target;
                                                          production_cells_per_h=smoothing_length_ratio)
                     for target in TARGET_ANGLES)
    boundary_distance = inv(2smoothing_length_ratio)
    flooded_reference = halfspace_color_value(profile_data, boundary_distance)
    return (; edge_data, flooded_reference, smoothing_length_ratio)
end

function r4_wetted_area_model(target, parameters)
    edge = parameters.edge_data[target]
    return CorrectedWettedAreaContact(target, edge.normalized_shift,
                                      parameters.flooded_reference)
end

function run_r4_simulation(target, initial_angle, parameters; active=true,
                           resolution=1500, threshold=0.1, damping=4000.0,
                           final_time=0.01)
    mechanism = active ? :corrected_wetted_area : :none
    validation_contact_model = active ? r4_wetted_area_model(target, parameters) : nothing
    return quiet_css_sessile_drop(target, final_time, nothing;
                                  target_particle_count=resolution, mechanism,
                                  initial_contact_angle=initial_angle,
                                  boundary_contact_threshold=threshold,
                                  damping_coefficient=damping,
                                  smoothing_length_ratio=parameters.smoothing_length_ratio,
                                  validation_contact_model)
end

function r4_perturbed_energy(model, system, boundary_system, coordinates,
                             boundary_coordinates, density, density_rate,
                             displacement, pairs, epsilon)
    colorfield = copy(boundary_system.boundary_model.cache.initial_colorfield)
    for (particle, neighbor) in pairs
        model.exposed[neighbor] || continue
        distance2 = zero(eltype(system))
        for dim in 1:TrixiParticles.ndims(system)
            difference = coordinates[dim, particle] +
                         epsilon * displacement[dim, particle] -
                         boundary_coordinates[dim, neighbor]
            distance2 += difference^2
        end
        perturbed_density = density[particle] + epsilon * density_rate[particle]
        colorfield[neighbor] += TrixiParticles.hydrodynamic_mass(system, particle) /
                                perturbed_density *
                                TrixiParticles.smoothing_kernel(system, sqrt(distance2),
                                                                particle)
    end

    particle_spacing = system.cache.reference_particle_spacing
    raw_area = zero(eltype(system))
    for boundary_particle in TrixiParticles.eachparticle(boundary_system)
        model.exposed[boundary_particle] || continue
        fraction = clamp(colorfield[boundary_particle] / model.flooded_reference, 0, 1)
        raw_area += particle_spacing^2 * smoothstep01(fraction)
    end
    raw_radius = sqrt(raw_area / pi)
    edge_shift = model.normalized_edge_shift *
                 TrixiParticles.initial_smoothing_length(system)
    corrected_radius = max(raw_radius - edge_shift, zero(raw_radius))
    contact_cosine = r4_contact_cosine(model)
    iszero(contact_cosine) && return zero(contact_cosine)
    return -system.surface_tension.surface_tension_coefficient * contact_cosine * pi *
           corrected_radius^2
end

function r4_directional_energy_gradient(result)
    semi = result.solution.prob.p.semi
    system = semi.systems[1]
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    model = TrixiParticles.surface_normal_method(system).contact_model
    boundary_system = r4_wetted_area_boundary_system(semi)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    boundary_coordinates = Array(TrixiParticles.current_coordinates(u_boundary,
                                                                    boundary_system))
    density = [TrixiParticles.current_density(v, system, particle)
               for particle in TrixiParticles.eachparticle(system)]

    displacement = similar(coordinates)
    displacement_scale = max(maximum(abs, coordinates),
                             system.cache.reference_particle_spacing)
    for particle in TrixiParticles.eachparticle(system)
        displacement[1, particle] = -coordinates[1, particle] / displacement_scale
        displacement[2, particle] = -coordinates[2, particle] / displacement_scale
        displacement[3, particle] = 2coordinates[3, particle] / displacement_scale
    end

    density_rate = zeros(eltype(system), TrixiParticles.nparticles(system))
    TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                                          points=TrixiParticles.each_integrated_particle(system),
                                          parallelization_backend=TrixiParticles.SerialBackend()) do particle,
                                                                                                     neighbor,
                                                                                                     pos_diff,
                                                                                                     distance
        gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                        particle)
        velocity_dot_gradient = zero(eltype(system))
        for dim in 1:TrixiParticles.ndims(system)
            velocity_dot_gradient += (displacement[dim, particle] -
                                      displacement[dim, neighbor]) * gradient[dim]
        end
        mass_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
        density_rate[particle] += density[particle] / density[neighbor] * mass_b *
                                  velocity_dot_gradient
    end

    pairs = Tuple{Int, Int}[]
    TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                          boundary_coordinates, semi;
                                          points=TrixiParticles.each_integrated_particle(system),
                                          parallelization_backend=TrixiParticles.SerialBackend()) do particle,
                                                                                                     neighbor,
                                                                                                     pos_diff,
                                                                                                     distance
        push!(pairs, (particle, neighbor))
    end

    epsilon = 1.0e-5 * system.cache.reference_particle_spacing
    energy_plus = r4_perturbed_energy(model, system, boundary_system, coordinates,
                                      boundary_coordinates, density, density_rate,
                                      displacement, pairs, epsilon)
    energy_minus = r4_perturbed_energy(model, system, boundary_system, coordinates,
                                       boundary_coordinates, density, density_rate,
                                       displacement, pairs, -epsilon)
    finite_difference = (energy_plus - energy_minus) / (2epsilon)
    analytic_derivative = zero(eltype(system))
    explicit_derivative = zero(eltype(system))
    density_derivative = zero(eltype(system))
    for particle in TrixiParticles.eachparticle(system),
        dim in 1:TrixiParticles.ndims(system)
        explicit_derivative -= model.explicit_force[dim, particle] *
                               displacement[dim, particle]
        density_derivative -= model.density_force[dim, particle] *
                              displacement[dim, particle]
    end
    analytic_derivative = explicit_derivative + density_derivative
    derivative_scale = max(abs(finite_difference), abs(analytic_derivative))
    relative_error = iszero(derivative_scale) ? zero(derivative_scale) :
                     abs(finite_difference - analytic_derivative) / derivative_scale
    return (; epsilon, energy_plus, energy_minus, finite_difference,
            analytic_derivative, explicit_derivative, density_derivative,
            relative_error, fluid_wall_pairs=length(pairs))
end

function r4_static_row(kind, target, initial_angle, result, diagnostics;
                       gradient=nothing, control_acceleration=NaN)
    expected_direction = sign(target - initial_angle)
    contact_shape_acceleration = isfinite(control_acceleration) ?
                                 result.shape_acceleration - control_acceleration : NaN
    total_sign_pass = kind == "force_sign" ?
                      expected_direction * result.shape_acceleration > 0 : true
    contact_sign_pass = kind == "force_sign" ?
                        (target == 90 ? iszero(contact_shape_acceleration) :
                         expected_direction * contact_shape_acceleration > 0) : true
    gradient_error = isnothing(gradient) ? NaN : gradient.relative_error
    gradient_pass = isnothing(gradient) ? true : gradient_error <= 1.0e-5
    zero_at_90 = target == 90 ?
                 iszero(diagnostics.energy) &&
                 iszero(diagnostics.explicit_force_scale) &&
                 iszero(diagnostics.density_force_scale) &&
                 iszero(diagnostics.wall_force_scale) : true
    reaction_pass = diagnostics.explicit_reaction_residual <= 1.0e-12 &&
                    diagnostics.density_resultant_residual <= 1.0e-12 &&
                    diagnostics.total_momentum_residual <= 1.0e-12
    finite_pass = all(isfinite,
                      (diagnostics.energy, diagnostics.raw_area,
                       diagnostics.corrected_area, diagnostics.area_derivative,
                       result.shape_acceleration,
                       diagnostics.explicit_reaction_residual,
                       diagnostics.density_resultant_residual,
                       diagnostics.total_momentum_residual))
    stage_pass = gradient_pass && zero_at_90 && reaction_pass && finite_pass &&
                 total_sign_pass
    return (; kind, target, initial_angle, requested_particles=1500,
            particle_count=result.particle_count, energy=diagnostics.energy,
            raw_area=diagnostics.raw_area, corrected_area=diagnostics.corrected_area,
            area_derivative=diagnostics.area_derivative,
            finite_difference=isnothing(gradient) ? NaN : gradient.finite_difference,
            analytic_derivative=isnothing(gradient) ? NaN : gradient.analytic_derivative,
            explicit_derivative=isnothing(gradient) ? NaN : gradient.explicit_derivative,
            density_derivative=isnothing(gradient) ? NaN : gradient.density_derivative,
            gradient_relative_error=gradient_error, gradient_pass,
            shape_acceleration=result.shape_acceleration, control_acceleration,
            contact_shape_acceleration, expected_direction, total_sign_pass,
            contact_sign_pass, zero_at_90,
            explicit_reaction_residual=diagnostics.explicit_reaction_residual,
            density_resultant_residual=diagnostics.density_resultant_residual,
            total_momentum_residual=diagnostics.total_momentum_residual,
            explicit_force_scale=diagnostics.explicit_force_scale,
            density_force_scale=diagnostics.density_force_scale,
            wall_force_scale=diagnostics.wall_force_scale,
            reaction_pass, finite_pass, stage_pass)
end

function run_r4_static_gate(; output_path=R4_STATIC_PATH)
    parameters = r4_wetted_area_parameters()
    rows = NamedTuple[]
    for target in TARGET_ANGLES
        result = run_r4_simulation(target, target, parameters; final_time=0.0)
        gradient = r4_directional_energy_gradient(result)
        diagnostics = corrected_wetted_area_contact_diagnostics(TrixiParticles.surface_normal_method(result.solution.prob.p.semi.systems[1]).contact_model)
        push!(rows,
              r4_static_row("energy_gradient", target, target, result,
                            diagnostics; gradient))
    end

    angle_pairs = ((60.0, 90.0), (90.0, 60.0),
                   (90.0, 120.0), (120.0, 90.0))
    for (target, initial_angle) in angle_pairs
        control = run_r4_simulation(target, initial_angle, parameters; active=false,
                                    final_time=0.0)
        result = run_r4_simulation(target, initial_angle, parameters; final_time=0.0)
        diagnostics = corrected_wetted_area_contact_diagnostics(TrixiParticles.surface_normal_method(result.solution.prob.p.semi.systems[1]).contact_model)
        push!(rows,
              r4_static_row("force_sign", target, initial_angle, result,
                            diagnostics;
                            control_acceleration=control.shape_acceleration))
    end
    data = DataFrame(rows)
    CSV.write(output_path, data)
    println("R4-W static gate: ", count(data.stage_pass), "/", nrow(data),
            "; force signs ", count(data[data.kind .== "force_sign", :total_sign_pass]),
            "/4; max gradient error ",
            maximum(data[data.kind .== "energy_gradient", :gradient_relative_error]))
    println("Wrote R4-W static evidence to ", output_path)
    require(all(data.stage_pass), "R4-W algebra/static gate failed")
    return data
end

function require_r4_static_gate()
    require(isfile(R4_STATIC_PATH), "run `r4_static` before R4 dynamics")
    data = CSV.read(R4_STATIC_PATH, DataFrame)
    require(count(data.kind .== "energy_gradient") == 5 &&
                count(data.kind .== "force_sign") == 4 && all(data.stage_pass),
            "R4-W static evidence does not pass")
    return data
end

function r4_dynamic_row(kind, target, initial_angle, result; active=true,
                        resolution=1500, threshold=0.1, damping=4000.0,
                        final_time=0.01, repeat=1, total_runtime=result.runtime)
    initial_error = result.initial_circle.angle - target
    final_error = result.circle.angle - target
    error_reduction = abs(initial_error) > eps() ?
                      1 - abs(final_error) / abs(initial_error) : NaN
    diagnostics = active ? result.initial_contact_diagnostics : nothing
    final_diagnostics = active ? result.final_contact_diagnostics : nothing
    return (; kind=String(kind), target, initial_angle,
            mechanism=active ? "r4_wetted_area" : "none",
            requested_particles=resolution, particle_count=result.particle_count,
            threshold, damping, final_time, repeat,
            initial_circle=result.initial_circle.angle,
            final_circle=result.circle.angle, initial_error, final_error,
            error_reduction, cap_angle=result.final.angle,
            normal_angle=result.measured_contact_angle,
            shape_acceleration=result.shape_acceleration,
            boundary_shape_acceleration=result.boundary_shape_acceleration,
            wetting_energy=active ? diagnostics.energy : 0.0,
            explicit_reaction_residual=active ?
                                       diagnostics.explicit_reaction_residual : 0.0,
            density_resultant_residual=active ?
                                       diagnostics.density_resultant_residual : 0.0,
            total_momentum_residual=active ? diagnostics.total_momentum_residual : 0.0,
            max_explicit_reaction_residual=active ?
                                           final_diagnostics.max_explicit_reaction_residual :
                                           0.0,
            max_density_resultant_residual=active ?
                                           final_diagnostics.max_density_resultant_residual :
                                           0.0,
            max_total_momentum_residual=active ?
                                        final_diagnostics.max_total_momentum_residual : 0.0,
            below_wall=result.below_wall, density_min=result.density_range[1],
            density_max=result.density_range[2], rms_speed=result.rms_speed,
            settled=result.settled, accepted_steps=result.accepted_steps,
            rejected_steps=result.rejected_steps,
            rejected_fraction=result.rejected_fraction,
            minimum_dt=result.minimum_dt, dt_reference=result.dt_reference,
            eta_p01=result.eta_p01, eta_median=result.eta_median,
            eta_tail_head=result.eta_tail_head, solver_runtime=result.runtime,
            total_runtime,
            cache_bytes=Base.summarysize(result.solution.prob.p.semi.systems[1].cache),
            contact_cache_bytes=active ? diagnostics.cache_bytes : 0)
end

function run_r4_dynamic_case(kind, target, initial_angle, parameters; active=true,
                             resolution=1500, threshold=0.1, damping=4000.0,
                             final_time=0.01, repeat=1)
    result = nothing
    total_runtime = @elapsed result = run_r4_simulation(target, initial_angle,
                                                        parameters; active, resolution,
                                                        threshold, damping, final_time)
    return r4_dynamic_row(kind, target, initial_angle, result; active, resolution,
                          threshold, damping, final_time, repeat, total_runtime)
end

function r4_perturbation_data(parameters; final_time=0.01)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0),
                   (90.0, 120.0), (120.0, 90.0))
    rows = [run_r4_dynamic_case(:perturbation, target, initial, parameters; active,
                                final_time)
            for (target, initial) in angle_pairs for active in (false, true)]
    data = DataFrame(rows)
    data.control_error_reduction = fill(NaN, nrow(data))
    data.contact_shape_acceleration = fill(NaN, nrow(data))
    data.motion_toward_target = falses(nrow(data))
    data.beats_control = falses(nrow(data))
    data.acceleration_toward_target = falses(nrow(data))
    data.reaction_pass = falses(nrow(data))
    data.response_pass = falses(nrow(data))
    for (target, initial) in angle_pairs
        group = findall((data.target .== target) .& (data.initial_angle .== initial))
        control = only(filter(index -> data.mechanism[index] == "none", group))
        direction = sign(target - initial)
        for index in group
            data.control_error_reduction[index] = data.error_reduction[control]
            data.contact_shape_acceleration[index] = data.shape_acceleration[index] -
                                                     data.shape_acceleration[control]
            data.motion_toward_target[index] = direction *
                                               (data.final_circle[index] -
                                                data.initial_circle[index]) > 0
            data.beats_control[index] = data.mechanism[index] == "none" ||
                                        data.error_reduction[index] >
                                        data.error_reduction[control]
            data.acceleration_toward_target[index] = data.mechanism[index] == "none" ||
                                                     direction *
                                                     data.contact_shape_acceleration[index] >
                                                     0
            data.reaction_pass[index] = data.max_explicit_reaction_residual[index] <=
                                        1.0e-12 &&
                                        data.max_density_resultant_residual[index] <=
                                        1.0e-12 &&
                                        data.max_total_momentum_residual[index] <= 1.0e-12
            data.response_pass[index] = data.mechanism[index] == "none" ||
                                        (data.error_reduction[index] > 0 &&
                                         data.motion_toward_target[index] &&
                                         data.beats_control[index] &&
                                         data.acceleration_toward_target[index] &&
                                         data.below_wall[index] == 0 &&
                                         980 <= data.density_min[index] <= 1020 &&
                                         980 <= data.density_max[index] <= 1020 &&
                                         data.rejected_fraction[index] <= 0.25 &&
                                         data.reaction_pass[index])
        end
    end
    return data
end

function run_r4_perturbation_gate(; output_path=R4_PERTURBATION_PATH,
                                  final_time=0.01)
    require_r4_static_gate()
    parameters = r4_wetted_area_parameters()
    data = r4_perturbation_data(parameters; final_time)
    CSV.write(output_path, data)
    candidates = data[data.mechanism .== "r4_wetted_area", :]
    println("R4-W perturbation gate: ", count(candidates.response_pass), "/4",
            "; acceleration ", count(candidates.acceleration_toward_target), "/4",
            "; motion ", count(candidates.motion_toward_target), "/4")
    println("Wrote R4-W perturbation evidence to ", output_path)
    require(nrow(candidates) == 4 && all(candidates.response_pass),
            "R4-W perturbation gate failed")
    return data
end

function classify_r4_wetted_area_perturbation!(data)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0),
                   (90.0, 120.0), (120.0, 90.0))
    data.control_equivalent = falses(nrow(data))
    data.control_comparison_pass = falses(nrow(data))
    data.effective_acceleration_toward_target = falses(nrow(data))
    data.formulation_response_pass = falses(nrow(data))
    for (target, initial) in angle_pairs
        group = findall((data.target .== target) .& (data.initial_angle .== initial))
        control = only(filter(index -> data.mechanism[index] == "none", group))
        direction = sign(target - initial)
        for index in group
            if data.mechanism[index] == "none"
                data.control_equivalent[index] = true
                data.control_comparison_pass[index] = true
                data.effective_acceleration_toward_target[index] = true
                data.formulation_response_pass[index] = true
                continue
            end

            zero_target = target == 90
            data.control_equivalent[index] = zero_target &&
                                             iszero(data.contact_shape_acceleration[index]) &&
                                             data.final_circle[index] ==
                                             data.final_circle[control] &&
                                             data.error_reduction[index] ==
                                             data.error_reduction[control]
            data.control_comparison_pass[index] = zero_target ?
                                                  data.control_equivalent[index] :
                                                  data.beats_control[index]
            data.effective_acceleration_toward_target[index] = if zero_target
                data.control_equivalent[index] &&
                    direction * data.shape_acceleration[index] > 0
            else
                direction * data.contact_shape_acceleration[index] > 0
            end
            data.formulation_response_pass[index] = data.error_reduction[index] > 0 &&
                                                    data.motion_toward_target[index] &&
                                                    data.control_comparison_pass[index] &&
                                                    data.effective_acceleration_toward_target[index] &&
                                                    data.below_wall[index] == 0 &&
                                                    980 <= data.density_min[index] <=
                                                    1020 &&
                                                    980 <= data.density_max[index] <=
                                                    1020 &&
                                                    data.rejected_fraction[index] <= 0.25 &&
                                                    data.reaction_pass[index]
        end
    end
    return data
end

function classify_initial_r4_perturbation(; input_path=R4_PERTURBATION_PATH,
                                          output_path=R4_PERTURBATION_CLASSIFIED_PATH)
    require(isfile(input_path), "run `r4_perturbation` before classification")
    data = CSV.read(input_path, DataFrame)
    classify_r4_wetted_area_perturbation!(data)
    CSV.write(output_path, data)
    candidates = data[data.mechanism .== "r4_wetted_area", :]
    println("R4-W formulation-consistent initial classification: ",
            count(candidates.formulation_response_pass), "/4; acceleration ",
            count(candidates.effective_acceleration_toward_target), "/4; motion ",
            count(candidates.motion_toward_target), "/4")
    println("Wrote classified initial evidence to ", output_path)
    return data
end

function run_r4_perturbation_extension(; output_path=R4_PERTURBATION_EXTENDED_PATH,
                                       final_time=0.02)
    require(final_time == 0.02, "the sole R4-W extension is frozen at 0.02 s")
    require_r4_static_gate()
    initial = classify_initial_r4_perturbation()
    initial_candidates = initial[initial.mechanism .== "r4_wetted_area", :]
    unresolved = initial_candidates[.!initial_candidates.formulation_response_pass, :]
    require(nrow(initial_candidates) == 4 &&
                all(initial_candidates.effective_acceleration_toward_target),
            "R4-W is not eligible for the uniform extension")
    require(all(abs.(unresolved.final_circle .- unresolved.initial_circle) .< 1),
            "unresolved R4-W motion exceeds estimator resolution")

    parameters = r4_wetted_area_parameters()
    data = r4_perturbation_data(parameters; final_time)
    classify_r4_wetted_area_perturbation!(data)
    CSV.write(output_path, data)
    candidates = data[data.mechanism .== "r4_wetted_area", :]
    println("R4-W extended perturbation gate: ",
            count(candidates.formulation_response_pass), "/4; acceleration ",
            count(candidates.effective_acceleration_toward_target), "/4; motion ",
            count(candidates.motion_toward_target), "/4")
    println("Wrote extended R4-W perturbation evidence to ", output_path)
    require(nrow(candidates) == 4 && all(candidates.formulation_response_pass),
            "extended R4-W perturbation gate failed")
    return data
end

function require_r4_perturbation_gate()
    require(isfile(R4_PERTURBATION_EXTENDED_PATH),
            "run the permitted `r4_perturbation_extended` replay first")
    data = CSV.read(R4_PERTURBATION_EXTENDED_PATH, DataFrame)
    candidates = data[data.mechanism .== "r4_wetted_area", :]
    require(nrow(candidates) == 4 && all(candidates.formulation_response_pass),
            "R4-W perturbation evidence does not pass")
    return data
end

function run_r4_threshold_gate(; output_path=R4_THRESHOLD_PATH)
    require_r4_perturbation_gate()
    parameters = r4_wetted_area_parameters()
    rows = [run_r4_dynamic_case(:threshold, target, target, parameters)
            for target in TARGET_ANGLES]
    data = DataFrame(rows)
    data.reaction_pass = (data.max_explicit_reaction_residual .<= 1.0e-12) .&
                         (data.max_density_resultant_residual .<= 1.0e-12) .&
                         (data.max_total_momentum_residual .<= 1.0e-12)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                data.reaction_pass
    CSV.write(output_path, data)
    println("R4-W threshold gate: ", count(data.pass), "/5")
    println("Wrote R4-W threshold evidence to ", output_path)
    require(all(data.pass), "R4-W threshold replay failed")
    return data
end

function require_r4_threshold_gate()
    require(isfile(R4_THRESHOLD_PATH), "run `r4_threshold` first")
    data = CSV.read(R4_THRESHOLD_PATH, DataFrame)
    require(nrow(data) == 5 && all(data.pass),
            "R4-W threshold evidence does not pass")
    return data
end

function run_r4_timestep_gate(; output_path=R4_TIMESTEP_PATH)
    require_r4_threshold_gate()
    parameters = r4_wetted_area_parameters()
    rows = [run_r4_dynamic_case(:timestep, target, target, parameters; resolution)
            for (target, resolution) in ((90.0, 1500), (30.0, 3000))]
    data = DataFrame(rows)
    data.reaction_pass = (data.max_explicit_reaction_residual .<= 1.0e-12) .&
                         (data.max_density_resultant_residual .<= 1.0e-12) .&
                         (data.max_total_momentum_residual .<= 1.0e-12)
    data.pass = (data.eta_p01 .>= 0.05) .& (data.eta_tail_head .>= 0.5) .&
                (data.rejected_fraction .<= 0.25) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                data.reaction_pass
    CSV.write(output_path, data)
    println("R4-W timestep gate: ", count(data.pass), "/2")
    println("Wrote R4-W timestep evidence to ", output_path)
    require(all(data.pass), "R4-W timestep gate failed")
    return data
end

function require_r4_timestep_gate()
    require(isfile(R4_TIMESTEP_PATH), "run `r4_timestep` first")
    data = CSV.read(R4_TIMESTEP_PATH, DataFrame)
    require(nrow(data) == 2 && all(data.pass),
            "R4-W timestep evidence does not pass")
    return data
end

function run_r4_cost_cases(; output_path=R4_COST_PATH, target=90.0)
    require_r4_timestep_gate()
    parameters = r4_wetted_area_parameters()
    for active in (false, true)
        run_r4_dynamic_case(:warmup, target, target, parameters; active,
                            resolution=200, final_time=0.001)
    end
    orders = ((false, true), (true, false), (false, true))
    rows = NamedTuple[]
    for (repeat, order) in enumerate(orders), active in order
        push!(rows, run_r4_dynamic_case(:cost, target, target, parameters; active,
                                        repeat))
    end
    data = DataFrame(rows)
    control_median = median(data[data.mechanism .== "none", :solver_runtime])
    data.normalized_runtime = data.solver_runtime ./ control_median
    CSV.write(output_path, data)
    for mechanism in ("none", "r4_wetted_area")
        values = data[data.mechanism .== mechanism, :solver_runtime]
        center = median(values)
        @printf("%-18s median %.3f s MAD %.3f s overhead %.3f\n",
                mechanism, center, median(abs.(values .- center)),
                center / control_median)
    end
    println("Wrote R4-W ", target, "-degree cost evidence to ", output_path)
    require(all(isfinite, data.solver_runtime), "non-finite R4-W cost measurement")
    return data
end

function require_r4_cost_cases()
    require(isfile(R4_COST_PATH), "run `r4_cost` first")
    require(isfile(R4_ACTIVE_COST_PATH), "run `r4_cost_active` first")
    zero_path = CSV.read(R4_COST_PATH, DataFrame)
    active_path = CSV.read(R4_ACTIVE_COST_PATH, DataFrame)
    require(nrow(zero_path) == 6 && nrow(active_path) == 6 &&
                all(isfinite, zero_path.solver_runtime) &&
                all(isfinite, active_path.solver_runtime),
            "R4-W cost evidence is incomplete")
    return (; zero_path, active_path)
end

function run_r4_selected_matrix(; output_path=R4_SELECTED_PATH)
    require_r4_cost_cases()
    parameters = r4_wetted_area_parameters()
    rows = [run_r4_dynamic_case(:selected, target, target, parameters; resolution)
            for resolution in RESOLUTIONS for target in TARGET_ANGLES]
    data = DataFrame(rows)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                (data.eta_p01 .>= 0.05) .& (data.eta_tail_head .>= 0.5) .&
                (data.max_explicit_reaction_residual .<= 1.0e-12) .&
                (data.max_density_resultant_residual .<= 1.0e-12) .&
                (data.max_total_momentum_residual .<= 1.0e-12)
    CSV.write(output_path, data)
    println("R4-W selected matrix: ", count(data.pass), "/15")
    println("Wrote R4-W selected matrix to ", output_path)
    require(all(data.pass), "R4-W selected matrix failed")
    return data
end

function run_r4_sensitivity(; output_path=R4_SENSITIVITY_PATH)
    require(isfile(R4_SELECTED_PATH), "run `r4_selected` first")
    selected = CSV.read(R4_SELECTED_PATH, DataFrame)
    require(nrow(selected) == 15 && all(selected.pass),
            "R4-W selected matrix does not pass")
    parameters = r4_wetted_area_parameters()
    rows = [run_r4_dynamic_case(:sensitivity, 90.0, 90.0, parameters;
                                threshold, damping)
            for threshold in (0.0, 0.1) for damping in (2000.0, 8000.0)]
    data = DataFrame(rows)
    angle_span = maximum(data.final_circle) - minimum(data.final_circle)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                (data.max_total_momentum_residual .<= 1.0e-12) .&
                (angle_span .<= 1)
    data.angle_span = fill(angle_span, nrow(data))
    CSV.write(output_path, data)
    println("R4-W sensitivity: ", count(data.pass), "/4; span ", angle_span,
            " deg")
    println("Wrote R4-W sensitivity evidence to ", output_path)
    require(all(data.pass), "R4-W sensitivity matrix failed")
    return data
end

function run_production_wetted_area_simulation(target, initial_angle; active=true,
                                               resolution=1500, threshold=0.1,
                                               damping=4000.0, final_time=0.01)
    mechanism = active ? :wetted_area : :none
    return quiet_css_sessile_drop(target, final_time, nothing;
                                  target_particle_count=resolution, mechanism,
                                  initial_contact_angle=initial_angle,
                                  boundary_contact_threshold=threshold,
                                  damping_coefficient=damping,
                                  smoothing_length_ratio=1.4)
end

function production_contact_boundaries(semi)
    boundaries = Any[]
    for system in semi.systems
        cache = TrixiParticles.wetted_area_boundary_cache(system)
        isnothing(cache) || push!(boundaries, system)
    end
    return boundaries
end

function production_contact_rhs_diagnostics(result; state=result.solution.prob.u0.x)
    semi = result.solution.prob.p.semi
    system = only(filter(candidate -> candidate isa TrixiParticles.AbstractFluidSystem,
                         collect(semi.systems)))
    boundaries = production_contact_boundaries(semi)
    v_ode, u_ode = state
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    conjugate = copy(system.cache.wetted_area_density_conjugate)
    weights = [copy(boundary.boundary_model.cache.wetted_area_weight)
               for boundary in boundaries]
    zero_conjugate = zero(conjugate)
    zero_weights = map(zero, weights)

    function interaction(conjugate_values, weight_values)
        copyto!(system.cache.wetted_area_density_conjugate, conjugate_values)
        for (boundary, values) in zip(boundaries, weight_values)
            copyto!(boundary.boundary_model.cache.wetted_area_weight, values)
        end
        dv_ode = zero(v_ode)
        TrixiParticles.system_interaction!(dv_ode, v_ode, u_ode, semi)
        dv = TrixiParticles.wrap_v(dv_ode, system, semi)
        acceleration = Array(dv[1:3, :])
        reactions = [copy(boundary.boundary_model.cache.wetted_area_reaction)
                     for boundary in boundaries]
        return (; acceleration, reactions)
    end

    baseline = interaction(zero_conjugate, zero_weights)
    explicit = interaction(zero_conjugate, weights)
    density = interaction(conjugate, zero_weights)
    full = interaction(conjugate, weights)
    masses = system.mass
    explicit_force = (explicit.acceleration - baseline.acceleration) .* transpose(masses)
    density_force = (density.acceleration - baseline.acceleration) .* transpose(masses)
    full_force = (full.acceleration - baseline.acceleration) .* transpose(masses)

    explicit_resultant, explicit_scale = r4_force_resultant(explicit_force)
    density_resultant, density_scale = r4_force_resultant(density_force)
    full_resultant, full_scale = r4_force_resultant(full_force)
    wall_resultant = zeros(eltype(system), 3)
    wall_scale = zero(eltype(system))
    for reaction in explicit.reactions
        resultant, scale = r4_force_resultant(reaction)
        wall_resultant .+= resultant
        wall_scale += scale
    end
    explicit_reaction_residual = r4_relative_residual(explicit_resultant +
                                                      wall_resultant,
                                                      explicit_scale + wall_scale)
    density_resultant_residual = r4_relative_residual(density_resultant, density_scale)
    total_momentum_residual = r4_relative_residual(full_resultant + wall_resultant,
                                                   full_scale + wall_scale)
    force_consistency_residual = r4_relative_residual(vec(sum(full_force - explicit_force -
                                                              density_force; dims=2)),
                                                      full_scale + explicit_scale +
                                                      density_scale)
    raw_area = system.cache.wetted_area_raw_area[]
    corrected_area = system.cache.wetted_area[]
    raw_radius = sqrt(raw_area / pi)
    corrected_radius = sqrt(corrected_area / pi)
    area_derivative = raw_radius > eps(raw_radius) ? corrected_radius / raw_radius : 0.0
    contact_cache_bytes = wetted_area_contact_diagnostics(system.surface_normal_method.contact_model,
                                                          system,
                                                          first(boundaries)).cache_bytes
    return (; energy=system.cache.wetted_area_energy[], raw_area, corrected_area,
            area_derivative, explicit_force, density_force, full_force,
            explicit_resultant, density_resultant, wall_resultant,
            explicit_force_scale=explicit_scale, density_force_scale=density_scale,
            wall_force_scale=wall_scale,
            explicit_reaction_residual, density_resultant_residual,
            total_momentum_residual, force_consistency_residual,
            max_explicit_reaction_residual=explicit_reaction_residual,
            max_density_resultant_residual=density_resultant_residual,
            max_total_momentum_residual=total_momentum_residual,
            cache_bytes=contact_cache_bytes)
end

function production_directional_energy_gradient(result, diagnostics)
    semi = result.solution.prob.p.semi
    system = semi.systems[1]
    boundary_system = only(production_contact_boundaries(semi))
    v_ode, u_ode = result.solution.prob.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    v = TrixiParticles.wrap_v(v_ode, system, semi)
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    u_boundary = TrixiParticles.wrap_u(u_ode, boundary_system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    boundary_coordinates = Array(TrixiParticles.current_coordinates(u_boundary,
                                                                    boundary_system))
    density = collect(TrixiParticles.current_density(v, system))
    displacement = similar(coordinates)
    displacement_scale = max(maximum(abs, coordinates),
                             system.cache.reference_particle_spacing)
    for particle in TrixiParticles.eachparticle(system)
        displacement[1, particle] = -coordinates[1, particle] / displacement_scale
        displacement[2, particle] = -coordinates[2, particle] / displacement_scale
        displacement[3, particle] = 2coordinates[3, particle] / displacement_scale
    end

    density_rate = zeros(eltype(system), TrixiParticles.nparticles(system))
    TrixiParticles.foreach_point_neighbor(system, system, coordinates, coordinates, semi;
                                          points=TrixiParticles.each_integrated_particle(system),
                                          parallelization_backend=TrixiParticles.SerialBackend()) do particle,
                                                                                                     neighbor,
                                                                                                     pos_diff,
                                                                                                     distance
        gradient = TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance,
                                                        particle)
        mass_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
        density_rate[particle] += density[particle] / density[neighbor] * mass_b *
                                  dot(displacement[:, particle] -
                                      displacement[:, neighbor], gradient)
    end
    pairs = Tuple{Int, Int}[]
    TrixiParticles.foreach_point_neighbor(system, boundary_system, coordinates,
                                          boundary_coordinates, semi;
                                          points=TrixiParticles.each_integrated_particle(system),
                                          parallelization_backend=TrixiParticles.SerialBackend()) do particle,
                                                                                                     neighbor,
                                                                                                     pos_diff,
                                                                                                     distance
        push!(pairs, (particle, neighbor))
    end

    boundary_cache = boundary_system.boundary_model.cache
    function perturbed_energy(epsilon)
        colorfield = copy(boundary_cache.initial_colorfield)
        for (particle, neighbor) in pairs
            distance2 = zero(eltype(system))
            for dim in 1:3
                difference = coordinates[dim, particle] +
                             epsilon * displacement[dim, particle] -
                             boundary_coordinates[dim, neighbor]
                distance2 += difference^2
            end
            perturbed_density = density[particle] + epsilon * density_rate[particle]
            colorfield[neighbor] += system.mass[particle] / perturbed_density *
                                    TrixiParticles.smoothing_kernel(system,
                                                                    sqrt(distance2),
                                                                    particle)
        end
        raw_area = zero(eltype(system))
        for particle in TrixiParticles.eachparticle(boundary_system)
            measure = boundary_cache.wetted_area_surface_measure[particle]
            iszero(measure) && continue
            reference = boundary_cache.wetted_area_flooded_reference[particle]
            fraction = clamp(colorfield[particle] / reference, 0, 1)
            raw_area += measure * TrixiParticles.cubic_smoothstep(fraction)
        end
        raw_radius = sqrt(raw_area / pi)
        edge_shift = system.cache.wetted_area_normalized_edge_shift[] *
                     TrixiParticles.initial_smoothing_length(system)
        corrected_radius = max(raw_radius - edge_shift, zero(raw_radius))
        coefficient = TrixiParticles.wetted_area_coefficient(system.surface_tension,
                                                             system.surface_normal_method.contact_model)
        return -coefficient * pi * corrected_radius^2
    end

    epsilon = 1.0e-5system.cache.reference_particle_spacing
    energy_plus = perturbed_energy(epsilon)
    energy_minus = perturbed_energy(-epsilon)
    finite_difference = (energy_plus - energy_minus) / (2epsilon)
    explicit_derivative = zero(eltype(system))
    density_derivative = zero(eltype(system))
    for particle in TrixiParticles.eachparticle(system), dim in 1:3
        explicit_derivative -= diagnostics.explicit_force[dim, particle] *
                               displacement[dim, particle]
        density_derivative -= diagnostics.density_force[dim, particle] *
                              displacement[dim, particle]
    end
    analytic_derivative = explicit_derivative + density_derivative
    derivative_scale = max(abs(finite_difference), abs(analytic_derivative))
    relative_error = iszero(derivative_scale) ? zero(derivative_scale) :
                     abs(finite_difference - analytic_derivative) / derivative_scale
    return (; epsilon, energy_plus, energy_minus, finite_difference,
            analytic_derivative, explicit_derivative, density_derivative,
            relative_error, fluid_wall_pairs=length(pairs))
end

function run_production_static_gate(; output_path=PRODUCTION_STATIC_PATH)
    rows = NamedTuple[]
    for target in TARGET_ANGLES
        result = run_production_wetted_area_simulation(target, target; final_time=0.0)
        diagnostics = production_contact_rhs_diagnostics(result)
        gradient = production_directional_energy_gradient(result, diagnostics)
        push!(rows,
              r4_static_row("energy_gradient", target, target, result,
                            diagnostics; gradient))
    end
    angle_pairs = ((60.0, 90.0), (90.0, 60.0),
                   (90.0, 120.0), (120.0, 90.0))
    for (target, initial_angle) in angle_pairs
        control = run_production_wetted_area_simulation(target, initial_angle;
                                                        active=false, final_time=0.0)
        result = run_production_wetted_area_simulation(target, initial_angle;
                                                       final_time=0.0)
        diagnostics = production_contact_rhs_diagnostics(result)
        push!(rows,
              r4_static_row("force_sign", target, initial_angle, result,
                            diagnostics;
                            control_acceleration=control.shape_acceleration))
    end
    data = DataFrame(rows)
    data.mechanism = fill("wetted_area_production", nrow(data))
    CSV.write(output_path, data)
    println("Production wetted-area static gate: ", count(data.stage_pass), "/",
            nrow(data), "; max gradient error ",
            maximum(data[data.kind .== "energy_gradient", :gradient_relative_error]))
    require(all(data.stage_pass), "production wetted-area algebra/static gate failed")
    return data
end

function production_dynamic_row(kind, target, initial_angle, result,
                                initial_diagnostics, final_diagnostics; active=true,
                                resolution=1500, threshold=0.1, damping=4000.0,
                                final_time=0.01, repeat=1,
                                total_runtime=result.runtime)
    initial_error = result.initial_circle.angle - target
    final_error = result.circle.angle - target
    error_reduction = abs(initial_error) > eps() ?
                      1 - abs(final_error) / abs(initial_error) : NaN
    max_explicit_residual = active ?
                            max(initial_diagnostics.explicit_reaction_residual,
                                final_diagnostics.explicit_reaction_residual) : 0.0
    max_density_residual = active ?
                           max(initial_diagnostics.density_resultant_residual,
                               final_diagnostics.density_resultant_residual) : 0.0
    max_total_residual = active ?
                         max(initial_diagnostics.total_momentum_residual,
                             final_diagnostics.total_momentum_residual) : 0.0
    return (; kind=String(kind), target, initial_angle,
            mechanism=active ? "wetted_area_production" : "none",
            requested_particles=resolution, particle_count=result.particle_count,
            threshold, damping, final_time, repeat,
            initial_circle=result.initial_circle.angle,
            final_circle=result.circle.angle, initial_error, final_error,
            error_reduction, cap_angle=result.final.angle,
            normal_angle=result.measured_contact_angle,
            shape_acceleration=result.shape_acceleration,
            boundary_shape_acceleration=result.boundary_shape_acceleration,
            wetting_energy=active ? initial_diagnostics.energy : 0.0,
            explicit_reaction_residual=active ?
                                       initial_diagnostics.explicit_reaction_residual : 0.0,
            density_resultant_residual=active ?
                                       initial_diagnostics.density_resultant_residual : 0.0,
            total_momentum_residual=active ?
                                    initial_diagnostics.total_momentum_residual : 0.0,
            max_explicit_reaction_residual=max_explicit_residual,
            max_density_resultant_residual=max_density_residual,
            max_total_momentum_residual=max_total_residual,
            force_consistency_residual=active ?
                                       max(initial_diagnostics.force_consistency_residual,
                                           final_diagnostics.force_consistency_residual) :
                                       0.0,
            below_wall=result.below_wall, density_min=result.density_range[1],
            density_max=result.density_range[2], rms_speed=result.rms_speed,
            settled=result.settled, accepted_steps=result.accepted_steps,
            rejected_steps=result.rejected_steps,
            rejected_fraction=result.rejected_fraction,
            minimum_dt=result.minimum_dt, dt_reference=result.dt_reference,
            eta_p01=result.eta_p01, eta_median=result.eta_median,
            eta_tail_head=result.eta_tail_head, solver_runtime=result.runtime,
            total_runtime,
            cache_bytes=Base.summarysize(result.solution.prob.p.semi.systems[1].cache),
            contact_cache_bytes=active ? initial_diagnostics.cache_bytes : 0)
end

function run_production_dynamic_case(kind, target, initial_angle; active=true,
                                     resolution=1500, threshold=0.1,
                                     damping=4000.0, final_time=0.01, repeat=1)
    result = nothing
    initial_diagnostics = nothing
    final_diagnostics = nothing
    total_runtime = @elapsed begin
        result = run_production_wetted_area_simulation(target, initial_angle; active,
                                                       resolution, threshold, damping,
                                                       final_time)
        if active
            initial_diagnostics = production_contact_rhs_diagnostics(result;
                                                                     state=result.solution.prob.u0.x)
            final_diagnostics = production_contact_rhs_diagnostics(result;
                                                                   state=last(result.solution.u).x)
        end
    end
    return production_dynamic_row(kind, target, initial_angle, result,
                                  initial_diagnostics, final_diagnostics; active,
                                  resolution, threshold, damping, final_time, repeat,
                                  total_runtime)
end

function production_perturbation_data(; final_time=0.01)
    angle_pairs = ((60.0, 90.0), (90.0, 60.0),
                   (90.0, 120.0), (120.0, 90.0))
    rows = [run_production_dynamic_case(:perturbation, target, initial; active,
                                        final_time)
            for (target, initial) in angle_pairs for active in (false, true)]
    data = DataFrame(rows)
    data.control_error_reduction = fill(NaN, nrow(data))
    data.contact_shape_acceleration = fill(NaN, nrow(data))
    data.motion_toward_target = falses(nrow(data))
    data.beats_control = falses(nrow(data))
    data.acceleration_toward_target = falses(nrow(data))
    data.reaction_pass = falses(nrow(data))
    data.response_pass = falses(nrow(data))
    for (target, initial) in angle_pairs
        group = findall((data.target .== target) .& (data.initial_angle .== initial))
        control = only(filter(index -> data.mechanism[index] == "none", group))
        direction = sign(target - initial)
        for index in group
            data.control_error_reduction[index] = data.error_reduction[control]
            data.contact_shape_acceleration[index] = data.shape_acceleration[index] -
                                                     data.shape_acceleration[control]
            data.motion_toward_target[index] = direction *
                                               (data.final_circle[index] -
                                                data.initial_circle[index]) > 0
            data.beats_control[index] = data.mechanism[index] == "none" ||
                                        data.error_reduction[index] >
                                        data.error_reduction[control]
            data.acceleration_toward_target[index] = data.mechanism[index] == "none" ||
                                                     direction *
                                                     data.contact_shape_acceleration[index] >
                                                     0
            data.reaction_pass[index] = data.max_explicit_reaction_residual[index] <=
                                        1.0e-12 &&
                                        data.max_density_resultant_residual[index] <=
                                        1.0e-12 &&
                                        data.max_total_momentum_residual[index] <= 1.0e-12
            data.response_pass[index] = data.mechanism[index] == "none" ||
                                        (data.error_reduction[index] > 0 &&
                                         data.motion_toward_target[index] &&
                                         data.beats_control[index] &&
                                         data.acceleration_toward_target[index] &&
                                         data.below_wall[index] == 0 &&
                                         980 <= data.density_min[index] <= 1020 &&
                                         980 <= data.density_max[index] <= 1020 &&
                                         data.rejected_fraction[index] <= 0.25 &&
                                         data.reaction_pass[index])
        end
    end
    classify_r4_wetted_area_perturbation!(data)
    return data
end

function run_production_perturbation_gate(; output_path=PRODUCTION_PERTURBATION_PATH,
                                          final_time=0.01)
    require(isfile(PRODUCTION_STATIC_PATH), "run `production_static` first")
    static = CSV.read(PRODUCTION_STATIC_PATH, DataFrame)
    require(all(static.stage_pass), "production static evidence does not pass")
    data = production_perturbation_data(; final_time)
    CSV.write(output_path, data)
    candidates = data[data.mechanism .== "wetted_area_production", :]
    println("Production wetted-area perturbation: formulation-consistent ",
            count(candidates.formulation_response_pass), "/4")
    return data
end

function run_production_perturbation_extension(;
                                               output_path=PRODUCTION_PERTURBATION_EXTENDED_PATH,
                                               final_time=0.02)
    require(final_time == 0.02, "the production replay keeps the frozen 0.02 s extension")
    require(isfile(PRODUCTION_PERTURBATION_PATH),
            "run `production_perturbation` first")
    data = production_perturbation_data(; final_time)
    CSV.write(output_path, data)
    candidates = data[data.mechanism .== "wetted_area_production", :]
    println("Production wetted-area extended perturbation: ",
            count(candidates.formulation_response_pass), "/4")
    require(nrow(candidates) == 4 && all(candidates.formulation_response_pass),
            "production extended perturbation gate failed")
    return data
end

function require_production_perturbation_gate()
    require(isfile(PRODUCTION_PERTURBATION_EXTENDED_PATH),
            "run `production_perturbation_extended` first")
    data = CSV.read(PRODUCTION_PERTURBATION_EXTENDED_PATH, DataFrame)
    candidates = data[data.mechanism .== "wetted_area_production", :]
    require(nrow(candidates) == 4 && all(candidates.formulation_response_pass),
            "production perturbation evidence does not pass")
    return data
end

function run_production_threshold_gate(; output_path=PRODUCTION_THRESHOLD_PATH)
    require_production_perturbation_gate()
    rows = [run_production_dynamic_case(:threshold, target, target)
            for target in TARGET_ANGLES]
    data = DataFrame(rows)
    data.reaction_pass = (data.max_explicit_reaction_residual .<= 1.0e-12) .&
                         (data.max_density_resultant_residual .<= 1.0e-12) .&
                         (data.max_total_momentum_residual .<= 1.0e-12) .&
                         (data.force_consistency_residual .<= 1.0e-12)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                data.reaction_pass
    CSV.write(output_path, data)
    println("Production wetted-area threshold gate: ", count(data.pass), "/5")
    require(all(data.pass), "production threshold replay failed")
    return data
end

function require_production_threshold_gate()
    require(isfile(PRODUCTION_THRESHOLD_PATH), "run `production_threshold` first")
    data = CSV.read(PRODUCTION_THRESHOLD_PATH, DataFrame)
    require(nrow(data) == 5 && all(data.pass),
            "production threshold evidence does not pass")
    return data
end

function run_production_timestep_gate(; output_path=PRODUCTION_TIMESTEP_PATH)
    require_production_threshold_gate()
    rows = [run_production_dynamic_case(:timestep, target, target; resolution)
            for (target, resolution) in ((90.0, 1500), (30.0, 3000))]
    data = DataFrame(rows)
    data.reaction_pass = (data.max_explicit_reaction_residual .<= 1.0e-12) .&
                         (data.max_density_resultant_residual .<= 1.0e-12) .&
                         (data.max_total_momentum_residual .<= 1.0e-12) .&
                         (data.force_consistency_residual .<= 1.0e-12)
    data.pass = (data.eta_p01 .>= 0.05) .& (data.eta_tail_head .>= 0.5) .&
                (data.rejected_fraction .<= 0.25) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                data.reaction_pass
    CSV.write(output_path, data)
    println("Production wetted-area timestep gate: ", count(data.pass), "/2")
    require(all(data.pass), "production timestep gate failed")
    return data
end

function require_production_timestep_gate()
    require(isfile(PRODUCTION_TIMESTEP_PATH), "run `production_timestep` first")
    data = CSV.read(PRODUCTION_TIMESTEP_PATH, DataFrame)
    require(nrow(data) == 2 && all(data.pass),
            "production timestep evidence does not pass")
    return data
end

function run_production_cost_cases(; output_path=PRODUCTION_COST_PATH, target=90.0)
    require_production_timestep_gate()
    for active in (false, true)
        run_production_dynamic_case(:warmup, target, target; active,
                                    resolution=200, final_time=0.001)
    end
    orders = ((false, true), (true, false), (false, true))
    rows = NamedTuple[]
    for (repeat, order) in enumerate(orders), active in order
        push!(rows, run_production_dynamic_case(:cost, target, target; active, repeat))
    end
    data = DataFrame(rows)
    control_median = median(data[data.mechanism .== "none", :solver_runtime])
    data.normalized_runtime = data.solver_runtime ./ control_median
    CSV.write(output_path, data)
    for mechanism in ("none", "wetted_area_production")
        values = data[data.mechanism .== mechanism, :solver_runtime]
        center = median(values)
        @printf("%-26s median %.3f s MAD %.3f s overhead %.3f\n",
                mechanism, center, median(abs.(values .- center)),
                center / control_median)
    end
    if target == 60
        active_median = median(data[data.mechanism .== "wetted_area_production",
                                    :solver_runtime])
        require(active_median / control_median <= 1.2,
                "production active wetting overhead exceeds 20%")
    end
    require(all(isfinite, data.solver_runtime), "non-finite production cost measurement")
    return data
end

function require_production_cost_cases()
    require(isfile(PRODUCTION_COST_PATH), "run `production_cost` first")
    require(isfile(PRODUCTION_ACTIVE_COST_PATH), "run `production_cost_active` first")
    zero_path = CSV.read(PRODUCTION_COST_PATH, DataFrame)
    active_path = CSV.read(PRODUCTION_ACTIVE_COST_PATH, DataFrame)
    control_median = median(active_path[active_path.mechanism .== "none",
                                        :solver_runtime])
    active_median = median(active_path[active_path.mechanism .== "wetted_area_production",
                                       :solver_runtime])
    require(nrow(zero_path) == 6 && nrow(active_path) == 6 &&
                active_median / control_median <= 1.2,
            "production cost evidence is incomplete or exceeds the active gate")
    return (; zero_path, active_path)
end

function run_production_selected_matrix(; output_path=PRODUCTION_SELECTED_PATH)
    require_production_cost_cases()
    rows = [run_production_dynamic_case(:selected, target, target; resolution)
            for resolution in RESOLUTIONS for target in TARGET_ANGLES]
    data = DataFrame(rows)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                (data.eta_p01 .>= 0.05) .& (data.eta_tail_head .>= 0.5) .&
                (data.max_explicit_reaction_residual .<= 1.0e-12) .&
                (data.max_density_resultant_residual .<= 1.0e-12) .&
                (data.max_total_momentum_residual .<= 1.0e-12) .&
                (data.force_consistency_residual .<= 1.0e-12)
    CSV.write(output_path, data)
    println("Production wetted-area selected matrix: ", count(data.pass), "/15")
    require(all(data.pass), "production selected matrix failed")
    return data
end

function run_production_sensitivity(; output_path=PRODUCTION_SENSITIVITY_PATH)
    require(isfile(PRODUCTION_SELECTED_PATH), "run `production_selected` first")
    selected = CSV.read(PRODUCTION_SELECTED_PATH, DataFrame)
    require(nrow(selected) == 15 && all(selected.pass),
            "production selected matrix does not pass")
    rows = [run_production_dynamic_case(:sensitivity, 90.0, 90.0;
                                        threshold, damping)
            for threshold in (0.0, 0.1) for damping in (2000.0, 8000.0)]
    data = DataFrame(rows)
    angle_span = maximum(data.final_circle) - minimum(data.final_circle)
    data.pass = (abs.(data.final_error) .<= 5) .& (data.below_wall .== 0) .&
                (data.density_min .>= 980) .& (data.density_max .<= 1020) .&
                (data.rms_speed .< 5.0e-3) .& (data.rejected_fraction .<= 0.25) .&
                (data.max_total_momentum_residual .<= 1.0e-12) .&
                (data.force_consistency_residual .<= 1.0e-12) .& (angle_span .<= 1)
    data.angle_span = fill(angle_span, nrow(data))
    CSV.write(output_path, data)
    println("Production wetted-area sensitivity: ", count(data.pass),
            "/4; span ", angle_span, " deg")
    require(all(data.pass), "production sensitivity matrix failed")
    return data
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: contact_angle_decision.jl " *
          "score|r4_static|r4_perturbation|r4_perturbation_extended|r4_threshold|r4_timestep|r4_cost|r4_cost_active|r4_selected|r4_sensitivity|production_static|production_perturbation|production_perturbation_extended|production_threshold|production_timestep|production_cost|production_cost_active|production_selected|production_sensitivity")
    mode = Symbol(ARGS[1])
    if mode == :score
        score_phase2()
    elseif mode == :r4_static
        run_r4_static_gate()
    elseif mode == :r4_perturbation
        run_r4_perturbation_gate()
    elseif mode == :r4_perturbation_extended
        run_r4_perturbation_extension()
    elseif mode == :r4_threshold
        run_r4_threshold_gate()
    elseif mode == :r4_timestep
        run_r4_timestep_gate()
    elseif mode == :r4_cost
        run_r4_cost_cases()
    elseif mode == :r4_cost_active
        run_r4_cost_cases(; output_path=R4_ACTIVE_COST_PATH, target=60.0)
    elseif mode == :r4_selected
        run_r4_selected_matrix()
    elseif mode == :r4_sensitivity
        run_r4_sensitivity()
    elseif mode == :production_static
        run_production_static_gate()
    elseif mode == :production_perturbation
        run_production_perturbation_gate()
    elseif mode == :production_perturbation_extended
        run_production_perturbation_extension()
    elseif mode == :production_threshold
        run_production_threshold_gate()
    elseif mode == :production_timestep
        run_production_timestep_gate()
    elseif mode == :production_cost
        run_production_cost_cases()
    elseif mode == :production_cost_active
        run_production_cost_cases(; output_path=PRODUCTION_ACTIVE_COST_PATH,
                                  target=60.0)
    elseif mode == :production_selected
        run_production_selected_matrix()
    elseif mode == :production_sensitivity
        run_production_sensitivity()
    else
        error("unknown mode: $mode")
    end
end
