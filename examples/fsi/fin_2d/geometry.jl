# Real blade width profile, where `x` is the distance from the blade attachment in meters.
function blade_width(x, max_blade_width)
    if x > 0.12
        return max_blade_width
    end
    width = -1.199 * x^2 + 0.346 * x + 0.167
    return clamp(width, 7.5e-2, max_blade_width)
end

# The 2D model represents the blade width in the unmodeled third dimension.
# The foot pocket is narrower than the 19 cm blade, so we scale its modulus accordingly.
function foot_pocket_width(x_relative)
    foot_pocket_width_at_right_end = 2e-2
    foot_pocket_full_width = 10e-2
    foot_pocket_width_ramp_length = 15e-2

    ramp_coordinate = clamp(-x_relative / foot_pocket_width_ramp_length, 0.0, 1.0)
    width_range = foot_pocket_full_width - foot_pocket_width_at_right_end
    return foot_pocket_width_at_right_end + ramp_coordinate * width_range
end

function foot_pocket_height_top(x_normalized)
    # Linear approximation of the top edge of the foot pocket.
    # This is only used to limit the material interface blending width, so it doesn't
    # need to be exact, and it can be clamped once it's larger than the blending width.
    return clamp(-0.086 * x_normalized / 0.135, 0.0, 0.08)
end

function foot_pocket_height_bottom(x_normalized)
    # Cubic polynomial fitted to the bottom edge of the foot pocket.
    # This is only used to limit the material interface blending width, so it doesn't
    # need to be exact, and it can be clamped once it's larger than the blending width.
    x = clamp(x_normalized, -0.3, -0.1)
    return clamp(2.09 * x^3 + 1.25 * x^2 + 0.135 * x + 0.003, 0.0, 0.015)
end

function is_clamped_structure_particle(coordinates, particle, simulate_foot_pocket)
    x = coordinates[1, particle] - center[1]

    if simulate_foot_pocket
        y = coordinates[2, particle] - center[2]
        rigid_elastic_split_x = -0.28
        rounding_radius = 0.02
        x_circle_center = rigid_elastic_split_x - rounding_radius

        x > rigid_elastic_split_x && return false
        x <= x_circle_center && return true

        # At x = x_circle_center, the DXF intersects the top and bottom surfaces at
        # y = 0.06666 and y = -0.020261, respectively. The circle centers are
        # one radius inside the foot pocket so that each transition is a quarter circle.
        rounding_start_top = 0.04666
        rounding_start_bottom = -0.000261

        circle_y_offset = sqrt(max(rounding_radius^2 - (x - x_circle_center)^2, 0.0))

        y > rounding_start_top + circle_y_offset && return false
        y < rounding_start_bottom - circle_y_offset && return false

        return true
    end

    # The elastic part of the blade starts at x = 0 in the fin coordinate system.
    return x <= 0.0
end

function artificial_material_properties(x)
    x_relative = x - center[1]

    real_blade_thickness = real_thickness(x_relative / blade_length)
    real_width = blade_width(x_relative, max_blade_width)
    real_flexural_rigidity = real_modulus * real_width * real_blade_thickness^3 / 12

    foot_pocket_width_ratio = foot_pocket_width(x_relative) / max_blade_width
    blade_width_ratio = blade_width(x_relative, max_blade_width) / max_blade_width

    # Scale real densities to account for the artificially thickened blade and the 3D width.
    real_blade_density = 1600.0
    blade_density = blade_width_ratio * real_blade_density *
                    real_blade_thickness / artificial_blade_thickness
    real_foot_pocket_density = 1000.0
    foot_pocket_density = foot_pocket_width_ratio * real_foot_pocket_density

    # Scale modulus to maintain the correct flexural rigidity.
    blade_modulus = real_flexural_rigidity *
                    12 / (max_blade_width * artificial_blade_thickness^3)
    foot_pocket_modulus = foot_pocket_width_ratio * real_modulus_foot_pocket

    return (; blade_density, foot_pocket_density, blade_modulus, foot_pocket_modulus)
end

function apply_material_properties!(structure, simulate_foot_pocket)
    for particle in 1:nparticles(structure)
        x = structure.coordinates[1, particle]
        y = structure.coordinates[2, particle]

        if simulate_foot_pocket
            structure.density[particle] = structure_density(x, y)
        else
            structure.density[particle] = artificial_material_properties(x).blade_density
        end
        structure.mass[particle] = structure.density[particle] * particle_spacing^2
    end
end

function artificial_modulus(structure, simulate_foot_pocket)
    if simulate_foot_pocket
        return [structure_modulus(structure.coordinates[1, particle],
                                  structure.coordinates[2, particle])
                for particle in 1:nparticles(structure)]
    end

    return [artificial_material_properties(structure.coordinates[1, particle]).blade_modulus
            for particle in 1:nparticles(structure)]
end

function clamped_structure_particles(structure, simulate_foot_pocket)
    return findall(particle -> is_clamped_structure_particle(structure.coordinates,
                                                             particle,
                                                             simulate_foot_pocket),
                   1:nparticles(structure))
end

# The TLSPH method does not support discontinuities in density or modulus,
# so we apply a log-linear blending at the discontinuity between the blade
# and the foot pocket.
function structure_density(x, y)
    properties = artificial_material_properties(x)
    alpha = blade_blend_alpha(x, y)

    return log_linear_blend(properties.blade_density, properties.foot_pocket_density, alpha)
end

function structure_modulus(x, y)
    properties = artificial_material_properties(x)
    alpha = blade_blend_alpha(x, y)

    return log_linear_blend(properties.blade_modulus, properties.foot_pocket_modulus, alpha)
end

@inline function log_linear_blend(left_value, right_value, alpha)
    alpha <= 0 && return left_value
    alpha >= 1 && return right_value

    return exp(log(left_value) + alpha * (log(right_value) - log(left_value)))
end

@inline function blade_blend_alpha(x, y)
    x_relative = x - center[1]
    y_relative = y - center[2]
    distance_from_blade_center = abs(y_relative)

    # There is no material interface along the free part of the blade.
    x_relative >= 0 && return 0.0

    # Blend the material interface around the edge of the artificially thick blade.
    # The discontinuity is half a particle spacing beyond the outer blade particles.
    material_discontinuity_distance = artificial_blade_thickness / 2 + particle_spacing / 2

    foot_pocket_height = if y_relative >= 0
        foot_pocket_height_top(x_relative)
    else
        foot_pocket_height_bottom(x_relative)
    end
    inner_width, outer_width = local_material_blend_widths(foot_pocket_height,
                                                           material_discontinuity_distance)

    inner_edge = material_discontinuity_distance - inner_width
    outer_edge = material_discontinuity_distance + outer_width

    distance_from_blade_center <= inner_edge && return 0.0
    distance_from_blade_center >= outer_edge && return 1.0

    # Linear ramp from from inner edge to outer edge.
    alpha = (distance_from_blade_center - inner_edge) / (outer_edge - inner_edge)
    return clamp(alpha, 0, 1)
end

@inline function local_material_blend_widths(foot_pocket_height,
                                             material_discontinuity_distance)
    outer_width = artificial_blade_thickness * 5 / 6
    inner_width = artificial_blade_thickness / 6

    # If the surrounding foot pocket is too thin, reduce the blending widths accordingly.
    available_outer_height = max(foot_pocket_height - material_discontinuity_distance, 0.0)
    height_scale = min(available_outer_height / outer_width, 1.0)

    outer_width *= height_scale
    inner_width *= height_scale

    return inner_width, outer_width
end
