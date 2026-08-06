using Printf
using TrixiParticles

include(joinpath(@__DIR__, "simulate_delta_sph_wetting.jl"))
include(joinpath(@__DIR__, "surface_model_variants.jl"))

function wetting_model_config(model_name, parameter, final_time;
                              contact_angle=180.0,
                              total_surface_tension=nothing,
                              cohesion_coefficient=0.0,
                              target_particle_count=750)
    config = delta_sph_config("wetting_no", final_time; delta=0.1,
                              target_particle_count)
    particle_spacing = cbrt(1.0e-6 / target_particle_count)
    smoothing_kernel = config.kwargs.smoothing_kernel
    smoothing_length = config.kwargs.smoothing_length
    support_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
    reference_density = 1000.0

    surface_tension, normal_method,
    wall_coefficient = if model_name == "akinci"
        model = SurfaceTensionAkinci(; surface_tension_coefficient=parameter)
        model, ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf), 0.0
    elseif model_name == "akinci_invariant"
        calibration_factor = something(total_surface_tension, 1.0)
        reference_spacing = cbrt(1.0e-6 / 750)
        reference_support_radius = 2.8reference_spacing
        model = SurfaceTensionAkinciResolutionInvariant(;
                                                        surface_tension_coefficient=parameter,
                                                        calibration_factor,
                                                        reference_support_radius)
        model, ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf), 0.0
    elseif model_name == "cohesion"
        coefficient = akinci_cohesion_coefficient(parameter, reference_density,
                                                  support_radius)
        CohesionForceAkinci(; surface_tension_coefficient=coefficient), nothing, 0.0
    elseif model_name == "cohesion_physical"
        model = SurfaceTensionAkinciCohesionPhysical(;
                                                     surface_tension_coefficient=parameter,
                                                     reference_density)
        wall_ratio = (1 + cosd(contact_angle)) / 2
        model, nothing, wall_ratio
    elseif model_name == "akinci_wall"
        isnothing(total_surface_tension) &&
            throw(ArgumentError("`akinci_wall` requires `total_surface_tension`"))
        wall = akinci_wall_cohesion_coefficient(total_surface_tension, contact_angle,
                                                reference_density, support_radius)
        model = SurfaceTensionAkinciWallCohesion(;
                                                 surface_tension_coefficient=parameter,
                                                 wall_cohesion_coefficient=wall)
        model, ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf), wall
    elseif model_name == "akinci_wall_direct"
        isnothing(total_surface_tension) &&
            throw(ArgumentError("`akinci_wall_direct` requires an absolute wall coefficient"))
        wall = total_surface_tension
        model = SurfaceTensionAkinciWallCohesion(;
                                                 surface_tension_coefficient=parameter,
                                                 wall_cohesion_coefficient=wall)
        model, ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf), wall
    elseif model_name == "akinci_contact"
        isnothing(total_surface_tension) &&
            throw(ArgumentError("`akinci_contact` requires a contact-line surface tension"))
        model = SurfaceTensionAkinciContactLine(;
                                                surface_tension_coefficient=parameter,
                                                contact_line_surface_tension=total_surface_tension,
                                                contact_angle, reference_density)
        model, ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf), 0.0
    elseif model_name == "hybrid"
        wall = akinci_wall_cohesion_coefficient(parameter, contact_angle,
                                                reference_density, support_radius)
        model = SurfaceTensionMorrisAkinci(;
                                           surface_tension_coefficient=parameter,
                                           cohesion_coefficient,
                                           wall_cohesion_coefficient=wall)
        normal = ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf,
                                         interface_threshold=0.01,
                                         ideal_density_threshold=0.95)
        model, normal, wall
    elseif model_name == "morris"
        model = SurfaceTensionMorris(; surface_tension_coefficient=parameter)
        normal = ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf,
                                         interface_threshold=0.01,
                                         ideal_density_threshold=0.95)
        model, normal, 0.0
    elseif model_name == "momentum_morris"
        model = SurfaceTensionMomentumMorris(; surface_tension_coefficient=parameter)
        normal = ColorfieldSurfaceNormal(; interface_threshold=0.01,
                                         ideal_density_threshold=0.95,
                                         contact_angle)
        model, normal, 0.0
    else
        throw(ArgumentError("unknown model `$model_name`"))
    end

    boundary_adhesion = model_name == "cohesion_physical" ? wall_coefficient : 0.0
    kwargs = merge(config.kwargs,
                   (; surface_tension, surface_normal_method=normal_method,
                    adhesion_coefficient=boundary_adhesion))
    @printf("model=%s parameter=%.8g contact_angle=%.3f wall_coefficient=%.8g support_radius=%.8g particles=%d\n",
            model_name, parameter, contact_angle,
            wall_coefficient, support_radius, target_particle_count)
    return merge(config, (; name=(config.name * "_" * model_name), kwargs))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (4, 5, 6, 7) ||
        error("usage: investigate_wetting_models.jl MODEL PARAMETER OUTPUT.jls " *
              "FINAL_TIME [CONTACT_ANGLE [EXTRA [TARGET_PARTICLE_COUNT]]]")
    model_name = ARGS[1]
    parameter = parse(Float64, ARGS[2])
    output = ARGS[3]
    final_time = parse(Float64, ARGS[4])
    contact_angle = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 180.0
    extra = length(ARGS) == 6 ? parse(Float64, ARGS[6]) : nothing
    if length(ARGS) == 7
        extra = parse(Float64, ARGS[6])
    end
    target_particle_count = length(ARGS) == 7 ? parse(Int, ARGS[7]) : 750
    options = if model_name in ("akinci_invariant", "akinci_wall",
                                "akinci_wall_direct", "akinci_contact")
        (; contact_angle, total_surface_tension=extra, target_particle_count)
    elseif model_name == "hybrid"
        (; contact_angle, cohesion_coefficient=something(extra, 0.0),
         target_particle_count)
    else
        (; contact_angle, target_particle_count)
    end
    config = wetting_model_config(model_name, parameter, final_time; options...)
    write_snapshot(config, output)
end
