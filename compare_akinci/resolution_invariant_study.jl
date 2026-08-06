using Printf
using TrixiParticles

include(joinpath(@__DIR__, "surface_tension_calibration.jl"))

function resolution_study_model(model_name, coefficient, calibration_factor)
    if model_name == "legacy"
        return SurfaceTensionAkinci(; surface_tension_coefficient=coefficient)
    elseif model_name == "cohesion"
        return CohesionForceAkinci(; surface_tension_coefficient=coefficient)
    elseif model_name == "cohesion_physical"
        return SurfaceTensionAkinciCohesionPhysical(;
                                                    surface_tension_coefficient=coefficient,
                                                    reference_density=1000.0)
    elseif model_name == "invariant"
        reference_spacing = cbrt(1.0e-6 / 750)
        reference_support_radius = 2.8reference_spacing
        return SurfaceTensionAkinciResolutionInvariant(;
                                                       surface_tension_coefficient=coefficient,
                                                       calibration_factor,
                                                       reference_support_radius)
    end
    throw(ArgumentError("unknown model `$model_name`"))
end

function run_resolution_laplace_study(model_name, target_particle_count;
                                      coefficient=1.0,
                                      calibration_factor=1.0,
                                      final_time=0.02)
    model = resolution_study_model(model_name, coefficient, calibration_factor)
    result = laplace_pressure_series(model; final_time,
                                     base_target_particle_count=target_particle_count)
    particle_spacing = cbrt(1.0e-6 / target_particle_count)
    support_radius = 2.8particle_spacing
    internal_coefficient = if model isa SurfaceTensionAkinciResolutionInvariant
        coefficient * akinci_resolution_scale(model, support_radius)
    elseif model isa SurfaceTensionAkinciCohesionPhysical
        TrixiParticles.akinci_physical_cohesion_coefficient(model, support_radius)
    else
        coefficient
    end
    @printf("resolution model=%s target_n=%d dx=%.8g H=%.8g internal_gamma=%.8g sigma=%.8g p_bulk=%.8g residual=%.8g\n",
            model_name, target_particle_count, particle_spacing, support_radius,
            internal_coefficient, result.surface_tension, result.bulk_pressure,
            result.residual_rms)
    return result
end

function run_resolution_stiffness_study(target_particle_count)
    cohesion = rayleigh_stiffness(CohesionForceAkinci(;
                                                      surface_tension_coefficient=1.0);
                                  target_particle_count)
    full = rayleigh_stiffness(SurfaceTensionAkinci(;
                                                   surface_tension_coefficient=1.0);
                              target_particle_count)
    particle_spacing = cbrt(1.0e-6 / target_particle_count)
    support_radius = 2.8particle_spacing
    normal = full.inferred_surface_tension - cohesion.inferred_surface_tension
    @printf("stiffness target_n=%d dx=%.8g H=%.8g R_over_H=%.8g cohesion=%.8g normal=%.8g full=%.8g\n",
            target_particle_count, particle_spacing, support_radius,
            full.equivalent_radius / support_radius,
            cohesion.inferred_surface_tension, normal,
            full.inferred_surface_tension)
    return (; cohesion, normal, full)
end

@inline function cohesion_potential_shape_akinci(radius_ratio)
    x = radius_ratio
    polynomial(y) = y^4 / 4 - 3y^5 / 5 + y^6 / 2 - y^7 / 7
    integral = if x > 0.5
        1 / 140 - polynomial(x)
    else
        1 / 140 + polynomial(0.5) - 0.5 / 64 -
        2polynomial(x) + x / 64
    end
    return 32 / pi * integral
end

function cohesion_surface_energy(target_particle_count)
    initial_condition = deformed_drop(; stretch=1.0, target_particle_count)
    coordinates = initial_condition.coordinates
    particle_spacing = initial_condition.particle_spacing
    support_radius = 2.8particle_spacing
    support_ratio = support_radius / particle_spacing
    particle_mass = first(initial_condition.mass)
    origin = coordinates[:, 1]
    particle_at = Dict{NTuple{3, Int}, Bool}()
    for particle in axes(coordinates, 2)
        key = ntuple(dimension -> round(Int,
                                        (coordinates[dimension, particle] -
                                         origin[dimension]) / particle_spacing), 3)
        particle_at[key] = true
    end

    limit = ceil(Int, support_ratio)
    offsets = NTuple{3, Int}[]
    for i in (-limit):limit, j in (-limit):limit, k in (-limit):limit
        iszero(i) && iszero(j) && iszero(k) && continue
        # One representative of each undirected lattice bond.
        (i > 0 || (iszero(i) && j > 0) ||
         (iszero(i) && iszero(j) && k > 0)) || continue
        distance_lattice = sqrt(i^2 + j^2 + k^2)
        distance_lattice < support_ratio || continue
        push!(offsets, (i, j, k))
    end

    n_particles = size(coordinates, 2)
    excess_energy = 0.0
    for offset in offsets
        existing_pairs = 0
        for key in keys(particle_at)
            neighbor_key = ntuple(dimension -> key[dimension] + offset[dimension], 3)
            existing_pairs += haskey(particle_at, neighbor_key)
        end
        distance_ratio = sqrt(sum(abs2, offset)) / support_ratio
        potential = -particle_mass^2 / support_radius^2 *
                    cohesion_potential_shape_akinci(distance_ratio)
        excess_energy += (n_particles - existing_pairs) * -potential
    end

    volume = sum(initial_condition.mass) / 1000.0
    radius = cbrt(3volume / (4pi))
    area = 4pi * radius^2
    surface_tension = excess_energy / area
    moment = surface_tension / (1000.0^2 * support_radius^2)
    @printf("cohesion_energy target_n=%d actual_n=%d R_over_H=%.8g sigma=%.8g moment=%.12g lattice_moment=%.12g\n",
            target_particle_count, n_particles, radius / support_radius,
            surface_tension, moment, AKINCI_COHESION_LATTICE_MOMENT_3D_H28)
    return (; surface_tension, moment, radius, support_radius, n_particles)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 2 && ARGS[1] == "stiffness"
        run_resolution_stiffness_study(parse(Int, ARGS[2]))
    elseif length(ARGS) == 2 && ARGS[1] == "cohesion_energy"
        cohesion_surface_energy(parse(Int, ARGS[2]))
    else
        length(ARGS) in (3, 4, 5) ||
            error("usage: resolution_invariant_study.jl MODEL TARGET_PARTICLE_COUNT " *
                  "FINAL_TIME [COEFFICIENT [CALIBRATION_FACTOR]]")
        model_name = ARGS[1]
        target_particle_count = parse(Int, ARGS[2])
        final_time = parse(Float64, ARGS[3])
        coefficient = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.0
        calibration_factor = length(ARGS) == 5 ? parse(Float64, ARGS[5]) : 1.0
        run_resolution_laplace_study(model_name, target_particle_count;
                                     coefficient, calibration_factor, final_time)
    end
end
