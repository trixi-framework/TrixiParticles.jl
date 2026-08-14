# Compare all Akinci surface tension models across the supported SPH formulations.
using TrixiParticles
using TrixiParticles.CSV
using TrixiParticles.DataFrames
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSymplecticRK
using Statistics: mean, std

baseline_particles_per_dimension = 6
cube_side_length = 0.01
fluid_density = 1000.0
sound_speed = 20.0
cfl_number = 0.2
tspan = (0.0, 1.0)
analysis_interval = 0.01
winner_window_fraction = 0.2
iisph_min_iterations = 2
iisph_max_iterations = 10
run_resolution_study = true
resolution_levels = (4, 6, 8)
write_results = true
print_results = true

function support_radius_cv(coordinates; n_directions=512)
    centered = coordinates .- mean(coordinates; dims=2)
    golden_angle = pi * (3 - sqrt(5))
    support_radii = Vector{eltype(coordinates)}(undef, n_directions)

    for index in 1:n_directions
        z = 1 - 2 * (index - 0.5) / n_directions
        radius = sqrt(1 - z^2)
        azimuth = (index - 1) * golden_angle
        direction = (radius * cos(azimuth), radius * sin(azimuth), z)
        support_radii[index] = maximum(particle -> direction[1] * centered[1, particle] +
                                                   direction[2] * centered[2, particle] +
                                                   direction[3] * centered[3, particle],
                                       axes(centered, 2))
    end

    return std(support_radii; corrected=false) / mean(support_radii)
end

function particle_spacing_cv(coordinates)
    n_particles = size(coordinates, 2)
    n_particles > 1 || return zero(eltype(coordinates))
    nearest_neighbor_distances = Vector{eltype(coordinates)}(undef, n_particles)

    for particle in axes(coordinates, 2)
        minimum_distance_squared = typemax(eltype(coordinates))
        for neighbor in axes(coordinates, 2)
            particle == neighbor && continue
            distance_squared = zero(eltype(coordinates))
            for dimension in axes(coordinates, 1)
                difference = coordinates[dimension, particle] -
                             coordinates[dimension, neighbor]
                distance_squared = muladd(difference, difference, distance_squared)
            end
            minimum_distance_squared = min(minimum_distance_squared, distance_squared)
        end
        nearest_neighbor_distances[particle] = sqrt(minimum_distance_squared)
    end

    return std(nearest_neighbor_distances; corrected=false) /
           mean(nearest_neighbor_distances)
end

function analysis_times(tspan, interval)
    interval > 0 || throw(ArgumentError("`analysis_interval` must be positive"))
    duration = tspan[2] - tspan[1]
    n_intervals = max(1, ceil(Int, duration / interval))
    return collect(range(tspan[1], tspan[2]; length=n_intervals + 1))
end

particle_spacing = cube_side_length / baseline_particles_per_dimension
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
smoothing_length_factor = 1.0
smoothing_length = smoothing_length_factor * particle_spacing
time_step = cfl_number * smoothing_length / sound_speed
surface_tension_coefficient = 1.0

shootout_models = (CohesionForceAkinci(; surface_tension_coefficient),
                   SurfaceTensionAkinci(; surface_tension_coefficient))

# EDAC remains unregularized here: the available PST and TVF implementations cannot yet
# disable shifting near a free surface.
shootout_cases = ((name=:wcsph_continuity, sph_method="wcsph",
                   density_calculator=ContinuityDensity(), correction=nothing,
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:wcsph_continuity_akinci_free_surface_correction,
                   sph_method="wcsph", density_calculator=ContinuityDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:wcsph_summation, sph_method="wcsph",
                   density_calculator=SummationDensity(), correction=nothing,
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:wcsph_summation_akinci_free_surface_correction,
                   sph_method="wcsph", density_calculator=SummationDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:edac_continuity_inter_particle, sph_method="edac",
                   density_calculator=ContinuityDensity(), correction=nothing,
                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                   pressure_formulation=:inter_particle_averaged),
                  (name=:edac_continuity_inter_particle_akinci_free_surface_correction,
                   sph_method="edac", density_calculator=ContinuityDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                   pressure_formulation=:inter_particle_averaged),
                  (name=:edac_continuity_density_matched, sph_method="edac",
                   density_calculator=ContinuityDensity(), correction=nothing,
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:edac_continuity_density_matched_akinci_free_surface_correction,
                   sph_method="edac", density_calculator=ContinuityDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:edac_summation_inter_particle, sph_method="edac",
                   density_calculator=SummationDensity(), correction=nothing,
                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                   pressure_formulation=:inter_particle_averaged),
                  (name=:edac_summation_inter_particle_akinci_free_surface_correction,
                   sph_method="edac", density_calculator=SummationDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=TrixiParticles.inter_particle_averaged_pressure,
                   pressure_formulation=:inter_particle_averaged),
                  (name=:edac_summation_density_matched, sph_method="edac",
                   density_calculator=SummationDensity(), correction=nothing,
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:edac_summation_density_matched_akinci_free_surface_correction,
                   sph_method="edac", density_calculator=SummationDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=nothing, pressure_formulation=:density_matched),
                  (name=:iisph_summation, sph_method="iisph",
                   density_calculator=SummationDensity(), correction=nothing,
                   pressure_acceleration=nothing, pressure_formulation=:implicit),
                  (name=:iisph_summation_akinci_free_surface_correction,
                   sph_method="iisph", density_calculator=SummationDensity(),
                   correction=AkinciFreeSurfaceCorrection(fluid_density),
                   pressure_acceleration=nothing, pressure_formulation=:implicit))

rows = NamedTuple[]
shape_rows = NamedTuple[]
time_rows = NamedTuple[]
example_path = joinpath(examples_dir(), "fluid", "akinci_cube_to_sphere_3d.jl")
analysis_saveat = analysis_times(tspan, analysis_interval)

for surface_tension in shootout_models, case in shootout_cases
    surface_tension isa SurfaceTensionAkinci && isnothing(case.correction) && continue

    local result = trixi_include(@__MODULE__, example_path;
                                 sph_method=case.sph_method,
                                 surface_tension,
                                 density_calculator=case.density_calculator,
                                 pressure_acceleration=case.pressure_acceleration,
                                 correction=case.correction,
                                 particles_per_dimension=baseline_particles_per_dimension,
                                 cube_side_length, fluid_density,
                                 smoothing_length_factor, sound_speed, time_step, tspan,
                                 iisph_min_iterations,
                                 iisph_max_iterations,
                                 analysis_saveat,
                                 callbacks=CallbackSet())
    local metrics = result.metrics
    local model_name = String(nameof(typeof(surface_tension)))
    local correction_name = isnothing(case.correction) ? "none" :
                            String(nameof(typeof(case.correction)))
    local initial_support_radius_cv = support_radius_cv(result.initial_coordinates)
    local final_support_radius_cv = support_radius_cv(result.final_coordinates)
    local initial_particle_spacing_cv = particle_spacing_cv(result.initial_coordinates)
    local final_particle_spacing_cv = particle_spacing_cv(result.final_coordinates)

    push!(rows,
          (model=model_name, case=String(case.name),
           sph_method=String(case.sph_method),
           density_calculator=String(nameof(typeof(case.density_calculator))),
           pressure_formulation=String(case.pressure_formulation),
           correction=correction_name,
           final_time=tspan[2], cube_side_length,
           particles_per_dimension=baseline_particles_per_dimension,
           particle_spacing, smoothing_length_factor,
           smoothing_length, sound_speed, cfl_number,
           time_step, acoustic_cfl=time_step * sound_speed / smoothing_length,
           particle_count=metrics.particle_count,
           initial_support_radius_cv, final_support_radius_cv,
           initial_particle_spacing_cv, final_particle_spacing_cv,
           initial_radial_cv=metrics.initial_radial_cv,
           final_radial_cv=metrics.final_radial_cv,
           mean_radius_ratio=metrics.mean_radius_ratio,
           center_of_mass_drift=metrics.center_of_mass_drift,
           momentum_norm=metrics.momentum_norm,
           kinetic_energy=metrics.kinetic_energy,
           retcode=result.retcode))

    for (time, coordinates) in zip(result.times, result.coordinate_history)
        push!(time_rows,
              (model=model_name, case=String(case.name),
               sph_method=String(case.sph_method),
               density_calculator=String(nameof(typeof(case.density_calculator))),
               pressure_formulation=String(case.pressure_formulation),
               correction=correction_name,
               time,
               particle_count=metrics.particle_count,
               asphericity=support_radius_cv(coordinates),
               particle_spacing_cv=particle_spacing_cv(coordinates)))
    end

    for particle in axes(result.final_coordinates, 2)
        push!(shape_rows,
              (model=model_name, case=String(case.name),
               sph_method=String(case.sph_method),
               density_calculator=String(nameof(typeof(case.density_calculator))),
               pressure_formulation=String(case.pressure_formulation),
               correction=correction_name,
               final_time=tspan[2], cube_side_length,
               particle,
               x=result.final_coordinates[1, particle],
               y=result.final_coordinates[2, particle],
               z=result.final_coordinates[3, particle]))
    end
end

shootout_results = DataFrame(rows)
shootout_shapes = DataFrame(shape_rows)
shootout_time_series = DataFrame(time_rows)

0 < winner_window_fraction <= 1 ||
    throw(ArgumentError("`winner_window_fraction` must be in (0, 1]"))
winner_window_start = tspan[2] - winner_window_fraction * (tspan[2] - tspan[1])
winner_time_series = shootout_time_series[shootout_time_series.time .>= winner_window_start,
                                          :]
winner_scores = combine(groupby(winner_time_series, [:model, :case]),
                        :asphericity => mean => :late_time_mean_asphericity,
                        :particle_spacing_cv => mean => :late_time_mean_particle_spacing_cv)
leftjoin!(shootout_results, winner_scores; on=[:model, :case])

best_configuration_rows = [method_results[argmin(method_results.late_time_mean_asphericity),
                                          :]
                           for method_results in groupby(shootout_results,
                                       [:sph_method, :density_calculator])]
resolution_rows = NamedTuple[]

if run_resolution_study
    for winner in best_configuration_rows
        local case = only(candidate
                          for candidate in shootout_cases
                          if String(candidate.name) == winner.case)
        local surface_tension = only(model
                                     for model in shootout_models
                                     if String(nameof(typeof(model))) == winner.model)

        for resolution in resolution_levels
            local resolution_ = resolution
            local resolution_particle_spacing = cube_side_length / resolution_
            local resolution_smoothing_length = smoothing_length_factor *
                                                resolution_particle_spacing
            local resolution_time_step = cfl_number * resolution_smoothing_length /
                                         sound_speed

            if resolution_ == baseline_particles_per_dimension
                push!(resolution_rows,
                      (model=winner.model, case=winner.case,
                       sph_method=winner.sph_method,
                       density_calculator=winner.density_calculator,
                       pressure_formulation=winner.pressure_formulation,
                       correction=winner.correction,
                       final_time=winner.final_time, cube_side_length,
                       particles_per_dimension=resolution_,
                       particle_spacing=resolution_particle_spacing,
                       smoothing_length_factor,
                       smoothing_length=winner.smoothing_length,
                       sound_speed, cfl_number,
                       time_step=winner.time_step,
                       acoustic_cfl=winner.acoustic_cfl,
                       particle_count=winner.particle_count,
                       initial_support_radius_cv=winner.initial_support_radius_cv,
                       final_support_radius_cv=winner.final_support_radius_cv,
                       initial_particle_spacing_cv=winner.initial_particle_spacing_cv,
                       final_particle_spacing_cv=winner.final_particle_spacing_cv,
                       late_time_mean_asphericity=winner.late_time_mean_asphericity,
                       late_time_mean_particle_spacing_cv=winner.late_time_mean_particle_spacing_cv,
                       final_radial_cv=winner.final_radial_cv,
                       mean_radius_ratio=winner.mean_radius_ratio,
                       kinetic_energy=winner.kinetic_energy,
                       retcode=winner.retcode))
                continue
            end

            local run_module = Module(gensym(:AkinciResolutionRun))
            local result = trixi_include(run_module, example_path;
                                         sph_method=case.sph_method,
                                         surface_tension,
                                         density_calculator=case.density_calculator,
                                         pressure_acceleration=case.pressure_acceleration,
                                         correction=case.correction,
                                         particles_per_dimension=resolution_,
                                         cube_side_length, fluid_density,
                                         smoothing_length_factor, sound_speed,
                                         time_step=resolution_time_step,
                                         tspan, iisph_min_iterations,
                                         iisph_max_iterations,
                                         analysis_saveat,
                                         callbacks=CallbackSet())
            local metrics = result.metrics
            local resolution_asphericity = support_radius_cv.(result.coordinate_history)
            local resolution_particle_spacing_cv = particle_spacing_cv.(result.coordinate_history)
            local winner_indices = findall(>=(winner_window_start), result.times)

            push!(resolution_rows,
                  (model=winner.model, case=winner.case,
                   sph_method=winner.sph_method,
                   density_calculator=winner.density_calculator,
                   pressure_formulation=winner.pressure_formulation,
                   correction=winner.correction,
                   final_time=tspan[2], cube_side_length,
                   particles_per_dimension=resolution_,
                   particle_spacing=resolution_particle_spacing,
                   smoothing_length_factor,
                   smoothing_length=resolution_smoothing_length,
                   sound_speed, cfl_number,
                   time_step=resolution_time_step,
                   acoustic_cfl=resolution_time_step * sound_speed /
                                resolution_smoothing_length,
                   particle_count=metrics.particle_count,
                   initial_support_radius_cv=support_radius_cv(result.initial_coordinates),
                   final_support_radius_cv=support_radius_cv(result.final_coordinates),
                   initial_particle_spacing_cv=particle_spacing_cv(result.initial_coordinates),
                   final_particle_spacing_cv=particle_spacing_cv(result.final_coordinates),
                   late_time_mean_asphericity=mean(resolution_asphericity[winner_indices]),
                   late_time_mean_particle_spacing_cv=mean(resolution_particle_spacing_cv[winner_indices]),
                   final_radial_cv=metrics.final_radial_cv,
                   mean_radius_ratio=metrics.mean_radius_ratio,
                   kinetic_energy=metrics.kinetic_energy,
                   retcode=result.retcode))
        end
    end
end

resolution_results = DataFrame(resolution_rows)

if write_results
    output_directory = joinpath(@__DIR__, "out")
    mkpath(output_directory)
    CSV.write(joinpath(output_directory, "akinci_cube_to_sphere_3d.csv"),
              shootout_results)
    CSV.write(joinpath(output_directory, "akinci_cube_to_sphere_3d_shapes.csv"),
              shootout_shapes)
    CSV.write(joinpath(output_directory, "akinci_cube_to_sphere_3d_timeseries.csv"),
              shootout_time_series)
    if run_resolution_study
        CSV.write(joinpath(output_directory, "akinci_cube_to_sphere_3d_resolution.csv"),
                  resolution_results)
    end
end

if print_results
    show(shootout_results; allrows=true, allcols=true)
    println()
    if run_resolution_study
        show(resolution_results; allrows=true, allcols=true)
        println()
    end
end
