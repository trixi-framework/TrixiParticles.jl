using LinearAlgebra
using Printf
using Serialization
using Statistics

function radial_asphericity(coordinates, particle_spacing)
    center = vec(mean(coordinates; dims=2))
    centered = coordinates .- center
    radius_squared = vec(sum(abs2, centered; dims=1))

    volume = size(coordinates, 2) * particle_spacing^3
    equivalent_radius = cbrt(3volume / (4pi))
    second_moment_error = mean(radius_squared) / (3equivalent_radius^2 / 5) - 1
    fourth_moment_error = mean(abs2, radius_squared) /
                          (3equivalent_radius^4 / 7) - 1
    asphericity = sqrt((second_moment_error^2 + fourth_moment_error^2) / 2)
    return (; asphericity, second_moment_error, fourth_moment_error,
            equivalent_radius)
end

function angular_alignment(coordinates, particle_spacing)
    center = vec(mean(coordinates; dims=2))
    center_z = (minimum(coordinates[3, :]) + maximum(coordinates[3, :])) / 2
    slab = abs.(coordinates[3, :] .- center_z) .<= 0.75 * particle_spacing
    x = coordinates[1, slab] .- center[1]
    y = coordinates[2, slab] .- center[2]
    radius = hypot.(x, y)
    outer_radius = maximum(radius)
    annulus = (radius .>= 0.2outer_radius) .& (radius .<= 0.85outer_radius)
    annulus_x = x[annulus]
    annulus_y = y[annulus]
    angles = mod.(atan.(annulus_y, annulus_x), 2pi)

    bins = 72
    counts = zeros(Int, bins)
    for angle in angles
        index = mod(floor(Int, angle / (2pi) * bins), bins) + 1
        counts[index] += 1
    end
    angular_bin_cv = std(counts) / mean(counts)
    eightfold_alignment = abs(mean(cis.(8angles)))
    sixteenfold_alignment = abs(mean(cis.(16angles)))
    radial_neighbor_alignment = if length(angles) < 2
        NaN
    else
        nearest_neighbor_angles = similar(angles)
        for particle in eachindex(angles)
            nearest_distance_squared = Inf
            nearest_neighbor = firstindex(angles)
            for neighbor in eachindex(angles)
                particle == neighbor && continue
                distance_squared = abs2(annulus_x[particle] - annulus_x[neighbor]) +
                                   abs2(annulus_y[particle] - annulus_y[neighbor])
                if distance_squared < nearest_distance_squared
                    nearest_distance_squared = distance_squared
                    nearest_neighbor = neighbor
                end
            end
            nearest_neighbor_angles[particle] = atan(annulus_y[nearest_neighbor] -
                                                     annulus_y[particle],
                                                     annulus_x[nearest_neighbor] -
                                                     annulus_x[particle])
        end
        abs(mean(cis.(2 .* (nearest_neighbor_angles .- angles))))
    end
    return (; angular_bin_cv, eightfold_alignment, sixteenfold_alignment,
            radial_neighbor_alignment)
end

function figure_02_metrics(snapshot; system_index=1)
    return map(snapshot.times, snapshot.frames) do time, frame
        system = frame.systems[system_index]
        coordinates = system.coordinates
        particle_spacing = system.particle_spacing
        radial = radial_asphericity(coordinates, particle_spacing)
        alignment = angular_alignment(coordinates, particle_spacing)
        extents = vec(maximum(coordinates; dims=2) - minimum(coordinates; dims=2)) .+
                  particle_spacing
        width_x, width_y = extents[1], extents[2]
        width = sqrt(width_x * width_y)
        height = extents[3]
        density_min, density_max = extrema(system.density)
        rms_speed = sqrt(mean(vec(sum(abs2, system.velocity; dims=1))))
        return (; time, radial..., alignment..., width_x, width_y, width, height,
                height_to_width=height / width,
                planar_asymmetry=abs(width_x - width_y) / width,
                minimum_z=minimum(coordinates[3, :]),
                density_min, density_max, rms_speed)
    end
end

function write_figure_02_metrics(snapshot_path, output_path=nothing)
    snapshot = open(deserialize, snapshot_path)
    metrics = figure_02_metrics(snapshot)
    io = isnothing(output_path) ? stdout : open(output_path, "w")
    try
        println(io,
                "time,radial_asphericity,second_moment_error,fourth_moment_error,equivalent_radius,angular_bin_cv,eightfold_alignment,sixteenfold_alignment,radial_neighbor_alignment,width_x,width_y,width,height,height_to_width,planar_asymmetry,minimum_z,density_min,density_max,rms_speed")
        for row in metrics
            @printf(io,
                    "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                    row.time, row.asphericity, row.second_moment_error,
                    row.fourth_moment_error, row.equivalent_radius,
                    row.angular_bin_cv, row.eightfold_alignment,
                    row.sixteenfold_alignment, row.radial_neighbor_alignment,
                    row.width_x, row.width_y, row.width, row.height,
                    row.height_to_width, row.planar_asymmetry, row.minimum_z,
                    row.density_min, row.density_max, row.rms_speed)
        end
    finally
        isnothing(output_path) || close(io)
    end

    stats = snapshot.solver_stats
    @printf(stderr, "%s model=%s sigma=%s accepted=%s rejected=%s runtime=%.3f s\n",
            snapshot.case_name, snapshot.model,
            string(snapshot.surface_tension_coefficient),
            string(stats.accepted_steps), string(stats.rejected_steps), snapshot.runtime)
    return metrics
end

function metric_at_time(metrics, target_time)
    _, index = findmin([abs(row.time - target_time) for row in metrics])
    return metrics[index]
end

function figure_02_comparison(baseline, candidate)
    baseline_metrics = figure_02_metrics(baseline)
    candidate_metrics = figure_02_metrics(candidate)
    common_times = intersect(getproperty.(baseline_metrics, :time),
                             getproperty.(candidate_metrics, :time))
    pre_release_times = filter(time -> 0 < time <= 0.05, common_times)
    isempty(pre_release_times) &&
        error("Figure 2 comparison needs saved states before release")

    pre_release_error = maximum(pre_release_times) do time
        baseline_row = metric_at_time(baseline_metrics, time)
        candidate_row = metric_at_time(candidate_metrics, time)
        abs(candidate_row.asphericity - baseline_row.asphericity)
    end
    baseline_release = metric_at_time(baseline_metrics, 0.05)
    candidate_release = metric_at_time(candidate_metrics, 0.05)
    post_release_times = filter(time -> time >= 0.05, common_times)
    maximum_post_release_width_error = maximum(post_release_times) do time
        baseline_row = metric_at_time(baseline_metrics, time)
        candidate_row = metric_at_time(candidate_metrics, time)
        abs(candidate_row.width / baseline_row.width - 1)
    end
    final_time = min(last(baseline.times), last(candidate.times))
    baseline_final = metric_at_time(baseline_metrics, final_time)
    candidate_final = metric_at_time(candidate_metrics, final_time)
    final_height_error_in_spacing = abs(candidate_final.height - baseline_final.height) /
                                    candidate.frames[1].systems[1].particle_spacing
    final_aspect_error = abs(candidate_final.height_to_width -
                             baseline_final.height_to_width)
    maximum_angular_bin_cv = 1.25 * baseline_final.angular_bin_cv
    maximum_eightfold_alignment = max(0.02,
                                      2 * baseline_final.eightfold_alignment)
    maximum_radial_neighbor_alignment = max(0.05,
                                            2 * baseline_final.radial_neighbor_alignment)
    qualitative_maximum_angular_bin_cv = 2 * baseline_final.angular_bin_cv
    qualitative_maximum_eightfold_alignment = max(0.03,
                                                  4 * baseline_final.eightfold_alignment)
    qualitative_maximum_radial_neighbor_alignment = max(0.1,
                                                        4 *
                                                        baseline_final.radial_neighbor_alignment)

    density_min = minimum(row.density_min for row in candidate_metrics)
    density_max = maximum(row.density_max for row in candidate_metrics)
    minimum_z = minimum(row.minimum_z for row in candidate_metrics)
    accepted_steps = candidate.solver_stats.accepted_steps
    rejected_steps = candidate.solver_stats.rejected_steps
    baseline_accepted_steps = baseline.solver_stats.accepted_steps
    baseline_rejected_steps = baseline.solver_stats.rejected_steps
    rejected_fraction = rejected_steps / (accepted_steps + rejected_steps)
    timestep_p01 = candidate.timestep_stats.p01
    timestep_median = candidate.timestep_stats.median
    timestep_tail_to_head = candidate.timestep_stats.tail_to_head
    reliability_pass = rejected_fraction <= 0.25 &&
                       density_min >= 900 && density_max <= 1_020 && minimum_z >= 0 &&
                       !ismissing(timestep_tail_to_head) && timestep_tail_to_head >= 0.5
    shape_pass = pre_release_error <= 0.025 &&
                 candidate_release.asphericity <= baseline_release.asphericity + 0.01 &&
                 maximum_post_release_width_error <= 0.1 &&
                 final_height_error_in_spacing <= 2 && final_aspect_error <= 0.05 &&
                 candidate_final.height_to_width <= 0.2 &&
                 candidate_final.planar_asymmetry <= 0.05 &&
                 candidate_final.angular_bin_cv <= maximum_angular_bin_cv &&
                 candidate_final.eightfold_alignment <= maximum_eightfold_alignment &&
                 candidate_final.radial_neighbor_alignment <=
                 maximum_radial_neighbor_alignment
    qualitative_pass = candidate_release.asphericity <= 0.05 &&
                       candidate_final.height_to_width <= 0.25 &&
                       candidate_final.planar_asymmetry <= 0.1 &&
                       abs(candidate_final.width / baseline_final.width - 1) <= 0.2 &&
                       candidate_final.angular_bin_cv <=
                       qualitative_maximum_angular_bin_cv &&
                       candidate_final.eightfold_alignment <=
                       qualitative_maximum_eightfold_alignment &&
                       candidate_final.radial_neighbor_alignment <=
                       qualitative_maximum_radial_neighbor_alignment &&
                       reliability_pass
    nominal_acceptance_pass = shape_pass && reliability_pass
    spacing = candidate.frames[1].systems[1].particle_spacing
    return (; case_name=candidate.case_name,
            sigma=candidate.surface_tension_coefficient,
            artificial_viscosity_alpha=hasproperty(candidate,
                                                   :artificial_viscosity_alpha) ?
                                       candidate.artificial_viscosity_alpha : missing,
            surface_tension_mode=hasproperty(candidate, :surface_tension_mode) ?
                                 candidate.surface_tension_mode : missing,
            smoothing_kernel_mode=hasproperty(candidate, :smoothing_kernel_mode) ?
                                  candidate.smoothing_kernel_mode : missing,
            smoothing_length_ratio=hasproperty(candidate, :smoothing_length_ratio) ?
                                   candidate.smoothing_length_ratio : missing,
            normal_smoothing=hasproperty(candidate, :normal_smoothing) ?
                             candidate.normal_smoothing : missing,
            contact_angle=hasproperty(candidate, :contact_angle) ?
                          candidate.contact_angle : missing,
            ccsf_contact_angle=hasproperty(candidate, :ccsf_contact_angle) ?
                               candidate.ccsf_contact_angle : missing,
            viscosity_mode=hasproperty(candidate, :viscosity_mode) ?
                           candidate.viscosity_mode : missing,
            density_diffusion_mode=hasproperty(candidate, :density_diffusion_mode) ?
                                   candidate.density_diffusion_mode : missing,
            density_diffusion_delta=hasproperty(candidate, :density_diffusion_delta) ?
                                    candidate.density_diffusion_delta : missing,
            initial_particle_distribution=hasproperty(candidate,
                                                      :initial_particle_distribution) ?
                                          candidate.initial_particle_distribution : missing,
            pressure_stabilization=hasproperty(candidate, :pressure_stabilization) ?
                                   candidate.pressure_stabilization : missing,
            tic_strength=hasproperty(candidate, :tic_strength) ?
                         candidate.tic_strength : missing,
            shifting_mode=hasproperty(candidate, :shifting_mode) ?
                          candidate.shifting_mode : missing,
            shifting_v_max_factor=hasproperty(candidate,
                                              :shifting_v_max_factor) ?
                                  candidate.shifting_v_max_factor : missing,
            shifting_sound_speed_factor=hasproperty(candidate,
                                                    :shifting_sound_speed_factor) ?
                                        candidate.shifting_sound_speed_factor : missing,
            particle_spacing=spacing,
            pre_release_asphericity_error=pre_release_error,
            baseline_release_asphericity=baseline_release.asphericity,
            css_release_asphericity=candidate_release.asphericity,
            maximum_post_release_width_error,
            baseline_final_aspect=baseline_final.height_to_width,
            css_final_aspect=candidate_final.height_to_width,
            final_aspect_error, final_height_error_in_spacing,
            css_final_planar_asymmetry=candidate_final.planar_asymmetry,
            baseline_angular_bin_cv=baseline_final.angular_bin_cv,
            css_angular_bin_cv=candidate_final.angular_bin_cv,
            baseline_eightfold_alignment=baseline_final.eightfold_alignment,
            css_eightfold_alignment=candidate_final.eightfold_alignment,
            baseline_radial_neighbor_alignment=baseline_final.radial_neighbor_alignment,
            css_radial_neighbor_alignment=candidate_final.radial_neighbor_alignment,
            density_min, density_max, minimum_z,
            baseline_accepted_steps, baseline_rejected_steps,
            baseline_runtime=baseline.runtime,
            accepted_steps, rejected_steps, rejected_fraction,
            timestep_p01, timestep_median, timestep_tail_to_head,
            runtime=candidate.runtime, shape_pass, reliability_pass,
            qualitative_pass, nominal_acceptance_pass)
end

function write_figure_02_comparisons(baseline_path, candidate_paths, output_path)
    baseline = open(deserialize, baseline_path)
    rows = map(enumerate(candidate_paths)) do (index, path)
        candidate = open(deserialize, path)
        row = figure_02_comparison(baseline, candidate)
        gate_role = index == 1 ? :nominal_acceptance : :robustness
        required_pass = index == 1 ? row.nominal_acceptance_pass :
                        row.qualitative_pass
        merge(row, (; gate_role, required_pass))
    end
    names = propertynames(first(rows))
    open(output_path, "w") do io
        println(io, join(names, ','))
        for row in rows
            println(io, join((getproperty(row, name) for name in names), ','))
        end
    end
    for row in rows
        @printf("sigma=%.6g dx=%.6g release=%.5f final-h/w=%.4f symmetry=%.2f%% angular-cv=%.3f m8=%.3f radial=%.3f width-error=%.2f%% height-error=%.2f dx density=[%.2f, %.2f] rejected=%.1f%% qualitative=%s required=%s\n",
                row.sigma, row.particle_spacing,
                row.css_release_asphericity, row.css_final_aspect,
                100 * row.css_final_planar_asymmetry,
                row.css_angular_bin_cv, row.css_eightfold_alignment,
                row.css_radial_neighbor_alignment,
                100 * row.maximum_post_release_width_error,
                row.final_height_error_in_spacing, row.density_min, row.density_max,
                100 * row.rejected_fraction,
                row.qualitative_pass ? "pass" : "fail",
                row.required_pass ? "pass" : "fail")
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "compare"
        length(ARGS) >= 4 ||
            error("pass 'compare', an Akinci snapshot, output CSV, and CSS snapshots")
        write_figure_02_comparisons(ARGS[2], ARGS[4:end], ARGS[3])
    else
        length(ARGS) in (1, 2) ||
            error("pass a Figure 2 snapshot and optional CSV output path")
        output = length(ARGS) == 2 ? ARGS[2] : nothing
        write_figure_02_metrics(ARGS[1], output)
    end
end
