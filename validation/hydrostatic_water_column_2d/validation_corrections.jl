using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm
using CSV
using DataFrames
using JSON

include("correction_util.jl")

# Correction-enabled variants of the O'Connor and Rogers hydrostatic plate benchmark.
# All variants share the geometry, kernel, physical parameters, and integration settings.
resolutions = (3, 5, 9, 17)
methods = (:wcsph, :edac)
wcsph_density_mode = :continuity
edac_density_mode = :summation
wcsph_pressure_clipping = false
wcsph_artificial_viscosity = nothing
initialize_summation_density = true
correction_cases = HydrostaticCorrectionMethods.DEFAULT_CASES
check_reverse_order = true
tspan = (0.0, 1.0)
sample_dt = 0.0025
reinitialization_dt = 0.01
averaging_interval = (0.5, 1.0)
output_root = "out_hydrostatic_corrections"
save_solution = true

wcsph_density_mode in (:continuity, :summation) ||
    throw(ArgumentError("wcsph_density_mode must be :continuity or :summation"))
edac_density_mode in (:continuity, :summation) ||
    throw(ArgumentError("edac_density_mode must be :continuity or :summation"))

summaries = NamedTuple[]
order_comparisons = NamedTuple[]
histories = Dict{Tuple{Symbol, Symbol, Int, Bool}, Any}()
case_settings = Dict{String, Any}()
order_time_grids = Dict{Tuple{Symbol, Int}, Vector{Float64}}()

for resolution in resolutions, method in methods, correction in correction_cases
    local definition = HydrostaticCorrectionMethods.definition(correction)
    local base_density_mode = method == :wcsph ? wcsph_density_mode : edac_density_mode
    # A continuity-density combined case already covers this legacy control.
    base_density_mode == :continuity &&
        correction == :continuity_reinit && continue
    for reverse_order in (check_reverse_order && correction == :combined ?
         (false, true) : (false,))
        local continuity = base_density_mode == :continuity ||
                           correction == :continuity_reinit
        local reinitialize_density = continuity && definition.shepard
        local density_filter = !continuity && definition.shepard ?
                               ShepardKernelCorrection() : nothing
        local gradient_filter = hydrostatic_gradient_correction(correction)
        local label = "$(method)_$(correction)_$(resolution)" *
                      (reverse_order ? "_reversed" : "")
        local case_directory = joinpath(output_root, label)
        println("Running hydrostatic correction case $label")
        flush(stdout)

        # Assemble the example first, so callbacks can reference the copied runtime systems.
        trixi_include(@__MODULE__,
                      joinpath(examples_dir(), "fsi", "hydrostatic_water_column_2d.jl");
                      use_edac=method == :edac, n_particles_plate_y=resolution, tspan,
                      fluid_density_calculator=continuity ? ContinuityDensity() :
                                               SummationDensity(),
                      fluid_viscosity=method == :wcsph ? wcsph_artificial_viscosity :
                                      nothing,
                      density_correction=density_filter,
                      fluid_gradient_correction=gradient_filter,
                      boundary_gradient_correction=gradient_filter,
                      density_diffusion=method == :wcsph && continuity ?
                                        DensityDiffusionMolteniColagrossi(delta=0.1) :
                                        nothing,
                      reverse_system_order=reverse_order,
                      clip_negative_pressure=method == :wcsph && wcsph_pressure_clipping,
                      initialize_summation_density,
                      update_strategy=SerialUpdate(),
                      parallelization_backend=SerialBackend(),
                      sol=nothing)

        local runtime_semi = ode.p.semi
        local fluid = only(filter(system -> system isa TrixiParticles.AbstractFluidSystem,
                                  runtime_semi.systems))
        # RectangularTank translates the fluid by half a spacing. Its free surface is
        # half a spacing above the top particle, not at initial_fluid_size[2].
        global hydrostatic_surface_height = maximum(view(tank.fluid.coordinates, 2, :)) +
                                            fluid_particle_spacing / 2
        local balance = initial_hydrostatic_balance(ode, fluid)
        mkpath(case_directory)
        open(joinpath(case_directory, "initial_state.json"), "w") do io
            JSON.print(io,
                       (; method, correction, resolution, reverse_order,
                        base_density_mode,
                        density_mode=continuity ? :continuity : :summation,
                        density_calculator=string(typeof(fluid.density_calculator)),
                        hydrostatic_surface_height, initialization=summation_initialization,
                        balance...), 2)
        end
        local density_reinit = reinitialize_density ?
                               DensityReinitializationCallback(fluid, runtime_semi;
                                                               dt=reinitialization_dt,
                                                               reinit_initial_solution=false) :
                               nothing
        local reinit_audit = density_reinit_audit()
        # Use the callback's update logic at identical physical times in matched runs.
        # Its usual condition instead fires at the first accepted step after dt, which
        # makes the filtering times depend on the adaptive solver's step sequence.
        local reinit_callback = if reinitialize_density
            local times = collect((first(tspan) + reinitialization_dt):reinitialization_dt:last(tspan))
            TrixiParticles.PresetTimeCallback(times,
                                              integrator -> scheduled_density_reinit!(density_reinit,
                                                                                      integrator,
                                                                                      reinit_audit);
                                              initialize=(cb, u, t,
                                                          integrator) -> TrixiParticles.initialize_reinit_cb!(density_reinit.affect!,
                                                                                                              u,
                                                                                                              t,
                                                                                                              integrator),
                                              save_positions=(false, false))
        else
            nothing
        end
        local pp = PostprocessCallback(; dt=sample_dt, output_directory=case_directory,
                                       filename="validation_run_hydrostatic_corrections",
                                       hydrostatic_errors,
                                       eos_residual=equation_of_state_residual,
                                       plate_deflection=correction_plate_deflection,
                                       interface_force_balance,
                                       write_file_interval=0)
        local saving = save_solution ?
                       SolutionSavingCallback(; dt=0.1, output_directory=case_directory) :
                       nothing
        local recorded_times = Float64[first(tspan)]
        local record_steps = check_reverse_order && correction == :combined &&
                             !reverse_order ?
                             TrixiParticles.DiscreteCallback((u, t, integrator) -> true,
                                                             integrator -> begin
                                                                 push!(recorded_times,
                                                                       integrator.t)
                                                                 TrixiParticles.derivative_discontinuity!(integrator,
                                                                                                          false)
                                                             end;
                                                             save_positions=(false, false)) :
                             nothing
        # Reversing declaration order must test the same time discretization. Replaying
        # the forward accepted grid avoids comparing different adaptive/split trajectories.
        local integration_options = if reverse_order
            local times = order_time_grids[(method, resolution)]
            (; adaptive=false, dt=maximum(diff(times)), tstops=times[2:end])
        else
            (; dt=1e-8)
        end
        local callbacks = CallbackSet(split_integration, reinit_callback, pp, saving,
                                      record_steps)
        global sol = solve(ode, RDPK3SpFSAL49(); integration_options..., reltol=1e-6,
                           abstol=1e-8,
                           save_everystep=false, callback=callbacks)
        @assert sol.retcode == ReturnCode.Success "hydrostatic correction case $label failed"
        if !isnothing(record_steps)
            order_time_grids[(method, resolution)] = recorded_times
        end

        local data = JSON.parsefile(joinpath(case_directory,
                                             "validation_run_hydrostatic_corrections.json"))
        histories[(method, correction, resolution, reverse_order)] = data
        local summary = correction_summary(data; method, correction, reverse_order,
                                           resolution,
                                           spacing=fluid_particle_spacing,
                                           analytical_deflection=analytical_value,
                                           averaging_interval)
        local last_reinitialization = reinitialize_density ? density_reinit.affect!.last_t :
                                      missing
        local audit = reinit_audit[]
        push!(summaries,
              (; summary..., balance..., base_density_mode,
               density_mode=continuity ? :continuity : :summation, last_reinitialization,
               reinitialization_count=audit.count,
               reinitialization_volume_residual=audit.count > 0 ? audit.volume_residual :
                                                missing,
               frozen_volume_difference=audit.count > 0 ? audit.frozen_volume_difference :
                                        missing,
               evolved_density_change=audit.count > 0 ? audit.evolved_density_change :
                                      missing,
               post_reinitialization_eos_residual=audit.count > 0 && method == :wcsph ?
                                                  audit.eos_residual : missing,
               edac_reinitialization_pressure_change=audit.count > 0 && method == :edac ?
                                                     audit.edac_pressure_change : missing))
        case_settings[label] = (;
                                density_calculator=string(typeof(fluid.density_calculator)),
                                definition,
                                initialization=summation_initialization,
                                reinitialize_density, hydrostatic_surface_height,
                                pressure_clipping=method == :wcsph &&
                                                  wcsph_pressure_clipping,
                                viscosity=string(fluid.viscosity),
                                density_diffusion=fluid isa WeaklyCompressibleSPHSystem ?
                                                  string(fluid.density_diffusion) :
                                                  "nothing",
                                fluid_time_grid=check_reverse_order &&
                                                correction == :combined ?
                                                order_time_grids[(method, resolution)] :
                                                nothing,
                                fluid_gradient_correction=string(gradient_filter),
                                boundary_gradient_correction=string(gradient_filter))
        if reverse_order
            push!(order_comparisons,
                  correction_order_comparison(histories[(method, correction, resolution,
                                                         false)],
                                              data; method, resolution,
                                              analytical_deflection=analytical_value))
            CSV.write(joinpath(output_root, "hydrostatic_correction_order.csv"),
                      DataFrame(order_comparisons))
        end
        # Preserve completed cases even when a long refinement study is interrupted.
        CSV.write(joinpath(output_root, "hydrostatic_corrections.csv"),
                  DataFrame(summaries))
        open(joinpath(output_root, "parameters.json"), "w") do io
            JSON.print(io,
                       (; resolutions, methods, correction_cases, tspan, sample_dt,
                        reinitialization_dt, averaging_interval, check_reverse_order,
                        density_reinitialization_schedule="prescribed physical times",
                        order_comparison="replay the forward accepted fluid time grid",
                        analytical_deflection=analytical_value, gravity, fluid_density,
                        structure_density, sound_speed, E, nu, plate_size,
                        initial_fluid_size,
                        validation_version=2,
                        hydrostatic_reference="translated free surface; exact Cole hydrostatics for WCSPH",
                        edac_adami_pressure_gradient="fluid-sided",
                        wcsph_extrapolated_pressure_gradient="fluid-sided",
                        density_boundary_policy="raw-pressure-predictor",
                        wcsph_summation_initialization=initialize_summation_density ?
                                                       "operator-offset-hydrostatic" :
                                                       "uncalibrated",
                        wcsph_density_mode, edac_density_mode,
                        wcsph_density_calculator=wcsph_density_mode == :continuity ?
                                                 "ContinuityDensity" : "SummationDensity",
                        wcsph_density_diffusion="MolteniColagrossi(delta=0.1) for continuity density only",
                        wcsph_pressure_clipping,
                        wcsph_viscosity=string(wcsph_artificial_viscosity),
                        cases=case_settings,
                        save_solution,
                        smoothing_kernel=string(smoothing_kernel),
                        method_matrix="Shepard off/on × gradient none/kernel/gradient/blend(0.5)/mixed",
                        method_definitions=Dict(string(case) => HydrostaticCorrectionMethods.definition(case)
                                                for case in correction_cases),
                        julia_version=string(VERSION),
                        trixiparticles_version=string(pkgversion(TrixiParticles)),
                        smoothing_length_factor=sqrt(2)), 2)
        end
    end
end

summary_file = joinpath(output_root, "hydrostatic_corrections.csv")
mkpath(output_root)
CSV.write(summary_file, DataFrame(summaries))
comparisons = NamedTuple[]
for resolution in resolutions, method in methods,
    (baseline, corrected) in
    ((:none, :kernel), (:none, :gradient), (:none, :blended), (:none, :mixed),
     (:shepard, :shepard_kernel), (:shepard, :shepard_gradient),
     (:shepard, :shepard_blended), (:shepard, :combined))

    raw = filter(row -> row.resolution == resolution && row.method == method &&
                        row.correction == baseline && !row.reverse_order, summaries)
    improved = filter(row -> row.resolution == resolution && row.method == method &&
                             row.correction == corrected && !row.reverse_order, summaries)
    (isempty(raw) || isempty(improved)) && continue
    a, b = only(raw), only(improved)
    push!(comparisons,
          (; resolution, method, baseline, corrected,
           density_error_ratio=(b.final_density_error / a.final_density_error),
           pressure_error_ratio=(b.final_pressure_error / a.final_pressure_error),
           initial_momentum_error_ratio=(b.initial_momentum_error /
                                         a.initial_momentum_error),
           interior_momentum_error_ratio=(b.initial_momentum_error_interior /
                                          a.initial_momentum_error_interior)))
end
if !isempty(comparisons)
    CSV.write(joinpath(output_root, "hydrostatic_correction_comparisons.csv"),
              DataFrame(comparisons))
end
println("Hydrostatic correction validation results:\n", DataFrame(summaries))
println("Wrote $summary_file")
