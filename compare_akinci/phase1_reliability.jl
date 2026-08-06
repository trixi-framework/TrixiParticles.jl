using Printf

include(joinpath(@__DIR__, "surface_tension_calibration.jl"))

function run_phase1_case(model_name, alpha, support_width, final_time,
                         target_particle_count)
    model = calibration_model(model_name, 1.0)
    result = laplace_pressure_calibration(model; final_time, target_particle_count,
                                          interface_taper_start=alpha,
                                          support_taper_width=support_width,
                                          record_steps=true)
    print_laplace_calibration(model_name, 1.0, result)
    return result
end

function run_phase1_sensitivity(model_name, final_time, target_particle_count)
    println("model,alpha,support_width,sigma_fit,runtime,accepted,rejected,eta_p01," *
            "eta_median,eta_tail_head,speed_rms,active,transition")
    for alpha in (0.5, 0.8, 0.9), support_width in (0.025, 0.05, 0.10)
        result = run_phase1_case(model_name, alpha, support_width, final_time,
                                 target_particle_count)
        @printf("%s,%.3f,%.3f,%.8f,%.3f,%d,%d,%.6f,%.6f,%.6f,%.6e,%d,%d\n",
                model_name, alpha, support_width, result.inferred_surface_tension,
                result.runtime, result.accepted_steps, result.rejected_steps,
                result.eta_p01, result.eta_median, result.eta_tail_head,
                result.speed_rms, result.active_particles, result.transition_particles)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (6, 4) ||
        error("usage: phase1_reliability.jl single MODEL ALPHA SUPPORT_WIDTH FINAL_TIME TARGET_PARTICLES\n" *
              "   or: phase1_reliability.jl sensitivity MODEL FINAL_TIME TARGET_PARTICLES")

    mode = ARGS[1]
    model_name = ARGS[2]
    if mode == "single"
        alpha = parse(Float64, ARGS[3])
        support_width = parse(Float64, ARGS[4])
        final_time = parse(Float64, ARGS[5])
        target_particle_count = parse(Int, ARGS[6])
        run_phase1_case(model_name, alpha, support_width, final_time,
                        target_particle_count)
    elseif mode == "sensitivity"
        final_time = parse(Float64, ARGS[3])
        target_particle_count = parse(Int, ARGS[4])
        run_phase1_sensitivity(model_name, final_time, target_particle_count)
    else
        error("unknown mode `$mode`")
    end
end
