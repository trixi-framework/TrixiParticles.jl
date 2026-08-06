using CSV
using DataFrames
using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Statistics
using TrixiParticles

include(joinpath(@__DIR__, "..", "surface_tension_common.jl"))
using .SurfaceTensionValidation

const OUTPUT_PATH = joinpath(@__DIR__, "rayleigh_tensile_stability.csv")

function minimum_pair_ratio(solution, system, semi, particle_spacing)
    minimum_distance = Inf
    for state in solution.u
        _, u_ode = state.x
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        coordinates = TrixiParticles.current_coordinates(u, system)
        for particle in 1:(TrixiParticles.nparticles(system) - 1)
            for neighbor in (particle + 1):TrixiParticles.nparticles(system)
                minimum_distance = min(minimum_distance,
                                       norm(coordinates[:, particle] -
                                            coordinates[:, neighbor]))
            end
        end
    end
    return minimum_distance / particle_spacing
end

function density_extrema(solution, system, semi)
    minimum_density = Inf
    maximum_density = -Inf
    for state in solution.u
        v_ode, _ = state.x
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        density = collect(TrixiParticles.current_density(v, system))
        minimum_density = min(minimum_density, minimum(density))
        maximum_density = max(maximum_density, maximum(density))
    end
    return minimum_density, maximum_density
end

function run_stability_case(variant; target_particle_count=400, radius=0.01,
                            reference_density=1000.0,
                            surface_tension_coefficient=1.0,
                            background_pressure=0.0, periods=5.0, stretch=1.04,
                            shifting_technique=nothing,
                            pressure_acceleration=nothing,
                            tic_strength=0.0,
                            clip_negative_pressure=true,
                            reason="applicable shipped EOS option")
    setup = SurfaceTensionValidation.spherical_drop_initial_condition(2,
                                                                      target_particle_count;
                                                                      radius,
                                                                      reference_density,
                                                                      surface_tension_coefficient,
                                                                      stretch,
                                                                      initialize_laplace_pressure=true)
    (; initial_condition, particle_spacing) = setup
    state_equation = StateEquationCole(; sound_speed=100.0, reference_density,
                                       exponent=7, background_pressure,
                                       clip_negative_pressure)
    viscosity = ArtificialViscosityMonaghan(; alpha=0.05, beta=0.0)
    density_diffusion = DensityDiffusionAntuono(; delta=0.05)
    system = SurfaceTensionValidation.css_system(initial_condition, state_equation;
                                                 surface_tension_coefficient,
                                                 viscosity, density_diffusion,
                                                 shifting_technique,
                                                 pressure_acceleration)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    area = sum(initial_condition.mass) / reference_density
    radius_discrete = sqrt(area / pi)
    omega_exact = sqrt(6surface_tension_coefficient /
                       (reference_density * radius_discrete^3))
    period_exact = 2pi / omega_exact
    final_time = periods * period_exact
    ode = semidiscretize(semi, (0.0, final_time))
    capillary_dt = sqrt(reference_density * (1.4particle_spacing)^3 /
                        (2pi * surface_tension_coefficient))
    dt_reference = min(period_exact / 120, capillary_dt)
    accepted_dt = Float64[]
    termination_reason = Ref("final_time")
    termination_callback = DiscreteCallback((_, time, _) -> time > 0,
                                            integrator -> begin
                                                dt = abs(integrator.t - integrator.tprev)
                                                push!(accepted_dt, dt)
                                                if integrator.t > period_exact / 4 &&
                                                   dt / dt_reference < 1.0e-3
                                                    termination_reason[] = "timestep_collapse"
                                                    terminate!(integrator)
                                                elseif length(accepted_dt) >= 50_000
                                                    termination_reason[] = "step_limit"
                                                    terminate!(integrator)
                                                end
                                                u_modified!(integrator, false)
                                            end;
                                            save_positions=(false, false))
    callback = if TrixiParticles.requires_update_callback(shifting_technique)
        CallbackSet(UpdateCallback(), termination_callback)
    else
        termination_callback
    end
    saveat = range(0.0, final_time; step=period_exact / 50)
    solution = nothing
    runtime = @elapsed solution = solve(ode, RDPK3SpFSAL35(); abstol=1.0e-8,
                                        reltol=2.0e-5, dtmax=dt_reference,
                                        maxiters=50_001, save_everystep=false,
                                        saveat, callback)
    axes = [SurfaceTensionValidation.signed_axes(state, system, semi)
            for state in solution.u]
    deformation = first.(axes) .- last.(axes)
    fit = SurfaceTensionValidation.fit_angular_frequency(solution.t, deformation,
                                                         omega_exact)
    frequency_error = abs(fit.omega / omega_exact - 1)
    pair_ratio = minimum_pair_ratio(solution, system, semi, particle_spacing)
    density_min, density_max = density_extrema(solution, system, semi)
    periods_completed = last(solution.t) / period_exact
    minimum_dt_ratio = isempty(accepted_dt) ? NaN : minimum(accepted_dt) / dt_reference
    accepted = periods_completed >= periods && frequency_error <= 0.05 &&
               pair_ratio >= 0.5 && density_min >= 980 && density_max <= 1020 &&
               termination_reason[] == "final_time"
    return (; variant=String(variant), admissible=true, status=termination_reason[],
            reason, target_particle_count,
            particle_count=TrixiParticles.nparticles(system), particle_spacing,
            background_pressure, tic_strength, clip_negative_pressure,
            requested_periods=periods, periods_completed,
            omega_exact, omega_measured=fit.omega, frequency_error,
            fit_residual=fit.residual, minimum_pair_ratio=pair_ratio,
            density_min, density_max, dt_reference, minimum_dt_ratio,
            accepted_steps=solution.stats.naccept,
            rejected_steps=solution.stats.nreject, runtime, accepted)
end

function ineligible_row(variant, reason; target_particle_count=400, periods=5.0)
    return (; variant=String(variant), admissible=false, status="not_run",
            reason, target_particle_count, particle_count=0,
            particle_spacing=NaN, background_pressure=NaN, tic_strength=NaN,
            clip_negative_pressure=false,
            requested_periods=periods, periods_completed=0.0,
            omega_exact=NaN, omega_measured=NaN, frequency_error=NaN,
            fit_residual=NaN, minimum_pair_ratio=NaN,
            density_min=NaN, density_max=NaN, dt_reference=NaN,
            minimum_dt_ratio=NaN, accepted_steps=0, rejected_steps=0,
            runtime=0.0, accepted=false)
end

function run_tensile_stability_study(; output_path=OUTPUT_PATH)
    radius = 0.01
    surface_tension_coefficient = 1.0
    laplace_pressure = surface_tension_coefficient / radius
    rows = [run_stability_case(:baseline),
        run_stability_case(:eos_background_laplace;
                           background_pressure=laplace_pressure),
        ineligible_row(:transport_velocity,
                       "TVF requires an unavailable free-surface mask"),
        run_stability_case(:particle_shifting_tangential;
                           shifting_technique=ConsistentShiftingSun2019(;
                                                                        free_surface_treatment=FreeSurfaceTangentialShifting()),
                           reason="colorfield tangential free-surface shifting"),
        run_stability_case(:particle_shifting_sun2017_tangential;
                           shifting_technique=ParticleShiftingTechniqueSun2017(;
                                                                               free_surface_treatment=FreeSurfaceTangentialShifting()),
                           reason="callback shifting without Sun-2019 transport terms")]
    controls = ((:interface_tic_010_sun2017_tangential, 0.1),
                (:interface_tic_025_sun2017_tangential, 0.25),
                (:interface_tic_050_sun2017_tangential, 0.5),
                (:interface_tic_100_sun2017_tangential, 1.0))
    for (variant, strength) in controls
        reason = strength == 0.25 ?
                 "selected interface-aware TIC and callback-shifting combination" :
                 "bounded interface-aware TIC strength control"
        push!(rows,
              run_stability_case(variant;
                                 shifting_technique=ParticleShiftingTechniqueSun2017(;
                                                                                     free_surface_treatment=FreeSurfaceTangentialShifting()),
                                 pressure_acceleration=InterfaceAwareTensileInstabilityControl(;
                                                                                               strength),
                                 tic_strength=strength,
                                 clip_negative_pressure=false, reason))
    end
    data = DataFrame(rows)
    CSV.write(output_path, data)
    println(data)
    println("Wrote Rayleigh tensile-stability study to ", output_path)
    return data
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tensile_stability_study()
end
