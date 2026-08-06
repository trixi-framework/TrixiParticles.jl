using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Printf
using Statistics
using TrixiParticles

include(joinpath(@__DIR__, "wcsph_variants.jl"))
include(joinpath(@__DIR__, "surface_model_variants.jl"))

function deformed_drop(; stretch=1.1, drop_volume=1.0e-6, target_particle_count=750,
                       reference_density=1000.0)
    particle_spacing = cbrt(drop_volume / target_particle_count)
    drop_radius = cbrt(3 * drop_volume / (4pi))
    sampling_radius = drop_radius + particle_spacing / 2
    sphere = SphereShape(particle_spacing, sampling_radius, (0.0, 0.0, 0.0),
                         reference_density; sphere_type=VoxelSphere())

    coordinates = copy(sphere.coordinates)
    coordinates[1, :] .*= stretch
    coordinates[2, :] ./= sqrt(stretch)
    coordinates[3, :] ./= sqrt(stretch)
    initial_condition = InitialCondition(; particle_spacing, coordinates,
                                         velocity=sphere.velocity, mass=sphere.mass,
                                         density=sphere.density,
                                         pressure=sphere.pressure)
    return initial_condition
end

function rayleigh_stiffness(surface_tension; stretch=1.1, reference_density=1000.0,
                            drop_volume=1.0e-6, target_particle_count=750)
    initial_condition = deformed_drop(; stretch, reference_density, drop_volume,
                                      target_particle_count)
    particle_spacing = initial_condition.particle_spacing
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    state_equation = StateEquationCole(; sound_speed=100.0,
                                       reference_density, exponent=7,
                                       clip_negative_pressure=true)
    surface_normal_method = if TrixiParticles.requires_surface_normal(surface_tension)
        ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf,
                                interface_threshold=0.01,
                                ideal_density_threshold=0.95)
    else
        nothing
    end
    fluid_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=ContinuityDensity(),
                                               density_diffusion=DensityDiffusionAntuono(;
                                                                                         delta=0.1),
                                               state_equation,
                                               surface_tension, surface_normal_method,
                                               correction=AkinciFreeSurfaceCorrection(reference_density),
                                               reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(fluid_system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0e-3))
    v_ode, u_ode = ode.u0.x
    dv_ode = zero(v_ode)
    # The first pass initializes all surface interaction caches.
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)
    TrixiParticles.kick!(dv_ode, v_ode, u_ode, ode.p, 0.0)

    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    dv = TrixiParticles.wrap_v(dv_ode, fluid_system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, fluid_system))
    velocity = Array(TrixiParticles.current_velocity(v, fluid_system))
    acceleration = Array(dv[1:3, :])
    mass = fluid_system.mass
    center = vec(sum(coordinates .* reshape(mass, 1, :); dims=2) / sum(mass))
    relative_coordinates = coordinates .- center

    quadrupole = mean(relative_coordinates[1, :] .^ 2 .-
                      (relative_coordinates[2, :] .^ 2 .+
                       relative_coordinates[3, :] .^ 2) / 2)
    quadrupole_acceleration = 2mean(relative_coordinates[1, :] .* acceleration[1, :] .-
                                    (relative_coordinates[2, :] .* acceleration[2, :] .+
                                     relative_coordinates[3, :] .* acceleration[3, :]) / 2)
    actual_volume = sum(mass) / reference_density
    equivalent_radius = cbrt(3actual_volume / (4pi))
    inferred_surface_tension = -reference_density * equivalent_radius^3 *
                               quadrupole_acceleration / (8quadrupole)
    center_of_mass_acceleration = vec(sum(acceleration .* reshape(mass, 1, :);
                                          dims=2) / sum(mass))
    torque_per_mass = [
        sum(mass .* (relative_coordinates[2, :] .* acceleration[3, :] .-
             relative_coordinates[3, :] .* acceleration[2, :])),
        sum(mass .* (relative_coordinates[3, :] .* acceleration[1, :] .-
             relative_coordinates[1, :] .* acceleration[3, :])),
        sum(mass .* (relative_coordinates[1, :] .* acceleration[2, :] .-
             relative_coordinates[2, :] .* acceleration[1, :]))] /
                      sum(mass)

    return (; inferred_surface_tension, equivalent_radius,
            particle_count=size(coordinates, 2), quadrupole,
            quadrupole_acceleration,
            center_of_mass_acceleration=norm(center_of_mass_acceleration),
            torque_per_mass=norm(torque_per_mass),
            acceleration_rms=sqrt(mean(abs2, acceleration)),
            velocity_rms=sqrt(mean(abs2, velocity)))
end

function calibration_models()
    return (("cohesion gamma=1", CohesionForceAkinci(;
                                                     surface_tension_coefficient=1.0)),
            ("Akinci gamma=1", SurfaceTensionAkinci(;
                                                    surface_tension_coefficient=1.0)),
            ("Morris sigma=1", SurfaceTensionMorris(;
                                                    surface_tension_coefficient=1.0)),
            ("momentum Morris sigma=1",
             SurfaceTensionMomentumMorris(;
                                          surface_tension_coefficient=1.0)),
            ("distributed Morris sigma=1",
             SurfaceTensionMorrisAkinci(;
                                        surface_tension_coefficient=1.0)))
end

function print_calibration(label, result)
    @printf("%-29s sigma_eff=%10.6f N/m  |a_cm|=%9.3e m/s^2  |tau|/m=%9.3e m^2/s^2  a_rms=%9.3e m/s^2\n",
            label, result.inferred_surface_tension,
            result.center_of_mass_acceleration, result.torque_per_mass,
            result.acceleration_rms)
end

function run_calibration_suite()
    for (label, model) in calibration_models()
        print_calibration(label, rayleigh_stiffness(model))
    end

    particle_spacing = cbrt(1.0e-6 / 750)
    support_radius = 2.8particle_spacing
    sigma_cohesion = akinci_cohesion_surface_tension(1.0, 1000.0,
                                                     support_radius)
    @printf("virial cohesion prediction       sigma=%10.6f N/m\n", sigma_cohesion)
    @printf("published adhesion work ratio   I_A/I_C=%10.6f\n",
            AKINCI_ADHESION_TO_COHESION_WORK_3D)
end

function calibration_model(model_name, coefficient; cohesion_coefficient=0.0)
    if model_name == "akinci"
        return SurfaceTensionAkinci(; surface_tension_coefficient=coefficient)
    elseif model_name == "cohesion"
        return CohesionForceAkinci(; surface_tension_coefficient=coefficient)
    elseif model_name == "morris"
        return SurfaceTensionMorris(; surface_tension_coefficient=coefficient)
    elseif model_name == "momentum_morris"
        return SurfaceTensionMomentumMorris(; surface_tension_coefficient=coefficient)
    elseif model_name == "hybrid"
        return SurfaceTensionMorrisAkinci(;
                                          surface_tension_coefficient=coefficient,
                                          cohesion_coefficient)
    end
    throw(ArgumentError("unknown model `$model_name`"))
end

function laplace_pressure_calibration(surface_tension; final_time=0.1,
                                      reference_density=1000.0,
                                      drop_volume=1.0e-6,
                                      target_particle_count=750,
                                      viscosity_alpha=0.1,
                                      ideal_density_threshold=0.95,
                                      interface_taper_start=0.8,
                                      support_taper_width=0.025,
                                      record_steps=false,
                                      cfl_number=0.65)
    initial_condition = deformed_drop(; stretch=1.0, reference_density,
                                      drop_volume, target_particle_count)
    particle_spacing = initial_condition.particle_spacing
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    support_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
    state_equation = StateEquationCole(; sound_speed=100.0,
                                       reference_density, exponent=7,
                                       clip_negative_pressure=true)
    surface_normal_method = if TrixiParticles.requires_surface_normal(surface_tension)
        ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf,
                                interface_threshold=0.01,
                                ideal_density_threshold,
                                interface_taper_start,
                                support_taper_width)
    else
        nothing
    end
    viscosity = ArtificialViscosityMonaghan(; alpha=viscosity_alpha, beta=0.0)
    fluid_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                               smoothing_length,
                                               density_calculator=ContinuityDensity(),
                                               density_diffusion=DensityDiffusionAntuono(;
                                                                                         delta=0.1),
                                               state_equation,
                                               viscosity, surface_tension,
                                               surface_normal_method,
                                               correction=AkinciFreeSurfaceCorrection(reference_density),
                                               reference_particle_spacing=particle_spacing)
    semi = Semidiscretization(fluid_system)
    ode = semidiscretize(semi, (0.0, final_time))
    dtmax = 5.0e-4
    initial_v_ode, initial_u_ode = ode.u0.x
    dt_reference = min(dtmax,
                       TrixiParticles.calculate_dt(initial_v_ode, initial_u_ode, cfl_number,
                                                   fluid_system, semi))
    callback = record_steps ? StepsizeCallback(; cfl=cfl_number) : nothing
    solution = nothing
    runtime = @elapsed solution = solve(ode, RDPK3SpFSAL35(); abstol=1.0e-7,
                                        reltol=1.0e-4, dtmax,
                                        save_everystep=record_steps,
                                        saveat=record_steps ? () : (final_time,), callback)
    v_ode, u_ode = last(solution.u).x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, final_time)
    v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
    u = TrixiParticles.wrap_u(u_ode, fluid_system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, fluid_system))
    velocity = Array(TrixiParticles.current_velocity(v, fluid_system))
    density = collect(TrixiParticles.current_density(v, fluid_system))
    pressure = [TrixiParticles.current_pressure(v, fluid_system, particle)
                for particle in TrixiParticles.eachparticle(fluid_system)]
    mass = fluid_system.mass
    center = vec(sum(coordinates .* reshape(mass, 1, :); dims=2) / sum(mass))
    radii = vec(sqrt.(sum(abs2, coordinates .- center; dims=1)))
    volume = sum(mass ./ density)
    equivalent_radius = cbrt(3volume / (4pi))
    interior = findall(<=(max(equivalent_radius - support_radius, 0.0)), radii)
    if length(interior) < 8
        interior = sortperm(radii)[1:min(20, length(radii))]
    end
    interior_pressure = pressure[interior]
    pressure_jump = median(interior_pressure)
    inferred_surface_tension = pressure_jump * equivalent_radius / 2
    speed = sqrt.(vec(sum(abs2, velocity; dims=1)))
    accepted_steps = solution.stats.naccept
    rejected_steps = solution.stats.nreject
    eta_p01 = NaN
    eta_median = NaN
    eta_tail_head = NaN
    if record_steps && length(solution.t) > 7
        accepted_dt = diff(solution.t)
        sample = accepted_dt[6:(end - 1)] ./ dt_reference
        eta_p01 = quantile(sample, 0.01)
        eta_median = median(sample)
        segment_length = max(1, floor(Int, 0.2length(sample)))
        eta_tail_head = median(last(sample, segment_length)) /
                        median(first(sample, segment_length))
    end
    has_activity = surface_tension isa SurfaceTensionMorris ||
                   surface_tension isa SurfaceTensionMomentumMorris
    activity = has_activity ? fluid_system.cache.interface_activity : Float64[]
    support_moment = surface_tension isa SurfaceTensionMorris ?
                     fluid_system.cache.support_moment :
                     surface_tension isa SurfaceTensionMomentumMorris ?
                     fluid_system.cache.divergence_correction : Float64[]
    return (; inferred_surface_tension, pressure_jump,
            pressure_mean=mean(interior_pressure),
            pressure_range=extrema(interior_pressure), equivalent_radius,
            interior_particles=length(interior), density_range=extrema(density),
            speed_rms=sqrt(mean(abs2, speed)), speed_max=maximum(speed),
            particle_count=length(density), runtime, accepted_steps, rejected_steps,
            dt_reference, eta_p01, eta_median, eta_tail_head,
            active_particles=count(>(0), activity),
            transition_particles=count(value -> 0 < value < 1, activity),
            activity_range=isempty(activity) ? (NaN, NaN) : extrema(activity),
            support_moment_range=isempty(support_moment) ? (NaN, NaN) :
                                 extrema(support_moment))
end

function print_laplace_calibration(model_name, coefficient, result)
    @printf("Laplace %-16s coefficient=%10.5g sigma_eff=%10.6f N/m dp=%10.3f Pa R=%8.5f m interior=%d rho=[%.3f, %.3f] vrms=%.4e vmax=%.4e\n",
            model_name, coefficient, result.inferred_surface_tension,
            result.pressure_jump, result.equivalent_radius,
            result.interior_particles, result.density_range...,
            result.speed_rms, result.speed_max)
    @printf("  runtime=%.2f s steps=%d/%d dt_ref=%.3e eta[p01,median,tail/head]=[%.3f, %.3f, %.3f] active=%d transition=%d q=[%.4f, %.4f]\n",
            result.runtime, result.accepted_steps, result.rejected_steps,
            result.dt_reference, result.eta_p01, result.eta_median,
            result.eta_tail_head, result.active_particles, result.transition_particles,
            result.support_moment_range...)
end

function laplace_pressure_series(surface_tension; final_time=0.02,
                                 volume_factors=(0.5, 1.0, 2.0),
                                 base_target_particle_count=750,
                                 viscosity_alpha=0.1,
                                 ideal_density_threshold=0.95,
                                 interface_taper_start=0.8,
                                 support_taper_width=0.025,
                                 record_steps=false)
    results = map(volume_factors) do factor
        drop_volume = factor * 1.0e-6
        target_particle_count = round(Int, factor * base_target_particle_count)
        laplace_pressure_calibration(surface_tension; final_time, drop_volume,
                                     target_particle_count, viscosity_alpha,
                                     ideal_density_threshold,
                                     interface_taper_start, support_taper_width,
                                     record_steps)
    end
    inverse_radius = [2 / result.equivalent_radius for result in results]
    pressure = [result.pressure_jump for result in results]
    mean_inverse_radius = mean(inverse_radius)
    mean_pressure = mean(pressure)
    surface_tension = sum((inverse_radius .- mean_inverse_radius) .*
                          (pressure .- mean_pressure)) /
                      sum(abs2, inverse_radius .- mean_inverse_radius)
    bulk_pressure = mean_pressure - surface_tension * mean_inverse_radius
    fitted_pressure = bulk_pressure .+ surface_tension .* inverse_radius
    residual_rms = sqrt(mean(abs2, pressure .- fitted_pressure))
    return (; results, volume_factors, surface_tension, bulk_pressure, residual_rms)
end

function print_laplace_series(model_name, coefficient, series)
    for (factor, result) in zip(series.volume_factors, series.results)
        @printf("  V=%4.1f cm^3  n=%4d  R=%8.5f m  dp=%10.3f Pa  vrms=%9.3e m/s runtime=%6.2f steps=%d/%d eta01=%6.3f\n",
                factor, result.particle_count, result.equivalent_radius,
                result.pressure_jump, result.speed_rms, result.runtime,
                result.accepted_steps, result.rejected_steps, result.eta_p01)
    end
    @printf("Laplace slope %-10s coefficient=%10.5g sigma_eff=%10.6f N/m p_bulk=%10.3f Pa residual_rms=%8.3f Pa\n",
            model_name, coefficient, series.surface_tension,
            series.bulk_pressure, series.residual_rms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        run_calibration_suite()
    else
        length(ARGS) in (4, 5) ||
            error("usage: surface_tension_calibration.jl laplace MODEL COEFFICIENT " *
                  "FINAL_TIME [COHESION_COEFFICIENT]")
        ARGS[1] in ("laplace", "laplace_series") ||
            error("unknown calibration `$(ARGS[1])`")
        model_name = ARGS[2]
        coefficient = parse(Float64, ARGS[3])
        final_time = parse(Float64, ARGS[4])
        cohesion_coefficient = length(ARGS) == 5 ? parse(Float64, ARGS[5]) : 0.0
        model = calibration_model(model_name, coefficient; cohesion_coefficient)
        if ARGS[1] == "laplace"
            result = laplace_pressure_calibration(model; final_time)
            print_laplace_calibration(model_name, coefficient, result)
        else
            series = laplace_pressure_series(model; final_time)
            print_laplace_series(model_name, coefficient, series)
        end
    end
end
