module SurfaceTensionValidation

using LinearAlgebra
using OrdinaryDiffEqLowStorageRK
using Statistics
using TrixiParticles

export matched_2d_targets, observed_order, quadratic_peak_times, rayleigh_mode2,
       rayleigh_mode2_stiffness, fit_angular_frequency, young_laplace_operator_fit,
       young_laplace_series

function spherical_drop_initial_condition(ndims, target_particle_count;
                                          radius=0.006, reference_density=1000.0,
                                          surface_tension_coefficient=1.0,
                                          stretch=1.0,
                                          initialize_laplace_pressure=false)
    volume = ndims == 2 ? pi * radius^2 : 4pi * radius^3 / 3
    particle_spacing = (volume / target_particle_count)^(1 / ndims)
    center = ntuple(_ -> 0.0, ndims)
    sphere_type = ndims == 2 ? RoundSphere() : VoxelSphere()
    shape = SphereShape(particle_spacing, radius + particle_spacing / 2, center,
                        reference_density; sphere_type)
    coordinates = copy(shape.coordinates)
    coordinates[1, :] .*= stretch
    if ndims == 2
        coordinates[2, :] ./= stretch
    else
        coordinates[2, :] ./= sqrt(stretch)
        coordinates[3, :] ./= sqrt(stretch)
    end

    pressure_jump = (ndims - 1) * surface_tension_coefficient / radius
    state_equation = StateEquationCole(; sound_speed=100.0, reference_density,
                                       exponent=7, clip_negative_pressure=true)
    density = initialize_laplace_pressure ?
              TrixiParticles.inverse_state_equation(state_equation, pressure_jump) :
              reference_density
    initial_condition = InitialCondition(; coordinates, velocity=zero(shape.velocity),
                                         mass=shape.mass,
                                         density=fill(density, size(coordinates, 2)),
                                         particle_spacing)
    return (; initial_condition, state_equation, particle_spacing)
end

function css_system(initial_condition, state_equation;
                    surface_tension_coefficient=1.0, viscosity=nothing,
                    density_diffusion=nothing, source_terms=nothing,
                    ideal_density_threshold=0.95, shifting_technique=nothing,
                    pressure_acceleration=nothing)
    particle_spacing = initial_condition.particle_spacing
    smoothing_kernel = WendlandC2Kernel{size(initial_condition.coordinates, 1)}()
    smoothing_length = 1.4particle_spacing
    normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf,
                                            interface_threshold=0.01,
                                            ideal_density_threshold)
    surface_tension = SurfaceTensionMomentumMorris(; surface_tension_coefficient)
    return WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                       smoothing_length,
                                       density_calculator=ContinuityDensity(),
                                       state_equation, viscosity, density_diffusion,
                                       pressure_acceleration,
                                       shifting_technique,
                                       surface_tension,
                                       surface_normal_method=normal_method,
                                       reference_particle_spacing=particle_spacing,
                                       source_terms)
end

function initial_acceleration(system)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)
    acceleration = GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        dv = zeros(eltype(v), size(v))
        TrixiParticles.reset_interaction_caches!(semi)
        TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)
        Array(dv[1:ndims(system), :])
    end
    return acceleration, system
end

function equivalent_radius(volume, ndims)
    return ndims == 2 ? sqrt(volume / pi) : cbrt(3volume / (4pi))
end

function exact_surface_measure(radius, ndims)
    return ndims == 2 ? 2pi * radius : 4pi * radius^2
end

function young_laplace_operator_fit(ndims, target_particle_count;
                                    radius=0.006, reference_density=1000.0,
                                    surface_tension_coefficient=1.0,
                                    pressure_basis=1.0)
    setup = spherical_drop_initial_condition(ndims, target_particle_count; radius,
                                             reference_density,
                                             surface_tension_coefficient)
    (; initial_condition, state_equation, particle_spacing) = setup
    system = css_system(initial_condition, state_equation; surface_tension_coefficient)
    capillary_acceleration, system = initial_acceleration(system)

    sound_speed = 100.0
    pressure_reference_density = reference_density - pressure_basis / sound_speed^2
    pressure_state_equation = StateEquationCole(; sound_speed,
                                                reference_density=pressure_reference_density,
                                                exponent=1)
    pressure_system = WeaklyCompressibleSPHSystem(initial_condition;
                                                  smoothing_kernel=WendlandC2Kernel{ndims}(),
                                                  smoothing_length=1.4particle_spacing,
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation=pressure_state_equation)
    pressure_acceleration, _ = initial_acceleration(pressure_system)
    pressure_acceleration ./= pressure_basis

    interface = findall(>(0), system.cache.delta_s)
    capillary = vec(capillary_acceleration[:, interface])
    unit_pressure = vec(pressure_acceleration[:, interface])
    pressure_jump = -dot(capillary, unit_pressure) / dot(unit_pressure, unit_pressure)
    residual = capillary + pressure_jump * unit_pressure

    mass = system.mass
    volume = sum(mass) / reference_density
    radius_discrete = equivalent_radius(volume, ndims)
    sigma_fit = pressure_jump * radius_discrete / (ndims - 1)
    represented_surface = sum(mass .* system.cache.delta_s) / reference_density
    exact_surface = exact_surface_measure(radius_discrete, ndims)
    coordinates = initial_condition.coordinates
    center = vec(sum(coordinates .* reshape(mass, 1, :); dims=2) / sum(mass))
    relative_coordinates = coordinates .- center
    virial = -sum(mass .* vec(sum(relative_coordinates .* capillary_acceleration; dims=1)))
    sigma_virial = virial / ((ndims - 1) * exact_surface)
    total_force = vec(sum(capillary_acceleration .* reshape(mass, 1, :); dims=2))

    return (; ndims, particle_count=nparticles(system), target_particle_count,
            particle_spacing, radius=radius_discrete,
            interface_particles=length(interface), pressure_jump, sigma_fit,
            sigma_virial, relative_error=abs(sigma_fit / surface_tension_coefficient - 1),
            residual=norm(residual) / norm(capillary),
            surface_measure_ratio=represented_surface / exact_surface,
            total_force=norm(total_force))
end

function observed_order(results)
    spacings = [result.particle_spacing for result in results]
    errors = [max(result.relative_error, eps()) for result in results]
    x = log.(spacings)
    y = log.(errors)
    x_mean = mean(x)
    y_mean = mean(y)
    return sum((x .- x_mean) .* (y .- y_mean)) / sum(abs2, x .- x_mean)
end

function matched_2d_targets(targets_3d)
    return [round(Int, pi / (4pi / (3target))^(2 / 3)) for target in targets_3d]
end

function young_laplace_series(ndims;
                              targets_3d=(375, 750, 1500, 3000), kwargs...)
    targets = ndims == 2 ? matched_2d_targets(targets_3d) : collect(targets_3d)
    results = [young_laplace_operator_fit(ndims, target; kwargs...) for target in targets]
    return (; ndims, targets, results, observed_order=observed_order(results))
end

function signed_axes(state, system, semi)
    v_ode, u_ode = state.x
    u = TrixiParticles.wrap_u(u_ode, system, semi)
    coordinates = Array(TrixiParticles.current_coordinates(u, system))
    mass = system.mass
    center = vec(sum(coordinates .* reshape(mass, 1, :); dims=2) / sum(mass))
    relative = coordinates .- center
    covariance = (relative .* reshape(mass, 1, :)) * transpose(relative) / sum(mass)
    axis_x = 2sqrt(max(covariance[1, 1], zero(eltype(covariance))))
    axis_y = 2sqrt(max(covariance[2, 2], zero(eltype(covariance))))
    return axis_x, axis_y
end

function quadratic_peak_times(time, signal)
    peaks = Float64[]
    for index in 2:(length(signal) - 1)
        signal[index] > signal[index - 1] || continue
        signal[index] >= signal[index + 1] || continue
        y_left = signal[index - 1]
        y_center = signal[index]
        y_right = signal[index + 1]
        denominator = y_left - 2y_center + y_right
        offset = iszero(denominator) ? 0.0 :
                 0.5 * (y_left - y_right) / denominator
        dt = (time[index + 1] - time[index - 1]) / 2
        push!(peaks, time[index] + offset * dt)
    end
    return peaks
end

function fit_angular_frequency(time, signal, omega_reference;
                               minimum_ratio=0.5, maximum_ratio=1.5,
                               samples=1001)
    frequencies = range(minimum_ratio * omega_reference,
                        maximum_ratio * omega_reference; length=samples)
    best_frequency = first(frequencies)
    best_residual = Inf
    centered_time = time .- first(time)
    for omega in frequencies
        design = hcat(ones(length(time)), centered_time,
                      cos.(omega .* centered_time), sin.(omega .* centered_time))
        coefficients = design \ signal
        residual = sum(abs2, design * coefficients - signal)
        if residual < best_residual
            best_residual = residual
            best_frequency = omega
        end
    end
    return (; omega=best_frequency,
            residual=best_residual / sum(abs2, signal .- mean(signal)))
end

function rayleigh_mode2_stiffness(target_particle_count;
                                  radius=0.01, reference_density=1000.0,
                                  surface_tension_coefficient=1.0, stretch=1.02,
                                  viscosity_alpha=0.0)
    setup = spherical_drop_initial_condition(2, target_particle_count; radius,
                                             reference_density,
                                             surface_tension_coefficient, stretch,
                                             initialize_laplace_pressure=true)
    (; initial_condition, state_equation, particle_spacing) = setup
    viscosity = ArtificialViscosityMonaghan(; alpha=viscosity_alpha, beta=0.0)
    system = css_system(initial_condition, state_equation; surface_tension_coefficient,
                        viscosity)
    acceleration, system = initial_acceleration(system)
    coordinates = initial_condition.coordinates
    mass = system.mass
    center = vec(sum(coordinates .* reshape(mass, 1, :); dims=2) / sum(mass))
    relative = coordinates .- center
    quadrupole = sum(mass .* (relative[1, :] .^ 2 .- relative[2, :] .^ 2)) /
                 sum(mass)
    quadrupole_acceleration = 2sum(mass .* (relative[1, :] .* acceleration[1, :] .-
                                    relative[2, :] .* acceleration[2, :])) /
                              sum(mass)
    omega_squared = -quadrupole_acceleration / quadrupole
    area = sum(mass) / reference_density
    radius_discrete = equivalent_radius(area, 2)
    omega_exact = sqrt(6surface_tension_coefficient /
                       (reference_density * radius_discrete^3))
    omega_measured = sqrt(max(omega_squared, zero(omega_squared)))
    return (; target_particle_count, particle_count=nparticles(system), particle_spacing,
            radius=radius_discrete, quadrupole, quadrupole_acceleration,
            omega_squared, omega_exact, omega_measured,
            frequency_error=abs(omega_measured / omega_exact - 1),
            acceleration_rms=sqrt(mean(abs2, acceleration)))
end

function rayleigh_mode2(target_particle_count;
                        radius=0.01, reference_density=1000.0,
                        surface_tension_coefficient=1.0, stretch=1.0,
                        mode_amplitude=0.02, periods=1.2,
                        viscosity_alpha=0.05)
    setup = spherical_drop_initial_condition(2, target_particle_count; radius,
                                             reference_density,
                                             surface_tension_coefficient, stretch,
                                             initialize_laplace_pressure=true)
    (; initial_condition, state_equation, particle_spacing) = setup
    area = sum(initial_condition.mass) / reference_density
    radius_discrete = equivalent_radius(area, 2)
    omega_exact = sqrt(6surface_tension_coefficient /
                       (reference_density * radius_discrete^3))
    velocity = zeros(size(initial_condition.velocity))
    velocity[1, :] .= mode_amplitude * omega_exact .* initial_condition.coordinates[1, :]
    velocity[2, :] .= -mode_amplitude * omega_exact .* initial_condition.coordinates[2, :]
    initial_condition = InitialCondition(; coordinates=initial_condition.coordinates,
                                         velocity, mass=initial_condition.mass,
                                         density=initial_condition.density,
                                         particle_spacing)
    viscosity = ArtificialViscosityMonaghan(; alpha=viscosity_alpha, beta=0.0)
    density_diffusion = DensityDiffusionAntuono(; delta=0.05)
    system = css_system(initial_condition, state_equation; surface_tension_coefficient,
                        viscosity, density_diffusion)
    semi = Semidiscretization(system; parallelization_backend=SerialBackend())

    period_exact = 2pi / omega_exact
    final_time = periods * period_exact
    ode = semidiscretize(semi, (0.0, final_time))
    capillary_dt = sqrt(reference_density * (1.4particle_spacing)^3 /
                        (2pi * surface_tension_coefficient))
    saveat = range(0.0, final_time; step=period_exact / 50)
    solution = nothing
    runtime = @elapsed solution = solve(ode, RDPK3SpFSAL35(); abstol=1.0e-8,
                                        reltol=2.0e-5,
                                        dtmax=min(period_exact / 120, capillary_dt),
                                        save_everystep=false, saveat)

    axes = [signed_axes(state, system, semi) for state in solution.u]
    axis_x = first.(axes)
    axis_y = last.(axes)
    deformation = axis_x .- axis_y
    peak_times = quadratic_peak_times(solution.t, deformation)
    fit = fit_angular_frequency(solution.t, deformation, omega_exact)
    omega_measured = fit.omega
    measured_period = 2pi / omega_measured
    frequency_error = abs(omega_measured / omega_exact - 1)
    midpoint = length(deformation) ÷ 2
    damping_ratio = std(deformation[(midpoint + 1):end]) /
                    std(deformation[1:midpoint])

    return (; target_particle_count, particle_count=nparticles(system), particle_spacing,
            radius=radius_discrete, omega_exact, omega_measured, period_exact,
            measured_period, frequency_error, fit_residual=fit.residual,
            damping_ratio, runtime,
            accepted_steps=solution.stats.naccept, rejected_steps=solution.stats.nreject,
            time=collect(solution.t), axis_x, axis_y, deformation,
            peak_times)
end

end # module
