using LinearAlgebra
using Printf
using Statistics
using TrixiParticles

include(joinpath(@__DIR__, "surface_tension_calibration.jl"))

function initial_pair_acceleration(system)
    semi = Semidiscretization(system)
    ode = semidiscretize(semi, (0.0, 0.01))
    v_ode, u_ode = ode.u0.x
    TrixiParticles.update_systems_and_nhs(v_ode, u_ode, semi, 0.0)

    acceleration = GC.@preserve v_ode u_ode begin
        v = TrixiParticles.wrap_v(v_ode, system, semi)
        u = TrixiParticles.wrap_u(u_ode, system, semi)
        dv = zeros(eltype(v), size(v))
        TrixiParticles.interact!(dv, v, u, v, u, system, system, semi)
        Array(dv[1:ndims(system), :])
    end
    return acceleration, system
end

function css_static_laplace_balance(; target_particle_count=750,
                                    surface_tension_coefficient=1.0,
                                    reference_density=1000.0,
                                    pressure_basis=1.0,
                                    ideal_density_threshold=0.95,
                                    interface_taper_start=0.8,
                                    support_taper_width=0.025)
    initial_condition = deformed_drop(; stretch=1.0, reference_density,
                                      target_particle_count)
    particle_spacing = initial_condition.particle_spacing
    smoothing_kernel = WendlandC2Kernel{3}()
    smoothing_length = 1.4particle_spacing
    normal_method = ColorfieldSurfaceNormal(; boundary_contact_threshold=Inf,
                                            interface_threshold=0.01,
                                            ideal_density_threshold,
                                            interface_taper_start,
                                            support_taper_width)

    css_state_equation = StateEquationCole(; sound_speed=100.0, reference_density,
                                           exponent=1)
    css_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                             smoothing_length,
                                             density_calculator=ContinuityDensity(),
                                             state_equation=css_state_equation,
                                             surface_tension=SurfaceTensionMomentumMorris(;
                                                                                          surface_tension_coefficient),
                                             surface_normal_method=normal_method,
                                             reference_particle_spacing=particle_spacing)
    css_acceleration, css_system = initial_pair_acceleration(css_system)

    # With exponent one, Cole's equation reduces exactly to
    # p = c^2 (rho - rho_0). Adjusting rho_0 therefore provides a uniform pressure basis.
    sound_speed = 100.0
    pressure_reference_density = reference_density - pressure_basis / sound_speed^2
    pressure_state_equation = StateEquationCole(; sound_speed,
                                                reference_density=pressure_reference_density,
                                                exponent=1)
    pressure_system = WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                                  smoothing_length,
                                                  density_calculator=ContinuityDensity(),
                                                  state_equation=pressure_state_equation)
    pressure_acceleration, _ = initial_pair_acceleration(pressure_system)
    pressure_acceleration ./= pressure_basis

    interface = findall(>(0), css_system.cache.delta_s)
    capillary = vec(css_acceleration[:, interface])
    unit_pressure = vec(pressure_acceleration[:, interface])
    pressure_jump = -dot(capillary, unit_pressure) / dot(unit_pressure, unit_pressure)
    residual = capillary + pressure_jump * unit_pressure

    mass = css_system.mass
    volume = sum(mass) / reference_density
    equivalent_radius = cbrt(3volume / (4pi))
    inferred_surface_tension = pressure_jump * equivalent_radius / 2
    surface_area = sum(mass .* css_system.cache.delta_s) / reference_density
    surface_area_ratio = surface_area / (4pi * equivalent_radius^2)
    coordinates = initial_condition.coordinates
    center = vec(sum(coordinates .* reshape(mass, 1, :); dims=2) / sum(mass))
    relative_coordinates = coordinates .- center
    capillary_virial = -sum(mass .* vec(sum(relative_coordinates .* css_acceleration;
                                    dims=1)))
    virial_surface_tension = capillary_virial /
                             (2 * 4pi * equivalent_radius^2)
    total_capillary_force = vec(sum(css_acceleration .* reshape(mass, 1, :); dims=2))
    correction_range = extrema(css_system.cache.divergence_correction[interface])

    return (; particle_count=nparticles(css_system), interface_particles=length(interface),
            pressure_jump, equivalent_radius, inferred_surface_tension,
            surface_area_ratio, virial_surface_tension,
            relative_residual=norm(residual) / norm(capillary),
            correction_range,
            total_capillary_force=norm(total_capillary_force),
            acceleration_rms=sqrt(mean(abs2, capillary)))
end

function print_css_balance(result)
    @printf("CSS static n=%5d interface=%4d sigma_fit=%8.5f sigma_virial=%8.5f N/m dp=%9.3f Pa R=%8.5f m A/A0=%7.4f q=[%6.3f,%6.3f] residual=%8.3e |F|=%8.3e N a_rms=%8.3e m/s^2\n",
            result.particle_count, result.interface_particles,
            result.inferred_surface_tension, result.virial_surface_tension,
            result.pressure_jump,
            result.equivalent_radius, result.surface_area_ratio,
            result.correction_range..., result.relative_residual,
            result.total_capillary_force, result.acceleration_rms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    target_counts = isempty(ARGS) ? (375, 750, 1500) : parse.(Int, ARGS)
    for target_particle_count in target_counts
        print_css_balance(css_static_laplace_balance(; target_particle_count))
    end
end
