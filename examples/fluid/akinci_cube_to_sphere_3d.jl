# ==========================================================================================
# 3D Akinci Cube-to-Sphere Shootout
#
# An initially cubical free drop contracts under surface tension. The setup can be run with
# WCSPH, EDAC, or IISPH and any Akinci-family surface tension model. It is based on Figure 2
# of Akinci, Akinci, and Teschner (2013), https://doi.org/10.1145/2508363.2508395.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSymplecticRK
using LinearAlgebra: dot, norm
using Statistics: mean, std

# `sph_method` can be `"wcsph"`, `"edac"`, or `"iisph"`.
sph_method = "wcsph"

fluid_density = 1000.0
cube_side_length = 0.01
particles_per_dimension = 8
particle_spacing = cube_side_length / particles_per_dimension

smoothing_kernel = SchoenbergCubicSplineKernel{3}()
smoothing_length_factor = 1.0
smoothing_length = smoothing_length_factor * particle_spacing
sound_speed = 20.0
cfl_number = 0.2
nu = 1.0e-5
viscosity = ViscosityMorris(; nu)

surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=1.0)
density_calculator = SummationDensity()
correction = AkinciFreeSurfaceCorrection(fluid_density)
pressure_acceleration = sph_method == "edac" ?
                        TrixiParticles.inter_particle_averaged_pressure : nothing

time_step = cfl_number * smoothing_length / sound_speed
# Let the initial capillary transient decay before comparing the final shape.
tspan = (0.0, 1.0)
analysis_saveat = nothing
maxiters = 10^7
iisph_min_iterations = 2
iisph_max_iterations = 20

n_cube = ntuple(_ -> particles_per_dimension, 3)
cube_min = ntuple(_ -> -cube_side_length / 2, 3)
drop = RectangularShape(particle_spacing, n_cube, cube_min; density=fluid_density)

if sph_method == "wcsph"
    state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                       exponent=7, clip_negative_pressure=true)
    fluid_system = WeaklyCompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                               density_calculator, state_equation,
                                               viscosity,
                                               pressure_acceleration, surface_tension,
                                               correction,
                                               reference_particle_spacing=particle_spacing)
elseif sph_method == "edac"
    fluid_system = EntropicallyDampedSPHSystem(drop; smoothing_kernel, smoothing_length,
                                               sound_speed, density_calculator, viscosity,
                                               pressure_acceleration, correction,
                                               surface_tension,
                                               reference_particle_spacing=particle_spacing)
elseif sph_method == "iisph"
    density_calculator isa SummationDensity ||
        throw(ArgumentError("IISPH only supports SummationDensity"))
    fluid_system = ImplicitIncompressibleSPHSystem(drop; smoothing_kernel, smoothing_length,
                                                   reference_density=fluid_density,
                                                   viscosity, correction,
                                                   surface_tension,
                                                   reference_particle_spacing=particle_spacing,
                                                   min_iterations=iisph_min_iterations,
                                                   max_iterations=iisph_max_iterations,
                                                   time_step)
else
    throw(ArgumentError("`sph_method` must be `\"wcsph\"`, `\"edac\"`, or `\"iisph\"`"))
end

semi = Semidiscretization(fluid_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=max(time_step, (tspan[2] - tspan[1]) / 20))
callbacks = CallbackSet(info_callback, saving_callback)
save_options = isnothing(analysis_saveat) ? (; save_everystep=false, maxiters) :
               (; save_everystep=false, saveat=analysis_saveat, maxiters)

if sph_method == "iisph"
    sol = solve(ode, SymplecticEuler(); dt=time_step, adaptive=false,
                callback=callbacks, save_options...)
else
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false); dt=time_step,
                adaptive=false, callback=callbacks, save_options...)
end

# Common diagnostics used by the validation shootout.
v_final_ode, u_final_ode = sol.u[end].x
v_final = TrixiParticles.wrap_v(v_final_ode, fluid_system, semi)
u_final = TrixiParticles.wrap_u(u_final_ode, fluid_system, semi)
coordinates_final = collect(TrixiParticles.current_coordinates(u_final, fluid_system))
velocity_final = collect(TrixiParticles.current_velocity(v_final, fluid_system))

mass_weights = reshape(fluid_system.mass, 1, :)
total_mass = sum(fluid_system.mass)
initial_center = vec(sum(drop.coordinates .* mass_weights; dims=2) / total_mass)
final_center = vec(sum(coordinates_final .* mass_weights; dims=2) / total_mass)
initial_radii = [norm(drop.coordinates[:, particle] - initial_center)
                 for particle in axes(drop.coordinates, 2)]
final_radii = [norm(coordinates_final[:, particle] - final_center)
               for particle in axes(coordinates_final, 2)]
total_momentum = vec(sum(velocity_final .* mass_weights; dims=2))
kinetic_energy = sum(0.5 * fluid_system.mass[particle] *
                     dot(velocity_final[:, particle], velocity_final[:, particle])
                     for particle in eachindex(fluid_system.mass))
coordinate_history = map(sol.u) do state
    _, u_saved_ode = state.x
    u_saved = TrixiParticles.wrap_u(u_saved_ode, fluid_system, semi)
    collect(TrixiParticles.current_coordinates(u_saved, fluid_system))
end

cube_to_sphere_metrics = (;
                          particle_count=nparticles(fluid_system),
                          initial_radial_cv=std(initial_radii; corrected=false) /
                                            mean(initial_radii),
                          final_radial_cv=std(final_radii; corrected=false) /
                                          mean(final_radii),
                          mean_radius_ratio=mean(final_radii) / mean(initial_radii),
                          center_of_mass_drift=norm(final_center - initial_center),
                          momentum_norm=norm(total_momentum),
                          kinetic_energy)
cube_to_sphere_result = (; metrics=cube_to_sphere_metrics,
                         retcode=string(sol.retcode),
                         initial_coordinates=drop.coordinates,
                         final_coordinates=coordinates_final,
                         times=collect(sol.t), coordinate_history)
