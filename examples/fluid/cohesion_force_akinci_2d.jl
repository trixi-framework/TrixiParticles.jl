# ==========================================================================================
# 2D Cohesion-Only Akinci Surface Force
#
# This example evolves a rectangular fluid patch with `CohesionForceAkinci`. The
# cohesion-only model does not calculate surface normals and therefore does not require
# `reference_particle_spacing`. In 2D, its coefficient is an empirical numerical parameter.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

particle_spacing = 0.025
fluid_size = (0.2, 0.1)
fluid_density = 1000.0
sound_speed = 20.0
tspan = (0.0, 0.2)

fluid = RectangularShape(particle_spacing,
                         round.(Int, fluid_size ./ particle_spacing),
                         zeros(length(fluid_size)); density=fluid_density)

smoothing_length = particle_spacing - eps()
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=true)

nu = 0.01
alpha = 8 * nu / (smoothing_length * sound_speed)
viscosity = ArtificialViscosityMonaghan(; alpha, beta=0.0)
surface_tension = CohesionForceAkinci(surface_tension_coefficient=0.001)

fluid_system = WeaklyCompressibleSPHSystem(fluid; smoothing_kernel, smoothing_length,
                                           density_calculator=SummationDensity(),
                                           state_equation, viscosity, surface_tension,
                                           source_terms=SourceTermDamping(damping_coefficient=0.5))

semi = Semidiscretization(fluid_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02)
stepsize_callback = StepsizeCallback(cfl=0.5)
callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, save_everystep=false, callback=callbacks)

v_ode, u_ode = sol.u[end].x
v = TrixiParticles.wrap_v(v_ode, fluid_system, semi)
velocity = TrixiParticles.current_velocity(v, fluid_system)
total_momentum = vec(sum(velocity .* transpose(fluid_system.mass); dims=2))
center_of_mass_velocity = total_momentum / sum(fluid_system.mass)
final_kinetic_energy = kinetic_energy(fluid_system, nothing, nothing, v_ode, u_ode, semi,
                                      sol.t[end])

@info "Cohesion diagnostics" center_of_mass_velocity final_kinetic_energy
