# Channel ﬂow through periodic array of cylinders
#
# S. Adami et al
# "A transport-velocity formulation for smoothed particle hydrodynamics".
# In: Journal of Computational Physics, Volume 241 (2013), pages 292-307.
# https://doi.org/10.1016/j.jcp.2013.01.043

using TrixiParticles
using OrdinaryDiffEq

acceleration_x = 2.5e-4

# ==========================================================================================
# ==== Fluid

n_particles_x = 144

cylinder_radius = 0.02
box_length = 6cylinder_radius
box_height = 4cylinder_radius

fluid_density = 1000.0
nu = 0.1 / fluid_density

velocity_ref = 1.2e-4

# Adami uses c = 0.1*sqrt(acceleration_x*cylinder_radius)  but from the original setup
# from M. Ellero and N. A. Adams (https://doi.org/10.1002/nme.3088) is using c = 0.02
sound_speed = 0.02

pressure = sound_speed^2 * fluid_density

particle_spacing = box_length / n_particles_x
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergQuarticSplineKernel{2}()

box = RectangularTank(particle_spacing, (box_length, box_height), (box_length, box_height),
                      fluid_density, n_layers=3, spacing_ratio=1.0, pressure=pressure,
                      faces=(false, false, true, true))

cylinder = SphereShape(particle_spacing, cylinder_radius, (box_length / 2, box_height / 2),
                       fluid_density, sphere_type=RoundSphere())

fluid = setdiff(box.fluid, cylinder)
boundary = union(cylinder, box.boundary)

boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             viscosity=ViscosityAdami(nu),
                                             smoothing_kernel, smoothing_length)

fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=ViscosityAdami(nu),
                                           transport_velocity=TransportVelocityAdami(pressure),
                                           acceleration=(acceleration_x, 0.0))

boundary_system = BoundarySPHSystem(boundary, boundary_model)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch,
                          periodic_box_min_corner=[0.0, -box_length * 2],
                          periodic_box_max_corner=[box_length, box_length * 2])

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49((step_limiter!)=TrixiParticles.update_transport_velocity!),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
