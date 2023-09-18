# TODO: Description
using TrixiParticles
using OrdinaryDiffEq
using LinearAlgebra

gravity = 0.0

ReynoldsNumber = 200

# ==========================================================================================
# ==== Fluid

sphere_diameter = 2.0
domain_length = 15sphere_diameter
domain_width = 15sphere_diameter

particle_spacing = 0.005domain_length#sphere_diameter/domain_length
open_boundary_cols = 6

water_density = 1000.0

prescribed_velocity = (1.0, 0.0)

sound_speed = 10 * maximum(prescribed_velocity)
background_pressure = water_density*sound_speed^2
pressure = background_pressure#0.0

nu = maximum(prescribed_velocity) * sphere_diameter / ReynoldsNumber
viscosity = ViscosityAdami(nu) #alpha * smoothing_length * sound_speed / 8)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()


pipe = RectangularTank(particle_spacing, (domain_length, domain_width),
                       (domain_length + particle_spacing * open_boundary_cols,
                        domain_width), water_density, pressure=pressure,
                       n_layers=3, spacing_ratio=1, faces=(false, false, true, true),
                       init_velocity=prescribed_velocity)

cylinder = SphereShape(particle_spacing, 0.5sphere_diameter,
                       (5sphere_diameter, 7.5sphere_diameter),
                       water_density, sphere_type=RoundSphere())

fluid = setdiff(pipe.fluid, cylinder)

# ==========================================================================================
# ==== Open Boundary

n_particles_y = Int(floor(pipe.fluid_size[2] / particle_spacing))

length_open_boundary = particle_spacing * open_boundary_cols
height_open_boundary = particle_spacing * n_particles_y

end_inflow_zone = 0.0 - particle_spacing / 2

zone_origin_in = (-length_open_boundary, 0.0)
zone_origin_out = (domain_length, 0.0)

inflow = RectangularShape(particle_spacing, (open_boundary_cols, n_particles_y),
                          zone_origin_in, water_density,
                          init_velocity=prescribed_velocity, pressure=pressure)
outflow = RectangularShape(particle_spacing, (open_boundary_cols, n_particles_y),
                           zone_origin_out, water_density,
                           init_velocity=prescribed_velocity, pressure=pressure)

zone_plane_in = ([0.0; 0.0], [0.0; height_open_boundary])
zone_plane_out = ([domain_length + length_open_boundary; 0.0],
                  [domain_length + length_open_boundary; height_open_boundary])

# ==========================================================================================
# ==== Boundary
boundary_layers = 3
wall_position = -open_boundary_cols * particle_spacing

bottom_wall = RectangularShape(particle_spacing, (open_boundary_cols, boundary_layers),
                               (wall_position, -boundary_layers * particle_spacing),
                               water_density)
top_wall = RectangularShape(particle_spacing, (open_boundary_cols, boundary_layers),
                            (wall_position, n_particles_y * particle_spacing),
                            water_density)
boundary = union(bottom_wall, top_wall, pipe.boundary, cylinder)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Systems
n_buffer_particles = 4 * n_particles_y

#state_equation = StateEquationCole(sound_speed, 7, water_density, 100_000.0,
#                                   background_pressure=100_000.0)
#fluid_system = WeaklyCompressibleSPHSystem(pipe.fluid, ContinuityDensity(), state_equation,
#                                           smoothing_kernel, smoothing_length,
#                                           buffer=n_buffer_particles,
#                                           viscosity=viscosity, acceleration=(0.0, gravity))
#
fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           transport_velocity=TransportVelocityAdami(background_pressure),
                                           buffer=n_buffer_particles,
                                           acceleration=(0.0, gravity))
#
open_boundary_in = OpenBoundarySPHSystem(inflow, InFlow(), sound_speed, zone_plane_in,
                                         buffer=n_buffer_particles,
                                         zone_origin_in, fluid_system)

open_boundary_out = OpenBoundarySPHSystem(outflow, OutFlow(), sound_speed, zone_plane_out,
                                          buffer=n_buffer_particles,
                                          zone_origin_out, fluid_system)

boundary_system = BoundarySPHSystem(boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
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
#sol = solve(ode, Euler(),
#            dt=1e-3,
#            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
#            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
#            dtmax=1e-3, # Limit stepsize to prevent crashing
#            save_everystep=false, callback=callbacks);

sol = solve(ode, RDPK3SpFSAL49((step_limiter!)=TrixiParticles.update_step!),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=5e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks)
