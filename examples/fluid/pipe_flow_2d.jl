# TODO: Description
using TrixiParticles
using OrdinaryDiffEq

gravity = 0.0

# ==========================================================================================
# ==== Fluid

domain_length = 1.0
domain_width = 0.2

particle_spacing = 0.01 * domain_length

water_density = 1000.0
pressure = 0.0

prescribed_velocity = (1.0, 0.0)

sound_speed = 10 * maximum(prescribed_velocity)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, water_density, 1.0,
                                   background_pressure=1.0)

viscosity = ArtificialViscosityMonaghan(0.2, 0.0)

n_particles_x = Int(floor(domain_length / particle_spacing))
n_particles_y = Int(floor(domain_width / particle_spacing))
n_buffer_particles = 4 * n_particles_y

fluid = RectangularShape(particle_spacing, (n_particles_x, n_particles_y), (0.0, 0.0),
                         buffer=n_buffer_particles,
                         water_density, init_velocity=prescribed_velocity,
                         pressure=pressure)

# ==========================================================================================
# ==== Open Boundary
open_boundary_cols = 4

length_open_boundary = particle_spacing * open_boundary_cols
height_open_boundary = particle_spacing * n_particles_y

end_inflow_zone = 0.0 - particle_spacing / 2

start_outflow_zone = n_particles_x * particle_spacing #- particle_spacing / 2
end_outflow_zone = start_outflow_zone + length_open_boundary

zone_origin_in = (end_inflow_zone - length_open_boundary, 0.0)
zone_origin_out = ((start_outflow_zone - particle_spacing / 2), 0.0)

inflow = RectangularShape(particle_spacing, (open_boundary_cols, n_particles_y),
                          (-length_open_boundary, 0.0), water_density,
                          buffer=n_buffer_particles,
                          init_velocity=prescribed_velocity, pressure=pressure)
outflow = RectangularShape(particle_spacing, (open_boundary_cols, n_particles_y),
                           (start_outflow_zone, 0.0), water_density,
                           buffer=n_buffer_particles,
                           init_velocity=prescribed_velocity, pressure=pressure)

# First vector is in down stream direction
zone_points_in = ([end_inflow_zone; 0.0], [end_inflow_zone; height_open_boundary])
zone_points_out = ([end_outflow_zone; 0.0], [end_outflow_zone; height_open_boundary])

# ==========================================================================================
# ==== Boundary
n_boundary_particles_x = n_particles_x + 2 * (open_boundary_cols + 1) + 10
boundary_layers = 3

bottom_wall = RectangularShape(particle_spacing, (n_boundary_particles_x, boundary_layers),
                               (-10 * particle_spacing,
                                -boundary_layers * particle_spacing), water_density)
top_wall = RectangularShape(particle_spacing, (n_boundary_particles_x, boundary_layers),
                            (-10 * particle_spacing,
                             n_particles_y * particle_spacing), water_density)
boundary = InitialCondition(bottom_wall, top_wall)

wall_viscosity = ViscosityAdami(1e-4)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             state_equation, AdamiPressureExtrapolation(),
                                             viscosity=wall_viscosity,
                                             smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Systems

#fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(), state_equation,
#                                           smoothing_kernel, smoothing_length,
#                                           viscosity=viscosity,
#                                           acceleration=(0.0, gravity))
fluid_system = EntropicallyDampedSPH(fluid, smoothing_kernel, smoothing_length,
                                     sound_speed, viscosity=ViscosityAdami(0.05), #wall_viscosity,
                                     acceleration=(0.0, gravity))
open_boundary_in = OpenBoundarySPHSystem(inflow, InFlow(), sound_speed, zone_points_in,
                                         zone_origin_in, fluid_system)

open_boundary_out = OpenBoundarySPHSystem(outflow, OutFlow(), sound_speed, zone_points_out,
                                          zone_origin_out, fluid_system)

boundary_system = BoundarySPHSystem(boundary.coordinates, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          boundary_system,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback, UpdateAfterTimeStep())

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

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
