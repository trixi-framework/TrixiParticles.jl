# TODO: Description
using TrixiParticles
using OrdinaryDiffEq

gravity = 0.0

# ==========================================================================================
# ==== Fluid

domain_length = 10.0
inlet_width = 5.2
slip_length = 0.1 * domain_length
inlet_length = 0.15 * domain_length
step_height = 4.9
outlet_width = inlet_width + step_height
interior_length = 0.75 * domain_length

particle_spacing = 0.01 * domain_length
fluid_density = 1000.0
pressure = 0.0

prescribed_velocity = (2.0, 0.0)
initial_velocity = (0.0, 0.0)

u_max = max(0.1, maximum(prescribed_velocity))
u_mean = u_max * 2 / 3
sound_speed = 10 * u_max

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, fluid_density, 1.0,
                                   background_pressure=1.0)

alpha = 0.02
viscosity = ViscosityAdami(alpha * smoothing_length * sound_speed / 8) #ArtificialViscosityMonaghan(0.2, 0.0)

n_particles_interior_x = Int(floor(interior_length / particle_spacing))
n_particles_out_y = Int(floor(outlet_width / particle_spacing))

fluid_interior = RectangularShape(particle_spacing,
                                  (n_particles_interior_x, n_particles_out_y),
                                  (inlet_length, 0.0), fluid_density,
                                  init_velocity=initial_velocity, pressure=pressure)

n_particles_inlet_x = Int(floor(inlet_length / particle_spacing))
n_particles_in_y = Int(floor(inlet_width / particle_spacing))

fluid_inlet = RectangularShape(particle_spacing, (n_particles_inlet_x, n_particles_in_y),
                               (0.0, step_height), fluid_density,
                               init_velocity=initial_velocity, pressure=pressure)

n_particles_slip_x = Int(floor(slip_length / particle_spacing))

fluid_slip = RectangularShape(particle_spacing, (n_particles_slip_x, n_particles_in_y),
                              (-n_particles_slip_x * particle_spacing, step_height),
                              fluid_density, init_velocity=initial_velocity,
                              pressure=pressure)

n_buffer_particles_fluid = 200 * n_particles_in_y

fluid = InitialCondition(fluid_slip, fluid_inlet, fluid_interior,
                         buffer=n_buffer_particles_fluid)

# ==========================================================================================
# ==== Open Boundary

open_bounary_cols = 10
length_open_boundary = particle_spacing * open_bounary_cols
height_open_boundary_in = particle_spacing * n_particles_in_y
height_open_boundary_out = particle_spacing * n_particles_out_y

# extrude in upstream direction
end_inflow_zone = 0.0 - slip_length
start_inflow_zone = end_inflow_zone - length_open_boundary

# extrude in downstream direction
start_outflow_zone = inlet_length + interior_length
end_outflow_zone = start_outflow_zone + length_open_boundary

zone_origin_in = (end_inflow_zone - length_open_boundary - particle_spacing / 2,
                  step_height - particle_spacing / 2)
zone_origin_out = ((start_outflow_zone - particle_spacing / 2), 0.0)

inflow = RectangularShape(particle_spacing, (open_bounary_cols, n_particles_in_y),
                          (start_inflow_zone, step_height), fluid_density,
                          init_velocity=prescribed_velocity,
                          buffer=n_buffer_particles_fluid, pressure=pressure)
outflow = RectangularShape(particle_spacing, (open_bounary_cols, n_particles_out_y),
                           (start_outflow_zone, 0.0), fluid_density,
                           init_velocity=initial_velocity, buffer=n_buffer_particles_fluid,
                           pressure=pressure)

# First vector is in down stream direction
zone_points_in = ([end_inflow_zone - particle_spacing / 2;
                   step_height - particle_spacing / 2],
                  [end_inflow_zone - particle_spacing / 2;
                   height_open_boundary_out - particle_spacing / 2])
zone_points_out = ([end_outflow_zone - particle_spacing / 2; 0.0],
                   [end_outflow_zone - particle_spacing / 2;
                    height_open_boundary_out - particle_spacing / 2])

# ==========================================================================================
# ==== Boundary

n_boundary_particles_x = n_particles_slip_x + 2 * open_bounary_cols
boundary_layers = 3

bottom_slip_wall = RectangularShape(particle_spacing,
                                    (n_boundary_particles_x, boundary_layers),
                                    (start_inflow_zone -
                                     open_bounary_cols * particle_spacing,
                                     step_height - boundary_layers * particle_spacing),
                                    fluid_density)

top_slip_wall = RectangularShape(particle_spacing,
                                 (n_boundary_particles_x, boundary_layers),
                                 (start_inflow_zone -
                                  open_bounary_cols * particle_spacing,
                                  step_height + inlet_width),
                                 fluid_density)

n_interior_wall_x = Int(floor(interior_length / particle_spacing)) + boundary_layers +
                    2 * open_bounary_cols

bottom_interior_wall = RectangularShape(particle_spacing,
                                        (n_interior_wall_x, boundary_layers),
                                        (inlet_length, -particle_spacing * boundary_layers),
                                        fluid_density)

top_interior_wall = RectangularShape(particle_spacing,
                                     (n_interior_wall_x, boundary_layers),
                                     (inlet_length, inlet_width + step_height),
                                     fluid_density)

bottom_inlet_wall = RectangularShape(particle_spacing,
                                     (n_particles_inlet_x, boundary_layers),
                                     (0.0,
                                      step_height - particle_spacing * boundary_layers),
                                     fluid_density)

top_inlet_wall = RectangularShape(particle_spacing,
                                  (n_particles_inlet_x, boundary_layers),
                                  (0.0, inlet_width + step_height),
                                  fluid_density)

vertical_wall = RectangularShape(particle_spacing,
                                 (boundary_layers,
                                  Int(floor(step_height / particle_spacing))),
                                 (inlet_length - particle_spacing * boundary_layers,
                                  -particle_spacing * boundary_layers),
                                 fluid_density)

no_slip_boundary = InitialCondition(bottom_interior_wall, top_interior_wall,
                                    bottom_inlet_wall,
                                    top_inlet_wall, vertical_wall)
slip_boundary = InitialCondition(bottom_slip_wall, top_slip_wall)

wall_viscosity = ViscosityAdami(1.0e-4)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(no_slip_boundary.density,
                                             no_slip_boundary.mass, state_equation,
                                             AdamiPressureExtrapolation(),
                                             viscosity=wall_viscosity,
                                             smoothing_kernel, smoothing_length)
boundary_slip_model = BoundaryModelDummyParticles(slip_boundary.density,
                                                  slip_boundary.mass, state_equation,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Containers

#fluid_system = WeaklyCompressibleSPHSystem(fluid, SummationDensity(), state_equation,
#                                           smoothing_kernel, smoothing_length,
#                                           #viscosity=viscosity,
#                                           acceleration=(0.0, gravity))
fluid_system = EntropicallyDampedSPH(fluid, smoothing_kernel, smoothing_length,
                                     sound_speed, viscosity=viscosity, #wall_viscosity,
                                     acceleration=(0.0, gravity))

open_boundary_in = OpenBoundarySPHSystem(inflow, InFlow(), sound_speed, zone_points_in,
                                         zone_origin_in, fluid_system)

open_boundary_out = OpenBoundarySPHSystem(outflow, OutFlow(), sound_speed, zone_points_out,
                                          zone_origin_out, fluid_system)

boundary_system = BoundarySPHSystem(no_slip_boundary.coordinates, boundary_model)

boundary_slip_system = BoundarySPHSystem(slip_boundary.coordinates, boundary_slip_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          boundary_system, boundary_slip_system,
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
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
