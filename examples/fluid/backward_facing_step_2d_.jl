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

flow_direction = 1 # 1: x, 2: y
prescribed_velocity = (1.0, 0.0)
initial_velocity = (0.0, 0.0)

u_max = max(0.1, maximum(prescribed_velocity))
u_mean = u_max * 2 / 3
sound_speed = 10 * u_max

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, fluid_density, 1.0,
                                   background_pressure=1.0)

viscosity = ArtificialViscosityMonaghan(0.2, 0.0)

n_particles_interior_x = Int(floor(interior_length / particle_spacing))
n_particles_out_y = Int(floor(outlet_width / particle_spacing))

fluid_interior = RectangularShape(particle_spacing,
                                  (n_particles_interior_x, n_particles_out_y),
                                  (inlet_length, 0.0),
                                  density=fluid_density, init_velocity=initial_velocity)

n_particles_inlet_x = Int(floor(inlet_length / particle_spacing))
n_particles_in_y = Int(floor(inlet_width / particle_spacing))

fluid_inlet = RectangularShape(particle_spacing, (n_particles_inlet_x, n_particles_in_y),
                               (0.0, step_height),
                               density=fluid_density, init_velocity=initial_velocity)

n_particles_slip_x = Int(floor(slip_length / particle_spacing))

fluid_slip = RectangularShape(particle_spacing, (n_particles_slip_x, n_particles_in_y),
                              (-n_particles_slip_x * particle_spacing, step_height),
                              density=fluid_density, init_velocity=initial_velocity)

n_buffer_particles_fluid = 200 * n_particles_in_y

fluid = MergeShapes(fluid_slip, fluid_inlet, fluid_interior)

buffer_fluid = BufferParticles(size(fluid.coordinates, 2), n_buffer_particles_fluid)

# ==========================================================================================
# ==== Open Boundary

open_bounary_cols = 10
pressure_in = 0.0
pressure_out = 0.0

# extrude in upstream direction
end_inflow_zone = 0.0 - slip_length
start_inflow_zone = end_inflow_zone - particle_spacing * open_bounary_cols

# extrude in downstream direction
start_outflow_zone = inlet_length + interior_length
end_outflow_zone = start_outflow_zone + particle_spacing * open_bounary_cols

inflow = RectangularShape(particle_spacing, (open_bounary_cols, n_particles_in_y),
                          (start_inflow_zone, step_height),
                          density=fluid_density, init_velocity=prescribed_velocity)
outflow = RectangularShape(particle_spacing, (open_bounary_cols, n_particles_out_y),
                           (start_outflow_zone, 0.0),
                           density=fluid_density, init_velocity=initial_velocity)

range_in = (start_inflow_zone - 0.5 * particle_spacing,
            end_inflow_zone - 0.5 * particle_spacing)
range_out = (start_outflow_zone - 0.5 * particle_spacing, end_outflow_zone)

boundary_zone_in = InFlow(range_in, dim=flow_direction)
boundary_zone_out = OutFlow(range_out, dim=flow_direction)

n_buffer_particles_in = 10 * n_particles_in_y
n_buffer_particles_out = 10 * n_particles_out_y
buffer_inflow = BufferParticles(size(inflow.coordinates, 2), n_buffer_particles_in)
buffer_outflow = BufferParticles(size(outflow.coordinates, 2), n_buffer_particles_out)

# ==========================================================================================
# ==== Boundary

n_boundary_particles_x = n_particles_slip_x + 2 * open_bounary_cols
boundary_layers = 3

bottom_slip_wall = RectangularShape(particle_spacing,
                                    (n_boundary_particles_x, boundary_layers),
                                    (start_inflow_zone -
                                     open_bounary_cols * particle_spacing,
                                     step_height - boundary_layers * particle_spacing),
                                    density=fluid_density)

top_slip_wall = RectangularShape(particle_spacing,
                                 (n_boundary_particles_x, boundary_layers),
                                 (start_inflow_zone -
                                  open_bounary_cols * particle_spacing,
                                  step_height + inlet_width),
                                 density=fluid_density)

n_interior_wall_x = Int(floor(interior_length / particle_spacing)) + boundary_layers +
                    2 * open_bounary_cols

bottom_interior_wall = RectangularShape(particle_spacing,
                                        (n_interior_wall_x, boundary_layers),
                                        (inlet_length, -particle_spacing * boundary_layers),
                                        density=fluid_density)

top_interior_wall = RectangularShape(particle_spacing,
                                     (n_interior_wall_x, boundary_layers),
                                     (inlet_length, inlet_width + step_height),
                                     density=fluid_density)

bottom_inlet_wall = RectangularShape(particle_spacing,
                                     (n_particles_inlet_x, boundary_layers),
                                     (0.0,
                                      step_height - particle_spacing * boundary_layers),
                                     density=fluid_density)

top_inlet_wall = RectangularShape(particle_spacing,
                                  (n_particles_inlet_x, boundary_layers),
                                  (0.0, inlet_width + step_height),
                                  density=fluid_density)

vertical_wall = RectangularShape(particle_spacing,
                                 (boundary_layers,
                                  Int(floor(step_height / particle_spacing))),
                                 (inlet_length - particle_spacing * boundary_layers,
                                  -particle_spacing * boundary_layers),
                                 density=fluid_density)

no_slip_boundary = MergeShapes(bottom_interior_wall, top_interior_wall, bottom_inlet_wall,
                               top_inlet_wall, vertical_wall)
slip_boundary = MergeShapes(bottom_slip_wall, top_slip_wall)

eta = 0.1
wall_viscosity = ViscousInteractionAdami(eta, no_slip_boundary.coordinates)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(no_slip_boundary.densities,
                                             no_slip_boundary.masses, state_equation,
                                             AdamiPressureExtrapolation(),
                                             viscosity=wall_viscosity,
                                             smoothing_kernel, smoothing_length)
boundary_slip_model = BoundaryModelDummyParticles(slip_boundary.densities,
                                                  slip_boundary.masses, state_equation,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Containers

fluid_container = FluidParticleContainer(fluid, SummationDensity(), state_equation,
                                         smoothing_kernel, smoothing_length,
                                         viscosity=viscosity,
                                         buffer=buffer_fluid,
                                         acceleration=(0.0, gravity))

open_boundary_container_in = OpenBoundaryParticleContainer(inflow, pressure_in,
                                                           fluid_density,
                                                           prescribed_velocity[flow_direction],
                                                           boundary_zone_in,
                                                           buffer_inflow,
                                                           fluid_container)

open_boundary_container_out = OpenBoundaryParticleContainer(outflow, pressure_out,
                                                            fluid_density,
                                                            initial_velocity[flow_direction],
                                                            boundary_zone_out,
                                                            buffer_outflow,
                                                            fluid_container)

boundary_container = BoundaryParticleContainer(no_slip_boundary.coordinates, boundary_model)

boundary_slip_container = BoundaryParticleContainer(slip_boundary.coordinates,
                                                    boundary_slip_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_container,
                          open_boundary_container_in,
                          open_boundary_container_out,
                          boundary_container, boundary_slip_container,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 5.0)
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
sol = solve(ode, Euler(),
            dt=1e-3,
            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
