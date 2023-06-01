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
pressure_in = 0.0
pressure_out = 0.0

flow_direction = 1 # 1: x, 2: y
prescribed_velocity = (0.1, 0.0)

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
                         density=water_density, init_velocity=prescribed_velocity)

buffer_fluid = BufferParticles(size(fluid.coordinates, 2), n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
open_bounary_cols = 4
start_inflow_zone = -particle_spacing * open_bounary_cols
end_inflow_zone = 0.0 - particle_spacing / 2
start_outflow_zone = n_particles_x * particle_spacing
end_outflow_zone = start_outflow_zone + particle_spacing * open_bounary_cols

inflow = RectangularShape(particle_spacing, (open_bounary_cols, n_particles_y),
                          (start_inflow_zone, 0.0),
                          density=water_density, init_velocity=prescribed_velocity)
outflow = RectangularShape(particle_spacing, (open_bounary_cols, n_particles_y),
                           (start_outflow_zone, 0.0),
                           density=water_density, init_velocity=prescribed_velocity)

range_in = (start_inflow_zone - 0.5 * particle_spacing, end_inflow_zone)
range_out = (start_outflow_zone - 0.5 * particle_spacing, end_outflow_zone)

boundary_zone_in = InFlow(range_in, dim=flow_direction)
boundary_zone_out = OutFlow(range_out, dim=flow_direction)

buffer_inflow = BufferParticles(size(inflow.coordinates, 2), n_buffer_particles)
buffer_outflow = BufferParticles(size(outflow.coordinates, 2), n_buffer_particles)

# ==========================================================================================
# ==== Boundary
n_boundary_particles_x = n_particles_x + 2 * (open_bounary_cols + 1) + 10
boundary_layers = 3

bottom_wall = RectangularShape(particle_spacing, (n_boundary_particles_x, boundary_layers),
                               (start_inflow_zone - 4 * particle_spacing,
                                -boundary_layers * particle_spacing),
                               density=water_density)
top_wall = RectangularShape(particle_spacing, (n_boundary_particles_x, boundary_layers),
                            (start_inflow_zone - 4 * particle_spacing,
                             n_particles_y * particle_spacing),
                            density=water_density)
boundary_coordinates = hcat(bottom_wall.coordinates, top_wall.coordinates)
boundary_densities = vcat(bottom_wall.densities, top_wall.densities)
boundary_masses = vcat(bottom_wall.masses, top_wall.masses)

eta = 0.1
wall_viscosity = ViscousInteractionAdami(eta, boundary_coordinates)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(boundary_densities,
                                             boundary_masses, state_equation,
                                             AdamiPressureExtrapolation(),
                                             #viscosity=wall_viscosity,
                                             smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Containers

fluid_container = FluidParticleContainer(fluid, SummationDensity(), state_equation,
                                         smoothing_kernel, smoothing_length,
                                         viscosity=viscosity,
                                         buffer=buffer_fluid,
                                         acceleration=(0.0, gravity))

open_boundary_container_in = OpenBoundaryParticleContainer(inflow, pressure_in,
                                                           water_density,
                                                           prescribed_velocity[flow_direction],
                                                           boundary_zone_in,
                                                           buffer_inflow,
                                                           fluid_container)

open_boundary_container_out = OpenBoundaryParticleContainer(outflow, pressure_out,
                                                            water_density,
                                                            prescribed_velocity[flow_direction],
                                                            boundary_zone_out,
                                                            buffer_outflow,
                                                            fluid_container)

boundary_container = BoundaryParticleContainer(boundary_coordinates, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_container,
                          open_boundary_container_in,
                          open_boundary_container_out,
                          boundary_container,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(interval=10)

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
