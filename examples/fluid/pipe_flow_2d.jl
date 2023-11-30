# TODO: Description
using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
domain_length_factor = 0.01

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

open_boundary_cols = 5

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_length = 1.0
domain_width = 0.4
reynolds_number = 100

particle_spacing = domain_length_factor * domain_length

fluid_density = 1000.0
pressure = 1000.0

prescribed_velocity = (1.0, 0.0)

sound_speed = 10 * maximum(prescribed_velocity)

pipe = RectangularTank(particle_spacing, (domain_length, domain_width),
                       (domain_length + particle_spacing * open_boundary_cols,
                        domain_width), fluid_density, pressure=pressure,
                       n_layers=3, spacing_ratio=1, faces=(false, false, true, true),
                       init_velocity=prescribed_velocity)

n_particles_y = Int(floor(domain_width / particle_spacing))
n_buffer_particles = 4 * n_particles_y

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

nu = maximum(prescribed_velocity) * domain_length / reynolds_number
viscosity = ViscosityAdami(; nu) #alpha * smoothing_length * sound_speed / 8)

fluid_system = EntropicallyDampedSPHSystem(pipe.fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           transport_velocity=TransportVelocityAdami(pressure),
                                           buffer=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
open_boundary_length = particle_spacing * open_boundary_cols
open_boundary_size = (open_boundary_length, domain_width)

zone_origin_in = (-open_boundary_length, 0.0)
zone_origin_out = (domain_length, 0.0)

inflow = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density; n_layers=boundary_layers,
                         init_velocity=prescribed_velocity, pressure=pressure,
                         min_coordinates=zone_origin_in, spacing_ratio=spacing_ratio,
                         faces=(false, false, true, true))
outflow = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                          fluid_density; n_layers=boundary_layers,
                          init_velocity=prescribed_velocity, pressure=pressure,
                          min_coordinates=zone_origin_out, spacing_ratio=spacing_ratio,
                          faces=(false, false, true, true))
zone_plane_in = ([0.0; 0.0], [0.0; domain_width])
zone_plane_out = ([domain_length + open_boundary_length; 0.0],
                  [domain_length + open_boundary_length; domain_width])

open_boundary_in = OpenBoundarySPHSystem(inflow.fluid, InFlow(), sound_speed,
                                         zone_plane_in, buffer=n_buffer_particles,
                                         zone_origin_in, fluid_system)

open_boundary_out = OpenBoundarySPHSystem(outflow.fluid, OutFlow(), sound_speed,
                                          zone_plane_out, buffer=n_buffer_particles,
                                          zone_origin_out, fluid_system)

# ==========================================================================================
# ==== Boundary
boundary = union(pipe.boundary, inflow.boundary, outflow.boundary)

boundary_model = BoundaryModelDummyParticles(boundary.density, boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             viscosity=ViscosityAdami(nu=1e-4),
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback, UpdateEachTimeStep())

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
