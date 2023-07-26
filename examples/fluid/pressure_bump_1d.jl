using TrixiParticles
using OrdinaryDiffEq
using LinearAlgebra

domain_length = 1.0
particle_spacing = 0.005 * domain_length

# ==========================================================================================
# ==== Fluid

water_density = 1000.0
pressure = 1.0
prescribed_velocity = (1.0, 0.0)

sound_speed = max(10 * maximum(prescribed_velocity), 10.0)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

n_particles_x = Int(floor(domain_length / particle_spacing))
n_particles_y = 1
pad_particles = floor(Int, n_particles_y / 2)
n_buffer_particles = 1 * n_particles_y

fluid_1 = RectangularShape(particle_spacing, (n_particles_x, pad_particles), (0.0, 0.0),
                           water_density, init_velocity=prescribed_velocity,
                           pressure=pressure)
fluid_2 = RectangularShape(particle_spacing, (n_particles_x, pad_particles),
                           (0.0, (pad_particles + 1) * particle_spacing),
                           water_density, init_velocity=prescribed_velocity,
                           pressure=pressure)
fluid_3 = RectangularShape(particle_spacing, (n_particles_x, 1),
                           (0.0, pad_particles * particle_spacing),
                           water_density, init_velocity=prescribed_velocity,
                           pressure=pressure)
fluid = InitialCondition(fluid_3, fluid_1, fluid_2, buffer=n_buffer_particles)

p(x) = 1.0 - 0.2 * ℯ^-((norm(x) - 0.5)^2 / 0.001)

# ==========================================================================================
# ==== Open Boundary

open_boundary_cols = 6

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
# ==== Systems
alpha = 0.5
viscosity = ViscosityAdami(alpha * smoothing_length * sound_speed / 8)

fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           #viscosity=viscosity, alpha=alpha,
                                           pressure_function=p,
                                           sound_speed)

open_boundary_in = OpenBoundarySPHSystem(inflow, InFlow(), sound_speed, zone_points_in,
                                         zone_origin_in, fluid_system)

open_boundary_out = OpenBoundarySPHSystem(outflow, OutFlow(), sound_speed, zone_points_out,
                                          zone_origin_out, fluid_system)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.001)

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

sol = solve(ode, #Euler(), dt=1e-4,
            RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3 # Limit stepsize to prevent crashing
            save_everystep = false, callback=callbacks);
