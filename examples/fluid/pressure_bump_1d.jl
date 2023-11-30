# TODO: Description
using TrixiParticles
using OrdinaryDiffEq
using LinearAlgebra

# ==========================================================================================
# ==== Resolution
domain_length_factor = 0.05

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

open_boundary_cols = 6

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.2)

domain_length = 1.0
particle_spacing = domain_length_factor * domain_length

fluid_density = 1000.0
pressure = 1.0
prescribed_velocity = (1.0,)

sound_speed = max(10 * maximum(prescribed_velocity), 10.0)

# Pressure bump
p(x) = 0.0#1.0 - 0.2 * ℯ^-((norm(x) - 0.5)^2 / 0.001)

n_particles_x = Int(floor(domain_length / particle_spacing))
n_buffer_particles = 100

fluid = RectangularShape(particle_spacing, (n_particles_x,), (0.0,),
                         fluid_density, init_velocity=prescribed_velocity,
                         pressure=pressure)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{1}()

alpha = 0.5
viscosity = ViscosityAdami(nu=alpha * smoothing_length * sound_speed / 8)

fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           #viscosity=viscosity, alpha=alpha,
                                           initial_pressure_function=p, sound_speed,
                                           buffer=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
open_boundary_length = particle_spacing * open_boundary_cols

zone_origin_in = (-open_boundary_length,)
zone_origin_out = (domain_length,)

inflow = RectangularShape(particle_spacing, (open_boundary_cols,), (-open_boundary_length,),
                          fluid_density, init_velocity=prescribed_velocity,
                          pressure=pressure)

outflow = RectangularShape(particle_spacing, (open_boundary_cols,), (domain_length,),
                           fluid_density, init_velocity=prescribed_velocity,
                           pressure=pressure)

zone_plane_in = (0.0,)
zone_plane_out = (domain_length + open_boundary_length,)

open_boundary_in = OpenBoundarySPHSystem(inflow, InFlow(), sound_speed, zone_plane_in,
                                         zone_origin_in, fluid_system,
                                         buffer=n_buffer_particles)

open_boundary_out = OpenBoundarySPHSystem(outflow, OutFlow(), sound_speed, zone_plane_out,
                                          zone_origin_out, fluid_system,
                                          buffer=n_buffer_particles)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system,
                          open_boundary_in,
                          open_boundary_out,
                          neighborhood_search=GridNeighborhoodSearch)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback, UpdateEachTimeStep())

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
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
