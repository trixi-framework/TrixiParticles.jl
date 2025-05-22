# ==========================================================================================
# 2D Channel Flow Through a Periodic Array of Cylinders
#
# Based on:
#   S. Adami, X. Y. Hu, N. A. Adams.
#   "A transport-velocity formulation for smoothed particle hydrodynamics".
#   Journal of Computational Physics, Volume 241 (2013), pages 292-307.
#   https://doi.org/10.1016/J.JCP.2013.01.043
#
# This example simulates fluid flow driven by a body force through a channel
# containing a periodic array of cylinders. Due to periodicity, only one cylinder
# within a representative channel segment is simulated.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

cylinder_radius = 0.02

tank_width = 6 * cylinder_radius
tank_height = 4 * cylinder_radius
tank_size = (tank_width, tank_height)

# Number of particles along the width of the tank, determining the particle spacing.
num_particles_x_domain = 144
particle_spacing = tank_width / num_particles_x_domain

# Number of boundary particle layers for the solid walls and cylinder.
boundary_layers = 3

# Fluid and Flow Properties
fluid_density = 1000.0

# Constant body acceleration in the x-direction driving the flow.
body_acceleration_x = 2.5e-4
body_acceleration_vec = (body_acceleration_x, 0.0)

eta = 0.1
nu = eta / fluid_density

# Adami uses `c = 0.1 * sqrt(acceleration_x * cylinder_radius)``  but the original setup
# from M. Ellero and N. A. Adams (https://doi.org/10.1002/nme.3088) uses `c = 0.02`
sound_speed = 0.02

# Characteristic pressure for TransportVelocityAdami formulation (rho_0 * c_0^2).
characteristic_pressure = fluid_density * sound_speed^2

# Simulation time span
tspan = (0.0, 5.0)

# ------------------------------------------------------------------------------
# Experiment Setup:
# ------------------------------------------------------------------------------

# Create a rectangular domain (box) with fluid particles.
# Top and bottom faces are solid walls; left and right are periodic.
# `pressure` argument is used by `TransportVelocityAdami`.
fluid_domain_initial_size = tank_size # Fluid initially fills the entire tank
box_particles = RectangularTank(particle_spacing, fluid_domain_initial_size, tank_size,
                                fluid_density,
                                n_layers=boundary_layers,
                                pressure=characteristic_pressure,
                                faces=(false, false, true, true)) # (left, right, bottom, top)

# Create a circular cylinder obstacle in the center of the domain.
cylinder_center_position = (tank_width / 2, tank_height / 2)
cylinder_shape = SphereShape(particle_spacing, cylinder_radius, cylinder_center_position,
                             fluid_density,
                             sphere_type=RoundSphere())

# Define the actual fluid particles by removing cylinder particles from the box fluid.
fluid_particles = setdiff(box_particles.fluid, cylinder_shape)

# Define boundary particles: combination of cylinder surface and box walls.
boundary_particles = union(cylinder_shape, box_particles.boundary)

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergQuarticSplineKernel{2}()

viscosity_model_adami = ViscosityAdami(nu=nu)
transport_velocity_adami = TransportVelocityAdami(characteristic_pressure)

fluid_system = EntropicallyDampedSPHSystem(fluid_particles, smoothing_kernel,
                                           smoothing_length, sound_speed,
                                           viscosity=viscosity_model_adami,
                                           transport_velocity=transport_velocity_adami,
                                           acceleration=body_acceleration_vec,
                                           reference_particle_spacing=particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(boundary_particles.density,
                                             boundary_particles.mass,
                                             boundary_density_calculator,
                                             smoothing_kernel,
                                             smoothing_length,
                                             viscosity=viscosity_model_adami,
                                             reference_particle_spacing=particle_spacing)

boundary_system = BoundarySPHSystem(boundary_particles, boundary_model)

# ------------------------------------------------------------------------------
# Simulation Setup:
# ------------------------------------------------------------------------------

# Periodic boundary conditions in x-direction for the neighborhood search.
# The y-direction is bounded by solid walls and is not periodic.
# The periodic box needs to cover the domain plus an extended region for boundary layers if any.
# Here, `min_corner` and `max_corner` define the fundamental periodic domain.
# Particles "exiting" at x=tank_width will reappear at x=0 for interaction purposes.
periodic_box_min_corner = SVector(0.0, -2.0 * tank_height)
periodic_box_max_corner = SVector(tank_width, 2.0 * tank_height)

neighborhood_search = GridNeighborhoodSearch{2}(periodic_box=PeriodicBox(min_corner=periodic_box_min_corner,
                                                                         max_corner=periodic_box_max_corner))

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# `UpdateCallback` is required for the transport velocity model to work.
update_callback = UpdateCallback()

callbacks = CallbackSet(info_callback, saving_callback, update_callback)

# ------------------------------------------------------------------------------
# Simulation:
# ------------------------------------------------------------------------------

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
