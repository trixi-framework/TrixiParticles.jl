# Lid-driven cavity
#
# S. Adami et al
# "A transport-velocity formulation for smoothed particle hydrodynamics".
# In: Journal of Computational Physics, Volume 241 (2013), pages 292-307.
# https://doi.org/10.1016/j.jcp.2013.01.043

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.02

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 5.0)
reynolds_number = 100.0

cavity_size = (1.0, 1.0)

fluid_density = 1.0

const VELOCITY_LID = 1.0
sound_speed = 10 * VELOCITY_LID

pressure = sound_speed^2 * fluid_density

viscosity = ViscosityAdami(; nu=VELOCITY_LID / reynolds_number)

cavity = RectangularTank(particle_spacing, cavity_size, cavity_size, fluid_density,
                         n_layers=boundary_layers,
                         faces=(true, true, true, false), pressure=pressure)

lid_position = 0.0 - particle_spacing * boundary_layers
lid_length = cavity.n_particles_per_dimension[1] + 2boundary_layers

lid = RectangularShape(particle_spacing, (lid_length, 3),
                       (lid_position, cavity_size[2]), density=fluid_density)

# ==========================================================================================
# ==== Fluid

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

fluid_system = EntropicallyDampedSPHSystem(cavity.fluid, smoothing_kernel, smoothing_length,
                                           density_calculator=ContinuityDensity(),
                                           sound_speed, viscosity=viscosity,
                                           transport_velocity=TransportVelocityAdami(pressure))

# ==========================================================================================
# ==== Boundary

lid_movement_function(t) = SVector(VELOCITY_LID * t, 0.0)

is_moving(t) = true

lid_movement = BoundaryMovement(lid_movement_function, is_moving)

boundary_model_cavity = BoundaryModelDummyParticles(cavity.boundary.density,
                                                    cavity.boundary.mass,
                                                    AdamiPressureExtrapolation(),
                                                    viscosity=viscosity,
                                                    smoothing_kernel, smoothing_length)

boundary_model_lid = BoundaryModelDummyParticles(lid.density, lid.mass,
                                                 AdamiPressureExtrapolation(),
                                                 viscosity=viscosity,
                                                 smoothing_kernel, smoothing_length)

boundary_system_cavity = BoundarySPHSystem(cavity.boundary, boundary_model_cavity)

boundary_system_lid = BoundarySPHSystem(lid, boundary_model_lid, movement=lid_movement)

# ==========================================================================================
# ==== Simulation
bnd_thickness = boundary_layers * particle_spacing
periodic_box = PeriodicBox(min_corner=[-bnd_thickness, -bnd_thickness],
                           max_corner=cavity_size .+ [bnd_thickness, bnd_thickness])

semi = Semidiscretization(fluid_system, boundary_system_cavity, boundary_system_lid,
                          neighborhood_search=GridNeighborhoodSearch{2}(; periodic_box))

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)

saving_callback = SolutionSavingCallback(dt=0.02)

pp_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, pp_callback, UpdateCallback())

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            maxiters=Int(1e7),
            save_everystep=false, callback=callbacks);
