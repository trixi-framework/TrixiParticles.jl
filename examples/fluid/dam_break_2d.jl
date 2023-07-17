# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using TrixiParticles
using OrdinaryDiffEq

# Constants
gravity = 9.81
athmospheric_pressure = 100000.0
fluid_density = 1000.0

# Simulation settings
particle_spacing = 0.02
smoothing_length = 1.2 * particle_spacing
boundary_layers = 3
output_dt = 0.02
relaxation_step_file_prefix = "relaxation"
simulation_step_file_prefix = ""
relaxation_tspan = (0.0, 0.5)
simulation_tspan = (0.0, 5.7 / sqrt(gravity))

# Model settings
fluid_density_calculator = ContinuityDensity()
boundary_density_calculator = AdamiPressureExtrapolation()

# Boundary geometry and initial fluid particle positions
initial_fluid_height = 2.0
initial_fluid_size = (initial_fluid_height, 1.0)
tank_size = (floor(5.366 / particle_spacing) * particle_spacing, 4.0)
tank = RectangularTank(particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers)

# move the right wall of the tank to a new position
function move_wall(tank, new_wall_position)
    reset_faces = (false, true, false, false)
    positions = (0, new_wall_position, 0, 0)
    reset_wall!(tank, reset_faces, positions)
end

move_wall(tank, tank.fluid_size[1])

# ==========================================================================================
# ==== Fluid
sound_speed = 20 * sqrt(gravity * initial_fluid_height)

state_equation = StateEquationCole(sound_speed, 7, fluid_density, athmospheric_pressure,
                                   background_pressure=athmospheric_pressure)

smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity), correction=nothing)

# ==========================================================================================
# ==== Boundary models
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation, boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

# K = 9.81 * initial_fluid_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              tank.boundary.mass)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

ode = semidiscretize(semi, relaxation_tspan)

info_callback = InfoCallback(interval=100)
saving_callback_relaxation = SolutionSavingCallback(dt=output_dt,
                                                    prefix=relaxation_step_file_prefix)
density_reinit_cb = DensityReinitializationCallback(semi.systems[1], dt=0.01)

callbacks_relaxation = CallbackSet(info_callback, saving_callback_relaxation, PostprocessCallback(), density_reinit_cb)

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
            save_everystep=false, callback=callbacks_relaxation);

# move_wall(tank, tank.tank_size[1])

# # Use solution of the relaxing step as initial coordinates
# restart_with!(semi, sol)

# semi = Semidiscretization(fluid_system, boundary_system,
#                           neighborhood_search=SpatialHashingSearch)
# ode = semidiscretize(semi, simulation_tspan)

# saving_callback = SolutionSavingCallback(dt=output_dt, prefix=simulation_step_file_prefix)
# density_reinit_cb = DensityReinitializationCallback(semi.systems[1], dt=0.01)
# callbacks = CallbackSet(info_callback, saving_callback, density_reinit_cb)

# # See above for an explanation of the parameter choice
# sol = solve(ode, RDPK3SpFSAL49(),
#             abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
#             reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
#             dtmax=1e-2, # Limit stepsize to prevent crashing
#             save_everystep=false, callback=callbacks);
