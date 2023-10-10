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
atmospheric_pressure = 100000.0
fluid_density = 1000.0

# Simulation settings
fluid_particle_spacing = 0.02
smoothing_length = 3.0 * fluid_particle_spacing
boundary_layers = 4
# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
spacing_ratio = 1
boundary_particle_spacing = fluid_particle_spacing / spacing_ratio
output_dt = 0.02
relaxation_step_file_prefix = "relaxation"
simulation_step_file_prefix = ""
relaxation_tspan = (0.0, 3.0)
simulation_tspan = (0.0, 5.7 / sqrt(gravity))

# Model settings
fluid_density_calculator = ContinuityDensity()
boundary_density_calculator = AdamiPressureExtrapolation()
use_reinit = false

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 1.0)
tank_size = (floor(5.366 / boundary_particle_spacing) * boundary_particle_spacing, 4.0)
tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio)

# Move the right wall of the tank to a new position
function move_wall(tank, new_wall_position)
    reset_faces = (false, true, false, false)
    positions = (0, new_wall_position, 0, 0)
    reset_wall!(tank, reset_faces, positions)
end

# Move right wall to touch the fluid for the relaxing step
move_wall(tank, tank.fluid_size[1])

# ==========================================================================================
# ==== Fluid
sound_speed = 20 * sqrt(gravity * initial_fluid_size[2])

state_equation = StateEquationCole(sound_speed, 7, fluid_density, atmospheric_pressure,
                                   background_pressure=atmospheric_pressure)

smoothing_kernel = WendlandC2Kernel{2}()

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity), correction=nothing)

# ==========================================================================================
# ==== Boundary models
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

# K = gravity * initial_fluid_size[2]
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, fluid_particle_spacing / beta,
#                                              tank.boundary.mass)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch,
                          damping_coefficient=1e-4)

ode = semidiscretize(semi, relaxation_tspan)

info_callback = InfoCallback(interval=100)
saving_callback_relaxation = SolutionSavingCallback(dt=output_dt,
                                                    prefix=relaxation_step_file_prefix)
density_reinit_cb = use_reinit ? DensityReinitializationCallback(semi.systems[1], dt=0.01) :
                    nothing
callbacks_relaxation = CallbackSet(info_callback, saving_callback_relaxation,
                                   density_reinit_cb)

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

move_wall(tank, tank.tank_size[1])

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, simulation_tspan)

saving_callback = SolutionSavingCallback(dt=output_dt, prefix=simulation_step_file_prefix)
callbacks = CallbackSet(info_callback, saving_callback, density_reinit_cb)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
