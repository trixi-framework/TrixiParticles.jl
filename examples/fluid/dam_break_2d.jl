# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

using Pixie
using OrdinaryDiffEq

particle_spacing = 0.02
# Ratio of fluid particle spacing to boundary particle spacing
beta = 3

water_width = 2.0
water_height = 1.0
water_density = 1000.0

container_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
container_height = 2.0

setup = RectangularTank(particle_spacing, beta, water_width, water_height,
                        container_width, container_height, water_density)

# Move right boundary
reset_right_wall!(setup, container_width, wall_position=water_width)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(c, 7, water_density, 100000.0, background_pressure=100000.0)

particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81))

K = 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta))

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=alive_callback);

# Move right boundary
reset_right_wall!(setup, container_width)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, semi)
particle_container.initial_coordinates .= view(u_end, 1:2, :)
particle_container.initial_velocity .= view(u_end, 3:4, :)

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-5, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
