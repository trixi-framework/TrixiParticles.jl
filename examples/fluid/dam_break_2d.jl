# 2D dam break simulation based on
#
# S. Marrone, M. Antuono, A. Colagrossi, G. Colicchio, D. le Touzé, G. Graziani.
# "δ-SPH model for simulating violent impact flows".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 200, Issues 13–16 (2011), pages 1526–1542.
# https://doi.org/10.1016/J.CMA.2010.12.016

include("../Pixie_ODE_solve.jl")

particle_spacing = 0.02

# Spacing ratio between fluid and boundary particles
beta = 3

water_width = 2.0
water_height = 1.0
water_density = 1000.0

container_width = floor(5.366 / particle_spacing * beta) * particle_spacing / beta
container_height = 4

setup = RectangularTank(particle_spacing, beta, water_width, water_height,
                        container_width, container_height, water_density)

# Move right boundary
# Recompute the new water column width since the width has been rounded in `RectangularTank`.
new_wall_position = (setup.n_particles_per_dimension[1] + 1) * particle_spacing
reset_faces = (false, true, false, false)
positions = (0, new_wall_position, 0, 0)

reset_wall!(setup, reset_faces, positions)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(c, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

particle_container = FluidParticleContainer(setup, ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02,
                                                                                  0.0),
                                            acceleration=(0.0, -9.81))

K = 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates,
                                               setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta,
                                                                           particle_spacing /
                                                                           beta))

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = pixie_ode_solve(ode, RDPK3SpFSAL49(), tspan,
                      abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
                      reltol=1e-3, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
                      dtmax=1e-2, # Limit stepsize to prevent crashing
                      save=false);

# Move right boundary
positions = (0, container_width, 0, 0)
reset_wall!(setup, reset_faces, positions)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

# See above for an explanation of the parameter choice
sol = pixie_ode_solve(ode, RDPK3SpFSAL49(), tspan,
                      abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
                      reltol=1e-5, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
                      dtmax=1e-2, # Limit stepsize to prevent crashing
                      save=false);
