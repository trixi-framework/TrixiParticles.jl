# 2D dam break flow against an elastic plate based on Section 6.5 of
#
# L. Zhan, C. Peng, B. Zhang, W. Wu.
# "A stabilized TL–WC SPH approach with GPU acceleration for three-dimensional fluid–structure interaction".
# In: Journal of Fluids and Structures 86 (2019), pages 329-353.
# https://doi.org/10.1016/j.jfluidstructs.2019.02.002

using Pixie
using OrdinaryDiffEq

fluid_particle_spacing = 0.01
# Spacing ratio between fluid and boundary particles
beta = 3

water_width = 0.146
water_height = 0.292
water_density = 1000.0

container_width = 0.584
container_height = 4.0

setup = RectangularTank(fluid_particle_spacing, beta, water_width, water_height,
                        container_width, container_height, water_density)

# Move right boundary
# Recompute the new water column width since the width has been rounded in `RectangularTank`.
new_wall_position = (setup.n_particles_per_dimension[1] + 1) * particle_spacing
reset_face = (false, true, false, false)
position = (0, new_wall_position, 0, 0)

reset_wall!(setup, reset_face, position)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

particle_container = FluidParticleContainer(setup.particle_coordinates,
                                            setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02,
                                                                                  0.0),
                                            acceleration=(0.0, -9.81))

K = 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates,
                                               setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta,
                                                                           fluid_particle_spacing /
                                                                           beta))

length_beam = 0.08
thickness = 0.012
solid_density = 2500
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

n_particles_per_dimension = (n_particles_x,
                             round(Int, length_beam / solid_particle_spacing) + 1)

# The bottom layer is sampled separately below.
plate = RectangularShape(solid_particle_spacing,
                         (n_particles_per_dimension[1], n_particles_per_dimension[2] - 1),
                         (0.292, solid_particle_spacing),
                         density=solid_density)
fixed_particles = RectangularShape(solid_particle_spacing,
                                   (n_particles_per_dimension[1], 1),
                                   (0.292, 0.0), density=solid_density)

particle_coordinates = hcat(plate.coordinates, fixed_particles.coordinates)
particle_velocities = zeros(Float64, 2, prod(n_particles_per_dimension))
particle_masses = vcat(plate.masses, fixed_particles.masses)
particle_densities = vcat(plate.densities, fixed_particles.densities)

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Young's modulus and Poisson ratio
E = 1e6
nu = 0.0

beta = fluid_particle_spacing / solid_particle_spacing
solid_container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                         particle_masses, particle_densities,
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=n_particles_x,
                                         acceleration=(0.0, -9.81),
                                         # Use bigger K to prevent penetration into the solid
                                         BoundaryModelMonaghanKajtar(5K, beta,
                                                                     solid_particle_spacing),
                                         penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

# Relaxing of the fluid without solid
semi = Semidiscretization(particle_container, boundary_container,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan_relaxing = (0.0, 3.0)
ode = semidiscretize(semi, tspan_relaxing)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)

callbacks = CallbackSet(summary_callback, alive_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# Move right boundary
position = (0, container_width, 0, 0)
reset_wall!(setup, reset_face, position)

# Run full simulation
tspan = (0.0, 1.0)

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, particle_container, semi)
particle_container.initial_coordinates .= view(u_end, 1:2, :)
particle_container.initial_velocity .= view(u_end, 3:4, :)

semi = Semidiscretization(particle_container, boundary_container, solid_container,
                          neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.005:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
# pixie2vtk(saved_values)
