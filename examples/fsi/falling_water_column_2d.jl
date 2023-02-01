using Pixie
using OrdinaryDiffEq

fluid_particle_spacing = 0.0125 * 3

water_width = 0.525
water_height = 1.0125
water_density = 1000.0

container_width = 4.0
container_height = 4.0

setup = RectangularShape(fluid_particle_spacing,
                         (round(Int, (water_width / fluid_particle_spacing)),
                          round(Int, (water_height / fluid_particle_spacing))),
                         (0.1, 0.2), density=water_density)

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Create semidiscretization
fluid_container = FluidParticleContainer(setup.coordinates,
                                         zeros(Float64, size(setup.coordinates)),
                                         setup.masses, setup.densities,
                                         ContinuityDensity(), state_equation,
                                         smoothing_kernel, smoothing_length,
                                         viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                         acceleration=(0.0, -9.81))

length_beam = 0.35
thickness = 0.05
n_particles_y = 5
clamp_radius = 0.05
solid_density = 1000.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_y - 1)

fixed_particles = CircularShape(clamp_radius + solid_particle_spacing / 2, 0.0,
                                thickness / 2, solid_particle_spacing,
                                shape_type=FillCircle(x_recess=(0.0, clamp_radius),
                                                      y_recess=(0.0, thickness)),
                                density=solid_density)

n_particles_clamp_x = round(Int, clamp_radius / solid_particle_spacing)

# cantilever and clamped particles
n_particles_per_dimension = (round(Int, length_beam / solid_particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

beam = RectangularShape(solid_particle_spacing, n_particles_per_dimension, (0, 0),
                        density=solid_density)

particle_coordinates = hcat(beam.coordinates, fixed_particles.coordinates)
particle_velocities = zeros(Float64, size(particle_coordinates))

particle_masses = vcat(beam.masses, fixed_particles.masses)
particle_densities = vcat(beam.densities, fixed_particles.densities)

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Lamé constants
E = 1.4e6
nu = 0.4

K = 9.81 * water_height
beta = fluid_particle_spacing / solid_particle_spacing
solid_container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                         particle_masses, particle_densities,
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=fixed_particles.n_particles,
                                         acceleration=(0.0, -9.81),
                                         BoundaryModelMonaghanKajtar(K, beta,
                                                                     solid_particle_spacing),
                                         penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

semi = Semidiscretization(fluid_container, solid_container,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.005:1000.0)

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
# pixie2vtk(saved_values)
