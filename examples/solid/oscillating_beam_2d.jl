using Pixie
using OrdinaryDiffEq

acceleration = -2.0

# ==========================================================================================
# ==== Solid

length_beam = 0.35
thickness = 0.02
n_particles_y = 5
clamp_radius = 0.05
particle_density = 1000.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_y - 1)

smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Lamé constants
E = 1.4e6
nu = 0.4

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius.
fixed_particles = CircularShape(clamp_radius + particle_spacing / 2, 0.0, thickness / 2,
                                particle_spacing,
                                shape_type=FillCircle(x_recess=(0.0, clamp_radius),
                                                      y_recess=(0.0, thickness)),
                                density=particle_density)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

# cantilever and clamped particles
n_particles_per_dimension = (round(Int, length_beam / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

beam = RectangularShape(particle_spacing, n_particles_per_dimension, (0, 0),
                        density=particle_density)

particle_coordinates = hcat(beam.coordinates, fixed_particles.coordinates)
particle_velocities = zeros(Float64, size(particle_coordinates))
particle_masses = vcat(beam.masses, fixed_particles.masses)
particle_densities = vcat(beam.densities, fixed_particles.densities)

# ==========================================================================================
# ==== Containers

particle_container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                            particle_masses, particle_densities,
                                            smoothing_kernel, smoothing_length,
                                            E, nu,
                                            n_fixed_particles=fixed_particles.n_particles,
                                            acceleration=(0.0, acceleration),
                                            nothing) # No boundary model

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(particle_container, neighborhood_search=SpatialHashingSearch)
tspan = (0.0, 5.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
# pixie2vtk(saved_values)
