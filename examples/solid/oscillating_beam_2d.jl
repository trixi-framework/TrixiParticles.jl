using Pixie
using OrdinaryDiffEq

length_beam = 0.35
thickness = 0.02
n_particles_y = 5
clamp_radius = 0.05

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_y - 1)

fixed_particle_coords = fill_circle_with_recess(clamp_radius+particle_spacing/2, 0.0, thickness/2,
                                                [0,clamp_radius], [0, thickness], particle_spacing)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)
n_particles_fixed = size(fixed_particle_coords, 2)

# cantilever and clamped particles
n_particles_per_dimension = (round(Int, length_beam / particle_spacing) + n_particles_clamp_x + 1, n_particles_y)

particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension) + n_particles_fixed)
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension) + n_particles_fixed)

beam = RectangularShape(particle_spacing, n_particles_per_dimension[1], n_particles_per_dimension[2], 0.0, 0.0)

particle_coordinates = cat(beam.coordinates, fixed_particle_coords, dims=(2,2))
particle_velocities = zeros(Float64, size(particle_coordinates))

smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Lamé constants
E = 1.4e6
nu = 0.4

particle_container = SolidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                            SummationDensity(),
                                            smoothing_kernel, smoothing_length,
                                            E, nu,
                                            n_fixed_particles=n_particles_fixed,
                                            acceleration=(0.0, -2.0),
                                            nothing) # No boundary model

semi = Semidiscretization(particle_container, neighborhood_search=SpatialHashingSearch)
tspan = (0.0, 5.0)

ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
