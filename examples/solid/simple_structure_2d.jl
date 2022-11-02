using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (10, 10)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * 0.1
    particle_coordinates[2, particle] = y * 0.1

    # Velocity
    particle_velocities[1, particle] = 1.0
    particle_velocities[2, particle] = 1.0
end

smoothing_length = 0.12
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# Lamé constants
E = 1.4e6
nu = 0.4
semi = SPHSolidSemidiscretization{2}(particle_masses, particle_densities, SummationDensity(),
                                     smoothing_kernel, smoothing_length,
                                     E, nu,
                                     neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 1.0)

ode = semidiscretize(semi, particle_coordinates, particle_velocities, tspan)

alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0)

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), save_everystep=false, callback=callbacks);
