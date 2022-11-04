using Pixie
using OrdinaryDiffEq

n_particles_per_dimension = (99, 5)
particle_spacing = 0.02 / 5

particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension) + 29)
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension) + 29)
particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension) + 29)
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension) + 29)

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * particle_spacing
    particle_coordinates[2, particle] = y * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particle above the beam
for x in 1:12
    particle = prod(n_particles_per_dimension) + x

    # Coordinates
    particle_coordinates[1, particle] = (x - 1) * particle_spacing
    particle_coordinates[2, particle] = 6 * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particles below the beam
for x in 1:12
    particle = prod(n_particles_per_dimension) + 12 + x

    # Coordinates
    particle_coordinates[1, particle] = (x - 1) * particle_spacing
    particle_coordinates[2, particle] = 0.0

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particles to the left of the beam
for x in 1:5
    particle = prod(n_particles_per_dimension) + 24 + x

    # Coordinates
    particle_coordinates[1, particle] = 0.0
    particle_coordinates[2, particle] = x * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end


smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# Lamé constants
E = 1.4e6
nu = 0.4
semi = SPHSolidSemidiscretization{2}(particle_masses, particle_densities, SummationDensity(),
                                     smoothing_kernel, smoothing_length,
                                     E, nu,
                                     gravity=(0.0, -2.0),
                                     neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 5.0)

# Make the last 29 particles fixed (24 for the support above and below the beam, 5 for
# the first layer of particles to the left)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, tspan, n_fixed_particles=29)

alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0,
                                                       index=(u, t, integrator) -> Pixie.eachparticle(integrator.p))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), save_everystep=false, callback=callbacks);
