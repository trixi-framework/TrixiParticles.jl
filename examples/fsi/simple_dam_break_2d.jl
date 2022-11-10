using Pixie
using OrdinaryDiffEq

# Particle data
n_particles_per_dimension = (10, 30)
spacing = 0.005
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 1000 * spacing^2 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * spacing + 0.3
    particle_coordinates[2, particle] = y * spacing + 10 * spacing

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

smoothing_length = 1.2 * spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# Create semidiscretization
state_equation = StateEquationCole(100.0, 7, 1000.0, 1.0, background_pressure=1.0)
# state_equation = StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)

particle_container = FluidParticleContainer(particle_coordinates, particle_velocities, particle_masses,
                                            SummationDensity(), state_equation, smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.5, 1.0),
                                            acceleration=(0.0, -9.81),
                                            neighborhood_search=SpatialHashingSearch{2}(search_radius))




# Beam
length = 0.35
thickness = 0.02
n_particles_y = 5
clamp_radius = 0.05

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_y - 1)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)
n_particles_fixed = 2 * n_particles_clamp_x + n_particles_y + 2

n_particles_per_dimension = (round(Int, length / particle_spacing) + 1 + (n_particles_clamp_x - 1), n_particles_y)

particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension) + n_particles_fixed)
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension) + n_particles_fixed)
particle_masses = 10 * ones(Float64, prod(n_particles_per_dimension) + n_particles_fixed)
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension) + n_particles_fixed)

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
for x in 1:n_particles_clamp_x
    particle = prod(n_particles_per_dimension) + x

    # Coordinates
    particle_coordinates[1, particle] = x * particle_spacing
    particle_coordinates[2, particle] = (n_particles_y + 1) * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particles below the beam
for x in 1:n_particles_clamp_x
    particle = prod(n_particles_per_dimension) + n_particles_clamp_x + x

    # Coordinates
    particle_coordinates[1, particle] = x * particle_spacing
    particle_coordinates[2, particle] = 0.0

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particles to the left of the beam
for x in 1:(n_particles_y + 2)
    particle = prod(n_particles_per_dimension) + 2 * n_particles_clamp_x + x

    # Coordinates
    particle_coordinates[1, particle] = 0.0
    particle_coordinates[2, particle] = (x - 1) * particle_spacing

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

solid_particle_container = SolidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                                  SummationDensity(),
                                                  smoothing_kernel, smoothing_length,
                                                  E, nu,
                                                  n_fixed_particles=n_particles_fixed,
                                                  acceleration=(0.0, -2.0))
                                                #   neighborhood_search=SpatialHashingSearch{2}(search_radius))



semi = Semidiscretization(particle_container, solid_particle_container)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0)

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-6, reltol=1.0e-6, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks);
