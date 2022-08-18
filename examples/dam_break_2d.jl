using Pixie
using OrdinaryDiffEq

width = 2.0
water_height = 0.9
container_height = 1.0
particle_spacing = 0.01
boundary_distance = 2
boundary_offset = boundary_distance - 1

mass = 1000 * particle_spacing^2

# Particle data
n_particles_per_dimension = (Int(width / 2 / particle_spacing) - boundary_offset, Int(water_height / particle_spacing) - boundary_offset)
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = mass * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = (x + boundary_offset) * particle_spacing
    particle_coordinates[2, particle] = (y + boundary_offset) * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

# Boundary particle data
beta = 2

n_boundaries_vertical = Int(container_height / particle_spacing * beta) + 1
n_boundaries_horizontal = Int(width / particle_spacing * beta) - 1
n_boundaries = 2 * n_boundaries_vertical + n_boundaries_horizontal
boundary_coordinates = Array{Float64, 2}(undef, 2, n_boundaries)
boundary_masses = mass * ones(Float64, n_boundaries)

# Left boundary
for y in 1:n_boundaries_vertical
    boundary_particle = y

    boundary_coordinates[1, boundary_particle] = 0
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
end

# Right boundary
for y in 1:n_boundaries_vertical
    boundary_particle = n_boundaries_vertical + y

    boundary_coordinates[1, boundary_particle] = 2
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
end

# Bottom boundary
for y in 1:n_boundaries_horizontal
    boundary_particle = 2*n_boundaries_vertical + y

    boundary_coordinates[1, boundary_particle] = y * particle_spacing / beta
    boundary_coordinates[2, boundary_particle] = 0
end

c = 10 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

K = 9.81 * water_height
boundary_conditions = BoundaryConditionMonaghanKajtar(K, boundary_coordinates,
                                                      boundary_masses, beta,
                                                      neighborhood_search=SpatialHashingSearch{2}(search_radius))

# Create semidiscretization
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)
# state_equation = StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)

semi = SPHSemidiscretization{2}(particle_masses,
                                ContinuityDensity(), state_equation,
                                smoothing_kernel, smoothing_length,
                                viscosity=ArtificialViscosityMonaghan(1.0, 2.0),
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81),
                                neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 2.0)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-6, reltol=1.0e-6, # Tighter tolerance to prevent instabilities
            saveat=0.02, callback=alive_callback);
