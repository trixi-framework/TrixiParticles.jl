using Pixie
using OrdinaryDiffEq
setup = ["tank_2D",                   
        "BC_crespo"]  
width = 2.0
water_height = 0.9
container_height = 1.0
particle_spacing = 0.02

mass = 1000 * particle_spacing^2

# Particle data
n_particles_per_dimension = (Int(width / particle_spacing),
                             Int(water_height / particle_spacing))
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = mass * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * particle_spacing
    particle_coordinates[2, particle] = y * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

# Boundary particle data
beta = 3

n_boundaries_vertical = Int(container_height / particle_spacing * beta) + 1
n_boundaries_horizontal = Int((width + particle_spacing) / particle_spacing * beta) - 1
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

    boundary_coordinates[1, boundary_particle] = width + particle_spacing
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
end

# Bottom boundary
for x in 1:n_boundaries_horizontal
    boundary_particle = 2*n_boundaries_vertical + x

    boundary_coordinates[1, boundary_particle] = x * particle_spacing / beta
    boundary_coordinates[2, boundary_particle] = 0
end

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

K = 9.81 * water_height
boundary_conditions = Pixie.BoundaryConditionCrespo(boundary_coordinates, boundary_masses, c, neighborhood_search=SpatialHashingSearch{2}(search_radius))

# Create semidiscretization
semi = SPHSemidiscretization{2}(particle_masses,
                                ContinuityDensity(), state_equation,
                                smoothing_kernel, smoothing_length,
                                viscosity=ArtificialViscosityMonaghan(1.0, 2.0),
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81),
                                neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 1.0)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = AliveCallback(alive_interval=10)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), dt=1e-5, callback=alive_callback);
#sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), dt=1e-5, saveat=0.02, callback=alive_callback);
