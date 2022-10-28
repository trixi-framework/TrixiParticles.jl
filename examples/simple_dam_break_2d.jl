using Pixie
using OrdinaryDiffEq

# Particle data
n_particles_per_dimension = (10, 30)
spacing = 0.1
particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 1000 * spacing^2 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = (x - 1 - 0.5 * (n_particles_per_dimension[1] - 1)) * spacing
    particle_coordinates[2, particle] = y * spacing + 10 * spacing

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
end

# Boundary particle data
n_boundaries_per_dimension = (400,)
beta = 3

boundary_coordinates = Array{Float64, 2}(undef, 2, prod(n_boundaries_per_dimension))
boundary_masses = 10 * ones(Float64, prod(n_boundaries_per_dimension))

for y in 1:n_boundaries_per_dimension[1]
    boundary_particle = y

    boundary_coordinates[1, boundary_particle] = spacing / beta * (y - 1 - 0.5 * (n_boundaries_per_dimension[1] - 1))
    boundary_coordinates[2, boundary_particle] = -spacing
end

smoothing_length = 0.12
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

K = 1000.0
boundary_conditions = BoundaryParticlesMonaghanKajtar(boundary_coordinates, boundary_masses,
                                                      K, beta, spacing / beta,
                                                      neighborhood_search=SpatialHashingSearch{2}(search_radius))

# Create semidiscretization
state_equation = StateEquationCole(100.0, 7, 1000.0, 1.0, background_pressure=1.0)
# state_equation = StateEquationIdealGas(10.0, 3.0, 10.0, background_pressure=10.0)

semi = SPHSemidiscretization{2}(particle_masses,
                                ContinuityDensity(), state_equation,
                                smoothing_kernel, smoothing_length,
                                viscosity=ArtificialViscosityMonaghan(0.5, 1.0),
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81),
                                neighborhood_search=SpatialHashingSearch{2}(search_radius))
                            #   neighborhood_search=nothing)

tspan = (0.0, 5.0)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = AliveCallback(alive_interval=100)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0)

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            # abstol=1.0e-6, reltol=1.0e-6, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks);
