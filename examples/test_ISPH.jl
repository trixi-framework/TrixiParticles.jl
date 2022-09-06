using Pixie
using OrdinaryDiffEq
setup = ["test_ISPH"]
# Particle data
n_particles_per_dimension = (20, 60)
spacing = 0.01 #smoothing_length/1.25
smoothing_length = 1.2 * spacing
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
boundary_masses = particle_masses[1] * ones(Float64, prod(n_boundaries_per_dimension))

for y in 1:n_boundaries_per_dimension[1]
    boundary_particle = y

    boundary_coordinates[1, boundary_particle] = spacing / beta * (y - 1 - 0.5 * (n_boundaries_per_dimension[1] - 1))
    boundary_coordinates[2, boundary_particle] = -spacing
end


smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

K = 10.0
c = 100.0
#boundary_conditions = BoundaryConditionCrespo(boundary_coordinates, boundary_masses, c)
boundary_conditions = BoundaryConditionMonaghanKajtar(boundary_coordinates, boundary_masses,
                                                      K, beta, spacing / beta,
                                                      neighborhood_search=SpatialHashingSearch{2}(search_radius))
#boundary_conditions = BoundaryConditionFixedParticleLiu(boundary_coordinates, boundary_masses, particle_densities[1])

# Create semidiscretization
pressure_poisson_eq = PPEExplicitLiu(0.1*smoothing_length)
semi = EISPHSemidiscretization{2}(particle_masses,
                                SummationDensity(), pressure_poisson_eq,
                                smoothing_kernel, smoothing_length,
                                viscosity=ViscosityClearyMonaghan(1e-6), #ViscosityClearyMonaghan(1e-6) # ArtificialViscosityMonaghan(100.0, 0.02, 0.0)
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81),
                                neighborhood_search=SpatialHashingSearch{2}(search_radius))

tspan = (0.0, 0.5)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)
alive_callback = AliveCallback(alive_interval=100)
dt_callback    = StepSizeCallback(callback_interval=100)
saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0)

callbacks = CallbackSet(alive_callback, saving_callback.callback, dt_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, 
            RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()), #Euler(), #
            dt=1e-4,
            saveat=0.02, 
            callback=callbacks);
