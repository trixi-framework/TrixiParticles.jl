using Pixie
using OrdinaryDiffEq

particle_spacing = 0.02
beta = 3

water_width = 0.2
water_height = 0.4
container_width = 0.8
container_height = 1.0
wall_height = container_height

particle_density = 997.0

setup = RectangularTank(particle_spacing, beta, water_width, water_height,
                             container_width, container_height, particle_density,
                             n_layers=1)

setup_wall = VerticalWall(particle_spacing, beta, wall_height, water_width, particle_density)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

state_equation = StateEquationCole(c, 7, 997.0, 100000.0, background_pressure=100000.0)

particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81))

K = 4 * 9.81 * water_height
boundary_container_tank = BoundaryParticlesMonaghanKajtar(setup.boundary_coordinates, setup.boundary_masses,
                                                     K, beta, particle_spacing / beta)

boundary_container_wall = BoundaryParticlesMonaghanKajtar(setup_wall.boundary_coordinates, setup_wall.boundary_masses,
                                                     K, beta, particle_spacing / beta)

# boundary_container = BoundaryParticlesFrozen(setup.boundary_coordinates, setup.boundary_masses,
#                                              particle_density,
#                                              neighborhood_search=SpatialHashingSearch{2}(search_radius))


length = 0.09
thickness = 0.004
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_x - 1)

n_particles_per_dimension = (n_particles_x, round(Int, length / particle_spacing) + 1)

particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 1161.54 * particle_spacing^2 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1161.54 * ones(Float64, prod(n_particles_per_dimension))

for x in 1:n_particles_per_dimension[1],
        y in 1:n_particles_per_dimension[2]
    particle = (y - 1) * n_particles_per_dimension[1] + x

    # Coordinates
    particle_coordinates[1, particle] = 0.6 + (x - 1) * particle_spacing
    particle_coordinates[2, particle] = length - (y - 1) * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# Young's modulus and Poisson ratio
E = 3.5e6
nu = 0.49

solid_container = SolidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                         ContinuityDensity(),
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=n_particles_x,
                                         acceleration=(0.0, -9.81))


# Relaxing of the fluid without solid
semi = Semidiscretization(particle_container, boundary_container_tank, boundary_container_wall,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities, use 2e-5 for spacing 0.004
            save_everystep=false, callback=alive_callback);


# Run full simulation
tspan = (0.0, 1.0)

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, semi)
particle_container.initial_coordinates .= view(u_end, 1:2, :)
particle_container.initial_velocity .= view(u_end, 3:4, :)

semi = Semidiscretization(particle_container, boundary_container_tank, boundary_container_wall, solid_container, neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.005:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))
move_wall = MoveParticleCallback(callback_interval=100)

callbacks = CallbackSet(alive_callback, saving_callback, move_wall)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-4, # Initial guess of the time step to prevent too large guesses
            abstol=1.0e-4, reltol=1.0e-4, # Tighter tolerance to prevent instabilities
            save_everystep=false, callback=callbacks);
