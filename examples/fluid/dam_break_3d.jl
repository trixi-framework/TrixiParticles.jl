using Pixie
using OrdinaryDiffEq

width = 2.0
length = 1.0
water_height = 0.9
container_height = 1.0
particle_spacing = 0.05

mass = 1000 * particle_spacing^3

# Particle data
n_particles_per_dimension = (Int(width / 2 / particle_spacing),
                             Int(water_height / particle_spacing),
                             Int(length / particle_spacing))
particle_coordinates = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 3, prod(n_particles_per_dimension))
particle_masses = mass * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension))

for z in 1:n_particles_per_dimension[3],
        y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] * n_particles_per_dimension[3] +
        (y - 1) * n_particles_per_dimension[3] + z

    # Coordinates
    particle_coordinates[2, particle] = y * particle_spacing
    particle_coordinates[3, particle] = z * particle_spacing
    particle_coordinates[1, particle] = x * particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0
    particle_velocities[2, particle] = 0
    particle_velocities[3, particle] = 0
end

# Boundary particle data
beta = 3

n_boundaries_x = ceil(Int, (width + particle_spacing) / particle_spacing * beta) - 1
n_boundaries_y = Int(container_height / particle_spacing * beta) + 1
n_boundaries_z = Int((length + particle_spacing) / particle_spacing * beta) - 1
n_boundaries = n_boundaries_x * n_boundaries_z + 2 * n_boundaries_x * n_boundaries_y + 2 * (n_boundaries_z + 1) * n_boundaries_y
boundary_coordinates = Array{Float64, 2}(undef, 3, n_boundaries)
boundary_masses = mass * ones(Float64, n_boundaries)

boundary_particle = 0
# -x boundary
for z in 1:(n_boundaries_z + 1), y in 1:n_boundaries_y
    global boundary_particle += 1

    boundary_coordinates[1, boundary_particle] = 0
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
    boundary_coordinates[3, boundary_particle] = (z - 1) * particle_spacing / beta
end

# +x boundary
for z in 1:(n_boundaries_z + 1), y in 1:n_boundaries_y
    global boundary_particle += 1

    boundary_coordinates[1, boundary_particle] = width + particle_spacing
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
    boundary_coordinates[3, boundary_particle] = (z - 1) * particle_spacing / beta
end

# -z boundary
for y in 1:n_boundaries_y, x in 1:n_boundaries_x
    global boundary_particle += 1

    boundary_coordinates[1, boundary_particle] = x * particle_spacing / beta
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
    boundary_coordinates[3, boundary_particle] = 0
end

# -z boundary
for y in 1:n_boundaries_y, x in 1:n_boundaries_x
    global boundary_particle += 1

    boundary_coordinates[1, boundary_particle] = x * particle_spacing / beta
    boundary_coordinates[2, boundary_particle] = (y - 1) * particle_spacing / beta
    boundary_coordinates[3, boundary_particle] = length + particle_spacing
end

# Bottom boundary
for z in 1:n_boundaries_z, x in 1:n_boundaries_x
    global boundary_particle += 1

    boundary_coordinates[1, boundary_particle] = x * particle_spacing / beta
    boundary_coordinates[2, boundary_particle] = 0
    boundary_coordinates[3, boundary_particle] = z * particle_spacing / beta
end

c = 10 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{3}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# This factor of 2 should not be necessary.
# However, without it, the particles are pushed off the wall at the beginning
# of the simulation.
K = 9.81 * water_height / 2
boundary_conditions = BoundaryParticlesMonaghanKajtar(boundary_coordinates, boundary_masses,
                                                      K, beta, particle_spacing / beta,
                                                      neighborhood_search=SpatialHashingSearch{3}(search_radius))

# Create semidiscretization
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

semi = SPHFluidSemidiscretization{3}(particle_masses,
                                ContinuityDensity(), state_equation,
                                smoothing_kernel, smoothing_length,
                                viscosity=ArtificialViscosityMonaghan(0.1, 0.2),
                                boundary_conditions=boundary_conditions,
                                gravity=(0.0, -9.81, 0.0),
                                neighborhood_search=SpatialHashingSearch{3}(search_radius))

tspan = (0.0, 5.0)
ode = semidiscretize(semi, particle_coordinates, particle_velocities, particle_densities, tspan)

alive_callback = AliveCallback(alive_interval=10)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=alive_callback);

# Move right boundary
reset_right_wall!(setup, container_width)

# Run full simulation
tspan = (0.0, 5.7 / sqrt(9.81))

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, semi)
particle_container.initial_coordinates .= view(u_end, 1:3, :)
particle_container.initial_velocity .= view(u_end, 4:6, :)

semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
