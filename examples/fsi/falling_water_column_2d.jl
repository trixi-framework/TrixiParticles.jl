using Pixie
using OrdinaryDiffEq

water_width = 0.5
water_height = 1.0
container_width = 4.0
container_height = 2.0
particle_density = 1000.0
fluid_particle_spacing = 0.005 * 3
# Ratio of fluid particle spacing to boundary particle spacing
beta = 3

setup = RectangularTank(fluid_particle_spacing, beta, water_width, water_height,
                        container_width, container_height, particle_density)

# Move water column
for i in axes(setup.particle_coordinates, 2)
    setup.particle_coordinates[:, i] .+= [0.1, 0.2]
end

c = 10 * sqrt(9.81 * water_height)
state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Create semidiscretization
fluid_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                         setup.particle_masses, setup.particle_densities,
                                         ContinuityDensity(), state_equation,
                                         smoothing_kernel, smoothing_length,
                                         viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                         acceleration=(0.0, -9.81))

length = 0.35
thickness = 0.05
n_particles_y = 5
clamp_radius = 0.05

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_y - 1)

n_particles_clamp_x = round(Int, clamp_radius / solid_particle_spacing)
n_particles_fixed = 2 * n_particles_clamp_x + n_particles_y + 2

n_particles_per_dimension = (round(Int, length / solid_particle_spacing) + 1 + (n_particles_clamp_x - 1), n_particles_y)

particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension) + n_particles_fixed)
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension) + n_particles_fixed)
particle_masses = 1000 * solid_particle_spacing^2 * ones(Float64, prod(n_particles_per_dimension) + n_particles_fixed)
particle_densities = 1000 * ones(Float64, prod(n_particles_per_dimension) + n_particles_fixed)

for y in 1:n_particles_per_dimension[2],
        x in 1:n_particles_per_dimension[1]
    particle = (x - 1) * n_particles_per_dimension[2] + y

    # Coordinates
    particle_coordinates[1, particle] = x * solid_particle_spacing
    particle_coordinates[2, particle] = y * solid_particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particle above the beam
for x in 1:n_particles_clamp_x
    particle = prod(n_particles_per_dimension) + x

    # Coordinates
    particle_coordinates[1, particle] = x * solid_particle_spacing
    particle_coordinates[2, particle] = (n_particles_y + 1) * solid_particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

# Fixed particles below the beam
for x in 1:n_particles_clamp_x
    particle = prod(n_particles_per_dimension) + n_particles_clamp_x + x

    # Coordinates
    particle_coordinates[1, particle] = x * solid_particle_spacing
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
    particle_coordinates[2, particle] = (x - 1) * solid_particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Lamé constants
E = 1.4e6
nu = 0.4

K = 9.81 * water_height
beta = fluid_particle_spacing / solid_particle_spacing
solid_container = SolidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                         SummationDensity(),
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=n_particles_fixed,
                                         acceleration=(0.0, -9.81),
                                         BoundaryModelMonaghanKajtar(K, beta, solid_particle_spacing),
                                         penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

semi = Semidiscretization(fluid_container, solid_container, neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=10)
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1000.0)

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-5, # Small initial stepsize because the automatic choice is usually too large
            abstol=1e-5, # Higher abstol (default is 1e-6) for performance reasons
            reltol=1e-4, # Smaller reltol (default is 1e-3) to prevent boundary penetration
            save_everystep=false, callback=callbacks);
