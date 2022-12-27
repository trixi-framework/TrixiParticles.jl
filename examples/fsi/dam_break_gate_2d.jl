# 2D dam break flow against an elastic plate based on
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033

using Pixie
using OrdinaryDiffEq

fluid_particle_spacing = 0.02
beta = 3

water_width = 0.2
water_height = 0.4
container_width = 0.8
container_height = 1.0
wall_height = container_height

particle_density = 997.0

setup = RectangularTank(fluid_particle_spacing, beta, water_width, water_height,
                        container_width, container_height, particle_density,
                        n_layers=1)

setup_wall = RectangularWall(fluid_particle_spacing, beta, wall_height, water_width, particle_density)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

state_equation = StateEquationCole(c, 7, 997.0, 100000.0, background_pressure=100000.0)

particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81))

# Add a factor of 4 to prevent boundary penetration
K = 4 * 9.81 * water_height

boundary_container_tank = BoundaryParticleContainer(setup.boundary_coordinates, setup.boundary_masses,
                                                    BoundaryModelMonaghanKajtar(K, beta, fluid_particle_spacing / beta))

# No moving boundaries for the relaxing step
movement_function(coordinates, t) = false
boundary_container_wall = BoundaryParticleContainer(setup_wall.coordinates, setup_wall.masses,
                                                    BoundaryModelMonaghanKajtar(K, beta, fluid_particle_spacing / beta),
                                                    movement_function=movement_function)


length = 0.09
thickness = 0.004
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

n_particles_per_dimension = (n_particles_x, round(Int, length / solid_particle_spacing) + 1)

particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 1161.54 * solid_particle_spacing^2 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 1161.54 * ones(Float64, prod(n_particles_per_dimension))

for x in 1:n_particles_per_dimension[1],
        y in 1:n_particles_per_dimension[2]
    particle = (y - 1) * n_particles_per_dimension[1] + x

    # Coordinates
    particle_coordinates[1, particle] = 0.6 + (x - 1) * solid_particle_spacing
    particle_coordinates[2, particle] = length - (y - 1) * solid_particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# Young's modulus and Poisson ratio
E = 3.5e6
nu = 0.45

beta = fluid_particle_spacing / solid_particle_spacing
solid_container = SolidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                         ContinuityDensity(),
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=n_particles_x,
                                         acceleration=(0.0, -9.81),
                                         BoundaryModelMonaghanKajtar(K, beta, solid_particle_spacing))


# Relaxing of the fluid without solid
semi = Semidiscretization(particle_container, boundary_container_tank, boundary_container_wall,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-5, # Small initial stepsize because the automatic choice is usually too large
            abstol=1e-5, # Higher abstol (default is 1e-6) for performance reasons
            reltol=1e-4, # Smaller reltol (default is 1e-3) to prevent boundary penetration
            save_everystep=false, callback=alive_callback);


# Run full simulation
tspan = (0.0, 1.0)

function movement_function(coordinates, t)

    if t < 0.1
        particle_spacing = coordinates[2,2] - coordinates[2,1]
        f(t) = -285.115t^3 + 72.305t^2 + 0.1463t + particle_spacing
        pos_1 = coordinates[2,1]
        pos_2 = f(t)
        diff_pos = pos_2 - pos_1
        coordinates[2,:] .+= diff_pos

        return true
    end

    return false
end

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, semi)
particle_container.initial_coordinates .= view(u_end, 1:2, :)
particle_container.initial_velocity .= view(u_end, 3:4, :)

semi = Semidiscretization(particle_container, boundary_container_tank,
                          boundary_container_wall, solid_container,
                          neighborhood_search=SpatialHashingSearch)

ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(thread=OrdinaryDiffEq.True()),
            dt=1e-5, # Small initial stepsize because the automatic choice is usually too large
            abstol=1e-5, # Higher abstol (default is 1e-6) for performance reasons
            reltol=1e-4, # Smaller reltol (default is 1e-3) to prevent boundary penetration
            save_everystep=false, callback=callbacks);
