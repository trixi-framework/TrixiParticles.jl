# 2D dam break flow against an elastic plate based on Section 6.5 of
#
# L. Zhan, C. Peng, B. Zhang, W. Wu.
# "A stabilized TL–WC SPH approach with GPU acceleration for three-dimensional fluid–structure interaction".
# In: Journal of Fluids and Structures 86 (2019), pages 329-353.
# https://doi.org/10.1016/j.jfluidstructs.2019.02.002

using Pixie
using OrdinaryDiffEq

fluid_particle_spacing = 0.01
# Spacing ratio between fluid and boundary particles
beta = 3

water_width = 0.146
water_height = 0.292
water_density = 1000.0

container_width = 0.584
container_height = 1.0

setup = RectangularTank(fluid_particle_spacing, beta, water_width, water_height,
                        container_width, container_height, water_density)

# Move right boundary
reset_right_wall!(setup, container_width, wall_position=water_width)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

state_equation = StateEquationCole(c, 7, 1000.0, 100000.0, background_pressure=100000.0)

particle_container = FluidParticleContainer(setup.particle_coordinates, setup.particle_velocities,
                                            setup.particle_masses, setup.particle_densities,
                                            ContinuityDensity(), state_equation,
                                            smoothing_kernel, smoothing_length,
                                            viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                                            acceleration=(0.0, -9.81), damping_coefficient=1e-5)

K = 9.81 * water_height
boundary_container = BoundaryParticleContainer(setup.boundary_coordinates, setup.boundary_masses,
                                               BoundaryModelMonaghanKajtar(K, beta, fluid_particle_spacing / beta))


length = 0.08
thickness = 0.012
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

n_particles_per_dimension = (n_particles_x, round(Int, length / solid_particle_spacing) + 1)

particle_coordinates = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_velocities = Array{Float64, 2}(undef, 2, prod(n_particles_per_dimension))
particle_masses = 2500 * solid_particle_spacing^2 * ones(Float64, prod(n_particles_per_dimension))
particle_densities = 2500 * ones(Float64, prod(n_particles_per_dimension))

for x in 1:n_particles_per_dimension[1],
        y in 1:n_particles_per_dimension[2]
    particle = (y - 1) * n_particles_per_dimension[1] + x

    # Coordinates
    particle_coordinates[1, particle] = 0.292 + (x - 1) * solid_particle_spacing
    particle_coordinates[2, particle] = length - (y - 1) * solid_particle_spacing

    # Velocity
    particle_velocities[1, particle] = 0.0
    particle_velocities[2, particle] = 0.0
end

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Young's modulus and Poisson ratio
E = 1e6
nu = 0.0

beta = fluid_particle_spacing / solid_particle_spacing
solid_container = SolidParticleContainer(particle_coordinates, particle_velocities, particle_masses, particle_densities,
                                         ContinuityDensity(),
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=n_particles_x,
                                         acceleration=(0.0, -9.81),
                                         # Use bigger K to prevent penetration into the solid
                                         BoundaryModelMonaghanKajtar(5K, beta, solid_particle_spacing),
                                         penalty_force=PenaltyForceGanzenmueller(alpha=0.01))


# Relaxing of the fluid without solid
semi = Semidiscretization(particle_container, boundary_container, neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

alive_callback = AliveCallback(alive_interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=alive_callback);

# Move right boundary
reset_right_wall!(setup, container_width)

# Run full simulation
tspan = (0.0, 1.0)

# Use solution of the relaxing step as initial coordinates
u_end = Pixie.wrap_array(sol[end], 1, particle_container, semi)
particle_container.initial_coordinates .= view(u_end, 1:2, :)
particle_container.initial_velocity .= view(u_end, 3:4, :)
particle_container.damping_coefficient[] = 0

semi = Semidiscretization(particle_container, boundary_container, solid_container, neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.005:20.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(alive_callback, saving_callback)

# see above
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# activate to save to vtk
# pixie2vtk(saved_values)
