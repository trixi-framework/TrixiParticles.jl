# 2D dam break flow against an elastic plate based on
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033

using Pixie
using OrdinaryDiffEq

# Note that the effect of the gate is less pronounced with lower resolutions,
# since "larger" particles don't fit through the slightly opened gate.
fluid_particle_spacing = 0.02
# Spacing ratio between fluid and boundary particles
beta = 3
n_layers = 1

water_width = 0.2
water_height = 0.4
water_density = 997.0

container_width = 0.8
container_height = 4.0
wall_height = container_height

setup = RectangularTank(fluid_particle_spacing, beta, water_width, water_height,
                        container_width, container_height, water_density,
                        n_layers=n_layers)

gate_position = (setup.n_particles_per_dimension[1] + 1) * fluid_particle_spacing
setup_gate = RectangularShape(fluid_particle_spacing / beta,
                              (n_layers,
                               round(Int, container_height / fluid_particle_spacing * beta)),
                              (gate_position, fluid_particle_spacing / beta),
                              density=water_density)

c = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

state_equation = StateEquationCole(c, 7, 997.0, 100000.0, background_pressure=100000.0)

particle_container = new_fluid(setup, ContinuityDensity(), state_equation,
                               smoothing_kernel, smoothing_length,
                               viscosity=ArtificialViscosityMonaghan(0.02, 0.0),
                               acceleration=(0.0, -9.81))

# Add a factor of 4 to prevent boundary penetration
K = 9.81 * water_height

boundary_container_tank = BoundaryParticleContainer(setup.boundary_coordinates,
                                                    setup.boundary_masses,
                                                    BoundaryModelMonaghanKajtar(K, beta,
                                                                                fluid_particle_spacing /
                                                                                beta))

# No moving boundaries for the relaxing step
movement_function(coordinates, t) = false
boundary_container_gate = BoundaryParticleContainer(setup_gate.coordinates,
                                                    setup_gate.masses,
                                                    BoundaryModelMonaghanKajtar(K, beta,
                                                                                fluid_particle_spacing /
                                                                                beta),
                                                    movement_function=movement_function)

length = 0.09
thickness = 0.004
solid_density = 1161.54
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

n_particles_per_dimension = (n_particles_x, round(Int, length / solid_particle_spacing) + 1)

plate = RectangularShape(solid_particle_spacing,
                         (n_particles_per_dimension[1], n_particles_per_dimension[2] - 1),
                         (0.6, solid_particle_spacing),
                         density=solid_density)
fixed_particles = RectangularShape(solid_particle_spacing,
                                   (n_particles_per_dimension[1], 1),
                                   (0.6, 0.0), density=solid_density)

particle_coordinates = hcat(plate.coordinates, fixed_particles.coordinates)
particle_velocities = zeros(Float64, 2, prod(n_particles_per_dimension))
particle_masses = vcat(plate.masses, fixed_particles.masses)
particle_densities = vcat(plate.densities, fixed_particles.densities)

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
search_radius = Pixie.compact_support(smoothing_kernel, smoothing_length)

# Young's modulus and Poisson ratio
E = 3.5e6
nu = 0.45

beta = fluid_particle_spacing / solid_particle_spacing
solid_container = SolidParticleContainer(particle_coordinates, particle_velocities,
                                         particle_masses, particle_densities,
                                         smoothing_kernel, smoothing_length,
                                         E, nu,
                                         n_fixed_particles=n_particles_x,
                                         acceleration=(0.0, -9.81),
                                         # Use bigger K to prevent penetration into the solid
                                         BoundaryModelMonaghanKajtar(5K, beta,
                                                                     solid_particle_spacing))

# Relaxing of the fluid without solid
semi = Semidiscretization(particle_container, boundary_container_tank,
                          boundary_container_gate,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=100)

callbacks = CallbackSet(summary_callback, alive_callback)

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
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# Run full simulation
tspan = (0.0, 1.0)

function movement_function(coordinates, t)
    if t < 0.1
        particle_spacing = coordinates[2, 2] - coordinates[2, 1]
        f(t) = -285.115t^3 + 72.305t^2 + 0.1463t + particle_spacing
        pos_1 = coordinates[2, 1]
        pos_2 = f(t)
        diff_pos = pos_2 - pos_1
        coordinates[2, :] .+= diff_pos

        return true
    end

    return false
end

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(particle_container, boundary_container_tank,
                          boundary_container_gate, solid_container,
                          neighborhood_search=SpatialHashingSearch)

ode = semidiscretize(semi, tspan)

saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.005:20.0,
                                                       index=(v, u, t, container) -> Pixie.eachparticle(container))

callbacks = CallbackSet(summary_callback, alive_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

# activate to save to vtk
# pixie2vtk(saved_values)
