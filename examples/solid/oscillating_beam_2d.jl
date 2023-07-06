using TrixiParticles
using OrdinaryDiffEq

acceleration = -2.0

# ==========================================================================================
# ==== Solid

length_beam = 0.35
thickness = 0.02
n_particles_y = 5
clamp_radius = 0.05
particle_density = 1000.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_y - 1)

smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Lamé constants
E = 1.4e6
nu = 0.4

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius.
fixed_particles = CircularShape(particle_spacing, clamp_radius + particle_spacing / 2,
                                (0.0, thickness / 2),
                                shape_type=FillCircle(x_recess=(0.0, clamp_radius),
                                                      y_recess=(0.0, thickness)),
                                particle_density)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

# Beam and clamped particles
n_particles_per_dimension = (round(Int, length_beam / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for solids.
# We therefore have to subtract half a particle spacing.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (-0.5particle_spacing, -0.5particle_spacing), particle_density)

solid = InitialCondition(beam, fixed_particles)

# ==========================================================================================
# ==== Systems

solid_system = TotalLagrangianSPHSystem(solid,
                                        smoothing_kernel, smoothing_length,
                                        E, nu,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        acceleration=(0.0, acceleration),
                                        nothing) # No boundary model

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(solid_system, neighborhood_search=SpatialHashingSearch)
tspan = (0.0, 5.0)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
