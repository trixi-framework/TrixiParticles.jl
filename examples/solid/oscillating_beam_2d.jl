using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
n_particles_y = 5

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 5.0)

length_beam = 0.35
thickness = 0.02
clamp_radius = 0.05
material_density = 1000.0

# Young's modulus and Poisson ratio
E = 1.4e6
nu = 0.4

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = thickness / (n_particles_y - 1)

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
fixed_particles = SphereShape(particle_spacing, clamp_radius + particle_spacing / 2,
                              (0.0, thickness / 2), material_density,
                              cutout_min=(0.0, 0.0), cutout_max=(clamp_radius, thickness),
                              tlsph=true)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

# Beam and clamped particles
n_particles_per_dimension = (round(Int, length_beam / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for solids.
# We therefore need to pass `tlsph=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, 0.0), density=material_density, tlsph=true)

solid = union(beam, fixed_particles)

# ==========================================================================================
# ==== Solid

smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

solid_system = TotalLagrangianSPHSystem(solid,
                                        smoothing_kernel, smoothing_length,
                                        E, nu, nothing,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        acceleration=(0.0, -gravity)) # No boundary model

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(solid_system, neighborhood_search=GridNeighborhoodSearch)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL49(), save_everystep=false, callback=callbacks);
