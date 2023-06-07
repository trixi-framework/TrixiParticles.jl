using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

fluid_particle_spacing = 0.0125 * 3

water_width = 0.525
water_height = 1.0125
water_density = 1000.0

sound_speed = 10 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

fluid = RectangularShape(fluid_particle_spacing,
                         (round(Int, (water_width / fluid_particle_spacing)),
                          round(Int, (water_height / fluid_particle_spacing))), (0.1, 0.2),
                         water_density)

# ==========================================================================================
# ==== Solid

length_beam = 0.35
thickness = 0.05
n_particles_y = 5
clamp_radius = 0.05
solid_density = 1000.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_y - 1)

smoothing_length = sqrt(2) * solid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Lamé constants
E = 1.4e6
nu = 0.4

fixed_particles = CircularShape(solid_particle_spacing,
                                clamp_radius + solid_particle_spacing / 2,
                                (0.0, thickness / 2),
                                shape_type=FillCircle(x_recess=(0.0, clamp_radius),
                                                      y_recess=(0.0, thickness)),
                                solid_density)

n_particles_clamp_x = round(Int, clamp_radius / solid_particle_spacing)

# Cantilever and clamped particles
n_particles_per_dimension = (round(Int, length_beam / solid_particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

beam = RectangularShape(solid_particle_spacing, n_particles_per_dimension, (0, 0),
                        solid_density)

solid = InitialCondition(beam, fixed_particles)

# ==========================================================================================
# ==== Boundary models

K = 9.81 * water_height
beta = fluid_particle_spacing / solid_particle_spacing

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites = water_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * solid_particle_spacing^2

boundary_model = BoundaryModelMonaghanKajtar(K, beta, solid_particle_spacing,
                                             hydrodynamic_masses)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(fluid,
                                           ContinuityDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           water_density,
                                           viscosity=viscosity,
                                           acceleration=(0.0, gravity))

solid_system = TotalLagrangianSPHSystem(solid,
                                        smoothing_kernel, smoothing_length,
                                        E, nu,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        acceleration=(0.0, gravity), boundary_model,
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.1))

# ==========================================================================================
# ==== Simulation

semi = Semidiscretization(fluid_system, solid_system,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.005)

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
