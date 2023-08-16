# 2D dam break flow against an elastic plate based on Section 6.5 of
#
# L. Zhan, C. Peng, B. Zhang, W. Wu.
# "A stabilized TL–WC SPH approach with GPU acceleration for three-dimensional fluid–structure interaction".
# In: Journal of Fluids and Structures 86 (2019), pages 329-353.
# https://doi.org/10.1016/j.jfluidstructs.2019.02.002

using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

fluid_particle_spacing = 0.01

# Spacing ratio between fluid and boundary particles
beta = 1
boundary_layers = 3

water_width = 0.146
water_height = 2water_width
water_density = 1000.0

tank_width = 4water_width
tank_height = 4water_width

sound_speed = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(fluid_particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=boundary_layers, spacing_ratio=beta)

# Move right boundary.
# Use the new fluid size, since it might have been rounded in `RectangularTank`.
reset_faces = (false, true, false, false)
positions = (0, tank.fluid_size[1], 0, 0)

reset_wall!(tank, reset_faces, positions)

# ==========================================================================================
# ==== Solid

length_beam = 0.08
thickness = 0.012
solid_density = 2500
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Young's modulus and Poisson ratio
E = 1e6
nu = 0.0

n_particles_per_dimension = (n_particles_x,
                             round(Int, length_beam / solid_particle_spacing) + 1)

# The bottom layer is sampled separately below. Note that the `RectangularShape` puts the
# first particle half a particle spacing away from the boundary, which is correct for fluids,
# but not for solids. We therefore need to pass `tlsph=true`.
plate = RectangularShape(solid_particle_spacing,
                         (n_particles_per_dimension[1], n_particles_per_dimension[2] - 1),
                         (2water_width, solid_particle_spacing), solid_density, tlsph=true)
fixed_particles = RectangularShape(solid_particle_spacing,
                                   (n_particles_per_dimension[1], 1),
                                   (2water_width, 0.0),
                                   solid_density, tlsph=true)

solid = InitialCondition(plate, fixed_particles)

# ==========================================================================================
# ==== Boundary models

boundary_model = BoundaryModelDummyParticles(tank.boundary.density,
                                             tank.boundary.mass, state_equation,
                                             AdamiPressureExtrapolation(), smoothing_kernel,
                                             smoothing_length)

# K = 9.81 * water_height
# boundary_model = BoundaryModelMonaghanKajtar(K, beta, particle_spacing / beta,
#                                              tank.boundary.mass)

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites = water_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * solid_particle_spacing^2

k_solid = 9.81 * water_height
beta_solid = fluid_particle_spacing / solid_particle_spacing
boundary_model_solid = BoundaryModelMonaghanKajtar(k_solid, beta_solid,
                                                   solid_particle_spacing,
                                                   hydrodynamic_masses)

# `BoundaryModelDummyParticles` usually produces better results, since Monaghan-Kajtar BCs
# tend to introduce a non-physical gap between fluid and boundary.
# However, `BoundaryModelDummyParticles` can only be used when the plate thickness is
# at least two fluid particle spacings, so that the compact support is fully sampled,
# or fluid particles can penetrate the solid.
# For higher fluid resolutions, uncomment the code below for better results.
#
# boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
#                                                    hydrodynamic_masses, state_equation,
#                                                    AdamiPressureExtrapolation(),
#                                                    smoothing_kernel, smoothing_length)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity,
                                           acceleration=(0.0, gravity))

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

solid_system = TotalLagrangianSPHSystem(solid,
                                        solid_smoothing_kernel, solid_smoothing_length,
                                        E, nu,
                                        n_fixed_particles=n_particles_x,
                                        acceleration=(0.0, gravity),
                                        boundary_model_solid,
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

# ==========================================================================================
# ==== Simulation

# Relaxing of the fluid without solid
semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=SpatialHashingSearch,
                          damping_coefficient=1e-5)

tspan_relaxing = (0.0, 3.0)
ode = semidiscretize(semi, tspan_relaxing)

info_callback = InfoCallback(interval=100)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Enable threading of the RK method for better performance on multiple threads.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so with Monaghan-Kajtar BC because forces
# become extremely large when fluid particles are very close to boundary particles,
# and the time integration method interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=info_callback);

# Move right boundary
positions = (0, tank.tank_size[1], 0, 0)
reset_wall!(tank, reset_faces, positions)

# Run full simulation
tspan = (0.0, 1.0)

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(fluid_system, boundary_system, solid_system,
                          neighborhood_search=SpatialHashingSearch)
ode = semidiscretize(semi, tspan)

saving_callback = SolutionSavingCallback(dt=0.02)
callbacks = CallbackSet(info_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
