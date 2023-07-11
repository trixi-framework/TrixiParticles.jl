# 2D dam break flow against an elastic plate based on
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033

using TrixiParticles
using OrdinaryDiffEq

gravity = -9.81

# ==========================================================================================
# ==== Fluid

# Note that the effect of the gate is less pronounced with lower resolutions,
# since "larger" particles don't fit through the slightly opened gate.
fluid_particle_spacing = 0.02

# Spacing ratio between fluid and boundary particles
beta_tank = 1
beta_gate = 3
tank_layers = 3
gate_layers = 1

water_width = 0.2
water_height = 0.4
water_density = 997.0

tank_width = 0.8
tank_height = 4.0
gate_height = water_height + 4 * fluid_particle_spacing # Make sure that it overlaps the fluid.

sound_speed = 20 * sqrt(9.81 * water_height)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

state_equation = StateEquationCole(sound_speed, 7, water_density, 100000.0,
                                   background_pressure=100000.0)

viscosity = ArtificialViscosityMonaghan(0.02, 0.0)

tank = RectangularTank(fluid_particle_spacing, (water_width, water_height),
                       (tank_width, tank_height), water_density,
                       n_layers=tank_layers, spacing_ratio=beta_tank)

gate_position = (tank.n_particles_per_dimension[1] + 1) * fluid_particle_spacing

gate = RectangularShape(fluid_particle_spacing / beta_gate,
                        (gate_layers,
                         round(Int, gate_height / fluid_particle_spacing * beta_gate)),
                        (gate_position, fluid_particle_spacing / beta_gate), water_density)

f_x(t) = 0.0
f_y(t) = -285.115t^3 + 72.305t^2 + 0.1463t
is_moving(t) = false # No moving boundaries for the relaxing step

movement = BoundaryMovement((f_x, f_y), is_moving)

# ==========================================================================================
# ==== Solid

length_beam = 0.09
thickness = 0.004
solid_density = 1161.54
n_particles_x = 5

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Young's modulus and Poisson ratio
E = 3.5e6
nu = 0.45

n_particles_per_dimension = (n_particles_x,
                             round(Int, length_beam / solid_particle_spacing) + 1)

plate = RectangularShape(solid_particle_spacing,
                         (n_particles_per_dimension[1], n_particles_per_dimension[2] - 1),
                         (0.6, solid_particle_spacing), solid_density)
fixed_particles = RectangularShape(solid_particle_spacing,
                                   (n_particles_per_dimension[1], 1), (0.6, 0.0),
                                   solid_density)

solid = InitialCondition(plate, fixed_particles)

# ==========================================================================================
# ==== Boundary models

boundary_model_tank = BoundaryModelDummyParticles(tank.boundary.density,
                                                  tank.boundary.mass, state_equation,
                                                  AdamiPressureExtrapolation(),
                                                  smoothing_kernel,
                                                  smoothing_length)

# K_tank = 9.81 * water_height
# boundary_model_tank = BoundaryModelMonaghanKajtar(K_tank, beta_tank,
#                                                   fluid_particle_spacing / beta_tank,
#                                                   tank.boundary.mass)

K_gate = 9.81 * water_height
boundary_model_gate = BoundaryModelMonaghanKajtar(K_gate, beta_gate,
                                                  fluid_particle_spacing / beta_gate,
                                                  gate.mass)

hydrodynamic_densites = water_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * solid_particle_spacing^2

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                   hydrodynamic_masses, state_equation,
                                                   AdamiPressureExtrapolation(),
                                                   smoothing_kernel, smoothing_length)

# Use bigger K to prevent penetration into the solid
# K_solid = 9.81 * water_height
# beta_solid = fluid_particle_spacing / solid_particle_spacing
# boundary_model_solid = BoundaryModelMonaghanKajtar(K_solid, beta_solid,
#                                                    solid_particle_spacing,
#                                                    hydrodynamic_masses)

# ==========================================================================================
# ==== Systems

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, ContinuityDensity(), state_equation,
                                           smoothing_kernel, smoothing_length,
                                           viscosity=viscosity, acceleration=(0.0, gravity))

boundary_system_tank = BoundarySPHSystem(tank.boundary, boundary_model_tank)

boundary_system_gate = BoundarySPHSystem(gate, boundary_model_gate,
                                         movement=movement)

solid_system = TotalLagrangianSPHSystem(solid,
                                        solid_smoothing_kernel, solid_smoothing_length,
                                        E, nu, boundary_model_solid,
                                        n_fixed_particles=n_particles_x,
                                        acceleration=(0.0, gravity))

# ==========================================================================================
# ==== Simulation

# Relaxing of the fluid without solid
semi = Semidiscretization(fluid_system, boundary_system_tank,
                          boundary_system_gate,
                          neighborhood_search=SpatialHashingSearch)

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

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

# Run full simulation
tspan = (0.0, 1.0)

is_moving(t) = t < 0.1

# Use solution of the relaxing step as initial coordinates
restart_with!(semi, sol)

semi = Semidiscretization(fluid_system, boundary_system_tank,
                          boundary_system_gate, solid_system,
                          neighborhood_search=SpatialHashingSearch)

ode = semidiscretize(semi, tspan)

saving_callback = SolutionSavingCallback(dt=0.02)
callbacks = CallbackSet(info_callback, saving_callback)

# See above for an explanation of the parameter choice
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
