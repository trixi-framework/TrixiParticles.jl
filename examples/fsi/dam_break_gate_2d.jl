# ==========================================================================================
# 2D Dam Break Flow Against an Elastic Gate with Opening Motion
#
# Based on:
#   P.N. Sun, D. Le Touzé, A.-M. Zhang.
#   "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
#   Engineering Analysis with Boundary Elements, 104 (2019), pp. 240-258.
#   https://doi.org/10.1016/j.enganabound.2019.03.033
#
# This example simulates a 2D dam break where the water column collapses and flows
# through a vertically moving gate towards a flexible elastic plate (beam) positioned
# behind the gate.
#
# Note: To accurately reproduce results from the reference paper, a significantly
# higher fluid resolution and a plate thickness closer to the paper's value (0.004m)
# are required. This example uses a coarser resolution and thicker plate for tractability.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
# Note that the effect of the gate is less pronounced with lower fluid resolutions,
# since "larger" particles don't fit through the slightly opened gate. Lower fluid
# resolutions thereforce cause a later and more violent fluid impact against the gate.
fluid_particle_spacing = 0.02
n_particles_x = 4

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.2, 0.4)
tank_size = (0.8, 0.8)

fluid_density = 997.0
sound_speed = 10 * sqrt(2 * gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# Make the gate slightly higher than the fluid
gate_height = initial_fluid_size[2] + 4 * fluid_particle_spacing

gate = RectangularShape(boundary_particle_spacing,
                        (boundary_layers,
                         round(Int, gate_height / boundary_particle_spacing)),
                        (initial_fluid_size[1], 0.0), density=fluid_density)

# Movement of the gate according to the paper
movement_function(x, t) = x + SVector(0.0, -285.115 * t^3 + 72.305 * t^2 + 0.1463 * t)
is_moving(t) = t < 0.1

gate_movement = PrescribedMotion(movement_function, is_moving)

# Elastic plate/beam.
# The paper is using a thickness of 0.004, which only works properly when a similar fluid
# resolution is used. Increase resolution and change to 0.004 to reproduce the results.
length_beam = 0.09
thickness = 0.004 * 10
structure_density = 1161.54

# Young's modulus and Poisson ratio
E = 3.5e6 / 10
nu = 0.45

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
structure_particle_spacing = thickness / (n_particles_x - 1)

n_particles_y = round(Int, length_beam / structure_particle_spacing) + 1

# The bottom layer is sampled separately below. Note that the `RectangularShape` puts the
# first particle half a particle spacing away from the shell of the shape, which is
# correct for fluids, but not for structures. We therefore need to pass `place_on_shell=true`.
#
# The right end of the plate is 0.2 from the right end of the tank.
plate_position = 0.6 - n_particles_x * structure_particle_spacing
plate = RectangularShape(structure_particle_spacing,
                         (n_particles_x, n_particles_y - 1),
                         (plate_position, structure_particle_spacing),
                         density=structure_density, place_on_shell=true)
clamped_particles = RectangularShape(structure_particle_spacing,
                                     (n_particles_x, 1), (plate_position, 0.0),
                                     density=structure_density, place_on_shell=true)

structure = union(clamped_particles, plate)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.75 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.1, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model_tank = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                                  state_equation=state_equation,
                                                  boundary_density_calculator,
                                                  smoothing_kernel, smoothing_length)

boundary_model_gate = BoundaryModelDummyParticles(gate.density, gate.mass,
                                                  state_equation=state_equation,
                                                  boundary_density_calculator,
                                                  smoothing_kernel, smoothing_length)

boundary_system_tank = WallBoundarySystem(tank.boundary, boundary_model_tank)
boundary_system_gate = WallBoundarySystem(gate, boundary_model_gate,
                                          prescribed_motion=gate_movement)

# ==========================================================================================
# ==== Structure
structure_smoothing_length = sqrt(2) * structure_particle_spacing
structure_smoothing_kernel = WendlandC2Kernel{2}()

# For the FSI we need the hydrodynamic masses and densities in the structure boundary model
hydrodynamic_densites = fluid_density * ones(size(structure.density))
hydrodynamic_masses = hydrodynamic_densites * structure_particle_spacing^2

boundary_model_structure = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                       hydrodynamic_masses,
                                                       state_equation=state_equation,
                                                       AdamiPressureExtrapolation(),
                                                       smoothing_kernel, smoothing_length)

structure_system = TotalLagrangianSPHSystem(structure,
                                            structure_smoothing_kernel,
                                            structure_smoothing_length,
                                            E, nu, boundary_model=boundary_model_structure,
                                            clamped_particles=1:nparticles(clamped_particles),
                                            acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system_tank,
                          boundary_system_gate, structure_system,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
