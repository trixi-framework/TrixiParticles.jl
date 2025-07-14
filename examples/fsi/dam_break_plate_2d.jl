# ==========================================================================================
# 2D Dam Break Flow Against an Elastic Plate
#
# Based on Section 6.5 of:
#   L. Zhan, C. Peng, B. Zhang, W. Wu.
#   "A stabilized TL–WC SPH approach with GPU acceleration for three-dimensional fluid–structure interaction".
#   Journal of Fluids and Structures, 86 (2019), pp. 329-353.
#   https://doi.org/10.1016/j.jfluidstructs.2019.02.002
#
# This example simulates a 2D dam break where the collapsing water column impacts
# a flexible elastic plate fixed at its base.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.01
n_particles_x = 5

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.146, 2 * 0.146)
tank_size = 4 .* initial_fluid_size

fluid_density = 1000.0
sound_speed = 20 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# Elastic plate/beam
length_beam = 0.08
thickness = 0.012
solid_density = 2500

# Young's modulus and Poisson ratio
E = 1e6
nu = 0.0

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
solid_particle_spacing = thickness / (n_particles_x - 1)

n_particles_y = round(Int, length_beam / solid_particle_spacing) + 1

# The bottom layer is sampled separately below. Note that the `RectangularShape` puts the
# first particle half a particle spacing away from the boundary, which is correct for fluids,
# but not for solids. We therefore need to pass `tlsph=true`.
plate = RectangularShape(solid_particle_spacing,
                         (n_particles_x, n_particles_y - 1),
                         (2initial_fluid_size[1], solid_particle_spacing),
                         density=solid_density, tlsph=true)
fixed_particles = RectangularShape(solid_particle_spacing,
                                   (n_particles_x, 1), (2initial_fluid_size[1], 0.0),
                                   density=solid_density, tlsph=true)

solid = union(plate, fixed_particles)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.75 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Solid
solid_smoothing_length = sqrt(2) * solid_particle_spacing
solid_smoothing_kernel = WendlandC2Kernel{2}()

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites = fluid_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * solid_particle_spacing^2

k_solid = gravity * initial_fluid_size[2]
spacing_ratio_solid = fluid_particle_spacing / solid_particle_spacing
boundary_model_solid = BoundaryModelMonaghanKajtar(k_solid, spacing_ratio_solid,
                                                   solid_particle_spacing,
                                                   hydrodynamic_masses)

# `BoundaryModelDummyParticles` usually produces better results, since Monaghan-Kajtar BCs
# tend to introduce a non-physical gap between fluid and boundary.
# However, `BoundaryModelDummyParticles` can only be used when the plate thickness is
# at least two fluid particle spacings, so that the compact support is fully sampled,
# or fluid particles can penetrate the solid.
# With higher fluid resolutions, uncomment the code below for better results.
#
# boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
#                                                    hydrodynamic_masses,
#                                                    state_equation=state_equation,
#                                                    boundary_density_calculator,
#                                                    smoothing_kernel, smoothing_length)

solid_system = TotalLagrangianSPHSystem(solid,
                                        solid_smoothing_kernel, solid_smoothing_length,
                                        E, nu, boundary_model=boundary_model_solid,
                                        n_fixed_particles=n_particles_x,
                                        acceleration=(0.0, -gravity),
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, solid_system)
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
