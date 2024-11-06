using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.02

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
# Note: Due to the dynamics at the inlets and outlets of open boundaries,
# it is recommended to use `open_boundary_layers > boundary_layers`
open_boundary_layers = 8

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
domain_size = (2.5, 1.0)

boundary_size = (domain_size[1] + 2 * particle_spacing * open_boundary_layers,
                 domain_size[2])

fluid_density = 1000.0

sound_speed = 10 * sqrt(gravity * domain_size[2])

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

tank = RectangularTank(particle_spacing, domain_size, boundary_size, fluid_density,
                       acceleration=(0.0, -gravity), state_equation=state_equation,
                       n_layers=boundary_layers, faces=(false, false, true, false))

# Shift tank walls in negative x-direction for the left boundary zone
tank.boundary.coordinates[1, :] .-= particle_spacing * open_boundary_layers

buffer_left = RectangularShape(particle_spacing,
                               (open_boundary_layers, tank.n_particles_per_dimension[2]),
                               (-particle_spacing * open_boundary_layers, 0.0),
                               acceleration=(0.0, -gravity), state_equation=state_equation)
buffer_right = RectangularShape(particle_spacing,
                                (open_boundary_layers, tank.n_particles_per_dimension[2]),
                                (domain_size[1], 0.0),
                                acceleration=(0.0, -gravity), state_equation=state_equation)

n_buffer_particles = 40 * tank.n_particles_per_dimension[2]

sphere1_radius = 0.2
nu = 0.0
sphere1_E = 7e4

sphere1_center = (1.25, 1.6)
sphere1 = SphereShape(particle_spacing, sphere1_radius, sphere1_center,
                      500.0, sphere_type=VoxelSphere())

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.0 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

# viscosity = ArtificialViscosityMonaghan(; alpha=0.02, beta=0.0)

# fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
#                                            state_equation, smoothing_kernel,
#                                            smoothing_length, viscosity=viscosity,
#                                            acceleration=(0.0, -gravity),
#                                            buffer_size=n_buffer_particles)

viscosity = ViscosityAdami(nu=1e-5)
fluid_system = EntropicallyDampedSPHSystem(tank.fluid, smoothing_kernel, smoothing_length,
                                           sound_speed, viscosity=viscosity,
                                           acceleration=(0.0, -gravity),
                                           density_calculator=fluid_density_calculator,
                                           buffer_size=n_buffer_particles)

zone_left = BoundaryZone(; plane=([0.0, 0.0], [0.0, 1.5 * domain_size[2]]),
                         initial_condition=buffer_left,
                         plane_normal=[1.0, 0.0], open_boundary_layers,
                         density=fluid_density, particle_spacing)

open_boundary_in = OpenBoundarySPHSystem(zone_left; fluid_system,
                                         boundary_model=BoundaryModelTafuni(),
                                         buffer_size=n_buffer_particles)

outflow = BoundaryZone(;
                       plane=([domain_size[1], 0.0],
                              [domain_size[1], 1.5 * domain_size[2]]),
                       initial_condition=buffer_right,
                       plane_normal=-[1.0, 0.0], open_boundary_layers,
                       density=fluid_density, particle_spacing)

open_boundary_out = OpenBoundarySPHSystem(outflow; fluid_system,
                                          boundary_model=BoundaryModelTafuni(),
                                          buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Solid
solid_smoothing_length = 2 * sqrt(2) * particle_spacing
solid_smoothing_kernel = WendlandC2Kernel{2}()

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites_1 = fluid_density * ones(size(sphere1.density))
hydrodynamic_masses_1 = hydrodynamic_densites_1 * particle_spacing^ndims(fluid_system)

solid_boundary_model_1 = BoundaryModelDummyParticles(hydrodynamic_densites_1,
                                                     hydrodynamic_masses_1,
                                                     state_equation=nothing,
                                                     AdamiPressureExtrapolation(),
                                                     smoothing_kernel, smoothing_length)

solid_system = TotalLagrangianSPHSystem(sphere1,
                                        solid_smoothing_kernel, solid_smoothing_length,
                                        sphere1_E, nu,
                                        acceleration=(0.0, -gravity),
                                        boundary_model=solid_boundary_model_1,
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.3))

# ==========================================================================================
# ==== Boundary

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=nothing,
                                             smoothing_kernel, smoothing_length)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, open_boundary_in, open_boundary_out,
                          boundary_system, solid_system)

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-3, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-3, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
