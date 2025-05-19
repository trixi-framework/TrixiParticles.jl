using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
n_particles_y = 3

# ==========================================================================================
# ==== Experiment Setup
gravity = 2.0
tspan = (0.0, 0.5)

fin_length=0.5
fin_thickness=0.02
flexural_rigidity = 5.0
poisson_ratio = 0.3
modulus = 12 * (1 - poisson_ratio^2) * flexural_rigidity / (fin_thickness^3)

fiber_volume_fraction = 0.6
fiber_density = 1800.0
epoxy_density = 1250.0
density = fiber_volume_fraction * fiber_density +
          (1 - fiber_volume_fraction) * epoxy_density

clamp_radius = 0.05

# The structure starts at the position of the first particle and ends
# at the position of the last particle.
particle_spacing = fin_thickness / (n_particles_y - 1)

# Add particle_spacing/2 to the clamp_radius to ensure that particles are also placed on the radius
fixed_particles = SphereShape(particle_spacing, clamp_radius + particle_spacing / 2,
                              (0.0, fin_thickness / 2), density,
                              cutout_min=(0.0, 0.0),
                              cutout_max=(clamp_radius, fin_thickness),
                              tlsph=true)

n_particles_clamp_x = round(Int, clamp_radius / particle_spacing)

# Beam and clamped particles
n_particles_per_dimension = (round(Int, fin_length / particle_spacing) +
                             n_particles_clamp_x + 1, n_particles_y)

# Note that the `RectangularShape` puts the first particle half a particle spacing away
# from the boundary, which is correct for fluids, but not for solids.
# We therefore need to pass `tlsph=true`.
beam = RectangularShape(particle_spacing, n_particles_per_dimension,
                        (0.0, 0.0), density=density, tlsph=true)

solid = union(beam, fixed_particles)
movement_amplitude = 0.1
solid.coordinates .+= 0.4 - 0.5 * fin_thickness #+ movement_amplitude

# Movement function
movement_function(t) = SVector(0.0, movement_amplitude * sin(5 * pi * t))#- movement_amplitude)

is_moving(t) = true

boundary_movement = BoundaryMovement(movement_function, is_moving)

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = particle_spacing

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
# tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
tank_size = (2.0, 0.8)
initial_fluid_size = tank_size
initial_velocity = (0.0, 0.0)

fluid_density = 1000.0
nu = 0.1 / fluid_density # viscosity parameter
sound_speed = 20.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, background_pressure=10_000.0)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), velocity=initial_velocity)
fluid = setdiff(tank.fluid, solid)

# ==========================================================================================
# ==== Solid
# The kernel in the reference uses a differently scaled smoothing length,
# so this is equivalent to the smoothing length of `sqrt(2) * particle_spacing` used in the paper.
smoothing_length = sqrt(2) * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_wall = nothing
# Activate to switch to no-slip walls
#viscosity_wall = ViscosityAdami(nu=0.0025 * smoothing_length * sound_speed / 8)

# For the FSI we need the hydrodynamic masses and densities in the solid boundary model
hydrodynamic_densites = fluid_density * ones(size(solid.density))
hydrodynamic_masses = hydrodynamic_densites * particle_spacing^2

boundary_model_solid = BoundaryModelDummyParticles(hydrodynamic_densites,
                                                   hydrodynamic_masses,
                                                   state_equation=state_equation,
                                                   boundary_density_calculator,
                                                   smoothing_kernel, smoothing_length,
                                                   viscosity=viscosity_wall)
# k_solid = 0.1
# beta_solid = fluid_particle_spacing / particle_spacing
# boundary_model_solid = BoundaryModelMonaghanKajtar(k_solid, beta_solid,
#                                                    particle_spacing,
#                                                    hydrodynamic_masses)

solid_system = TotalLagrangianSPHSystem(solid, smoothing_kernel, smoothing_length,
                                        modulus, poisson_ratio,
                                        n_fixed_particles=nparticles(fixed_particles),
                                        movement=boundary_movement,
                                        boundary_model=boundary_model_solid,
                                        penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           pressure_acceleration=tensile_instability_control)
# fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
#                                            sound_speed, viscosity=ViscosityAdami(; nu),
#                                            transport_velocity=TransportVelocityAdami(10 * sound_speed^2 * fluid_density))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
viscosity_wall = nothing
# Activate to switch to no-slip walls
#viscosity_wall = ViscosityAdami(nu=0.0025 * smoothing_length * sound_speed / 8)

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity_wall)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, -0.25], max_corner=[2.0, 1.0])
# cell_list = TrixiParticles.PointNeighbors.FullGridCellList(min_corner=[0.0, -0.25], max_corner=[1.0, 0.75])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)

semi = Semidiscretization(fluid_system, boundary_system, solid_system; neighborhood_search)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

callbacks = CallbackSet(info_callback, saving_callback, ParticleShiftingCallback())

# Use a Runge-Kutta method with automatic (error based) time step size control.
# Limiting of the maximum stepsize is necessary to prevent crashing.
# When particles are approaching a wall in a uniform way, they can be advanced
# with large time steps. Close to the wall, the stepsize has to be reduced drastically.
# Sometimes, the method fails to do so because forces become extremely large when
# fluid particles are very close to boundary particles, and the time integration method
# interprets this as an instability.
sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-8, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
