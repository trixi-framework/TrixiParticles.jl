using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.05

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 3

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 2.0)
tank_size = (2.0, 2.0)

fluid_density = 1000.0
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=2.0)

sphere = SphereShape(0.5 * fluid_particle_spacing, 0.3, (1.0, 1.0),
                     fluid_density, sphere_type=RoundSphere())
# ==========================================================================================
# ==== Fluid

fluid = setdiff(tank.fluid, sphere)

smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
pressure = 1000.0
particle_refinement = ParticleRefinement(;
                                         refinement_pattern=TrixiParticles.HexagonalSplitting(),
                                         max_spacing_ratio=1.2)
fluid_system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                           sound_speed; viscosity=ViscosityAdami(; nu=1e-4),
                                           particle_refinement=particle_refinement,
                                           transport_velocity=TransportVelocityAdami(pressure),
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary

# This is to set another boundary density calculation with `trixi_include`
boundary_density_calculator = AdamiPressureExtrapolation()

# This is to set wall viscosity with `trixi_include`
viscosity_wall = nothing
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity_wall)
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model, movement=nothing)

boundary_model_sphere = BoundaryModelDummyParticles(sphere.density, sphere.mass,
                                                    boundary_density_calculator,
                                                    smoothing_kernel, smoothing_length,
                                                    viscosity=viscosity_wall)
boundary_system_sphere = BoundarySPHSystem(sphere, boundary_model_sphere, movement=nothing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system, boundary_system_sphere)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# This is to easily add a new callback with `trixi_include`
extra_callback = nothing

refinement_callback = ParticleRefinementCallback(interval=1)

callbacks = CallbackSet(info_callback, saving_callback, extra_callback, UpdateCallback(),
                        refinement_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);
