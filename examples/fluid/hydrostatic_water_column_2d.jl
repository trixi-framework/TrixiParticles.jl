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
initial_fluid_size = (1.0, 0.9)
tank_size = (1.0, 1.0)

fluid_density = 1000.0
sound_speed = 10.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_density_calculator = ContinuityDensity()
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           acceleration=(0.0, -gravity),
                                           source_terms=nothing)

# ==========================================================================================
# ==== Boundary

# This is to set wall viscosity with `trixi_include`
viscosity_wall = NoViscosity()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             AdamiPressureExtrapolation(),
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity_wall)
boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# This is to easily add a new callback with `trixi_include`
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control
sol = solve(ode, RDPK3SpFSAL35(), save_everystep=false, callback=callbacks);
