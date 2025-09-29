# ==========================================================================================
# 2D Periodic Channel Flow Simulation
#
# This example simulates fluid flow in a 2D channel with periodic boundary
# conditions in the flow direction (x-axis) and solid walls at the top and bottom.
# The fluid is initialized with a uniform velocity.
# This setup can be used to study Poiseuille flow or turbulent channel flow characteristics.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.02

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 1.0)

# Boundary geometry and initial fluid particle positions
tank_size = (1.0, 0.5)
initial_fluid_size = tank_size
initial_velocity = (1.0, 0.0)

fluid_density = 1000.0
sound_speed = 10 * initial_velocity[1]
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=7)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), velocity=initial_velocity)

# ==========================================================================================
# ==== Fluid
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

# `pressure_acceleration=nothing` is the default and can be overwritten with `trixi_include`
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           shifting_technique=nothing,
                                           pressure_acceleration=nothing)

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

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Simulation
periodic_box = PeriodicBox(min_corner=[0.0, -0.25], max_corner=[1.0, 0.75])
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box)

semi = Semidiscretization(fluid_system, boundary_system; neighborhood_search,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")
# This can be overwritten with `trixi_include`
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

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
