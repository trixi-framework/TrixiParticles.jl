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

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

# Resolution
particle_spacing = 0.02

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# Simulation time span
tspan = (0.0, 1.0)

# ------------------------------------------------------------------------------
# Experiment Setup:
# ------------------------------------------------------------------------------

# Channel dimensions (width, height)
channel_size = (1.0, 0.5)
initial_fluid_size = channel_size # Fluid fills the entire channel initially

# Initial uniform velocity of the fluid in the x-direction
initial_fluid_velocity = (1.0, 0.0)

# Fluid properties
fluid_density_ref = 1000.0
sound_speed = initial_fluid_velocity[1]

state_equation = StateEquationCole(sound_speed=sound_speed,
                                   reference_density=fluid_density_ref,
                                   exponent=7)

# Setup for the rectangular channel (tank).
# Fluid particles fill the domain, with boundary particles for top/bottom walls.
# `faces=(false, false, true, true)` means no explicit boundary particles on left/right
# faces, as these will be handled by periodicity.
tank = RectangularTank(particle_spacing, initial_fluid_size, channel_size,
                       fluid_density_ref,
                       n_layers=boundary_layers,
                       spacing_ratio=spacing_ratio,
                       faces=(false, false, true, true), # (left, right, bottom, top)
                       velocity=initial_fluid_velocity) # Initialize fluid with this velocity

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.2 * particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity_model = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

# `pressure_acceleration=nothing` uses the default WCSPH pressure gradient formulation.
# This can be overridden by `trixi_include` if a different formulation is desired.
fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length,
                                           viscosity=viscosity_model,
                                           pressure_acceleration=nothing,
                                           reference_particle_spacing=particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

boundary_density_calculator = AdamiPressureExtrapolation()

# Wall viscosity model. `nothing` implies slip walls.
# To activate no-slip walls, uncomment and adjust the ViscosityAdami parameters.
viscosity_wall_model = nothing
# Example for no-slip walls:
# nu_wall = 0.0025 * smoothing_length * sound_speed / 8
# viscosity_wall_model = ViscosityAdami(nu=nu_wall)

boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=viscosity_wall_model,
                                             reference_particle_spacing=particle_spacing)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model)

# ------------------------------------------------------------------------------
# Simulation Setup:
# ------------------------------------------------------------------------------

# Periodic boundary conditions in the x-direction for the neighborhood search.
# The y-direction is bounded by solid walls.
# The periodic box should span the channel width and extend sufficiently in y.
periodic_box_min_x = 0.0
periodic_box_max_x = channel_size[1]

# Extend y-range for periodicity to safely include boundary layers, although y is not periodic.
periodic_box_min_y = -0.5 * channel_size[2]
periodic_box_max_y = 1.5 * channel_size[2]

periodic_box = PeriodicBox(min_corner=[periodic_box_min_x, periodic_box_min_y],
                           max_corner=[periodic_box_max_x, periodic_box_max_y])

neighborhood_search = GridNeighborhoodSearch{2}(periodic_box=periodic_box)

semi = Semidiscretization(fluid_system, boundary_system,
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# Placeholder for additional callbacks that can be passed via `trixi_include`.
extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, extra_callback)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------

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
