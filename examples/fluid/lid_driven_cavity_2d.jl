# ==========================================================================================
# 2D Lid-Driven Cavity Simulation
#
# Based on:
#   S. Adami, X. Y. Hu, N. A. Adams.
#   "A transport-velocity formulation for smoothed particle hydrodynamics".
#   Journal of Computational Physics, Volume 241 (2013), pages 292-307.
#   https://doi.org/10.1016/j.jcp.2013.01.043
#
# This example simulates a 2D lid-driven cavity flow using SPH with a
# transport velocity formulation. The top lid moves horizontally, driving the
# fluid motion within a square cavity.
#
# The simulation can be run with either a Weakly Compressible SPH (WCSPH)
# or an Entropically Damped SPH (EDSPH) formulation for the fluid.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Resolution and simulation choices
particle_spacing = 0.02
boundary_layers = 4

# Choose fluid formulation: true for WCSPH, false for EDSPH
use_wcsph_formulation = true

# Physical parameters
reynolds_number = 100.0

# Characteristic velocity of the moving lid
lid_velocity_magnitude = 1.0

# Simulation time span
tspan = (0.0, 5.0)

# ------------------------------------------------------------------------------
# Experiment Setup
# ------------------------------------------------------------------------------
# Cavity dimensions (square cavity)
cavity_size = (1.0, 1.0)

# Fluid properties
fluid_density = 1.0

# Speed of sound. Should be high enough to ensure low compressibility.
sound_speed = 10 * lid_velocity_magnitude

# Kinematic viscosity based on Reynolds number and characteristic length/velocity
# Characteristic length is cavity_size[1] = 1.0.
kinematic_viscosity_nu = lid_velocity_magnitude * cavity_size[1] / reynolds_number
viscosity_model = ViscosityAdami(nu=kinematic_viscosity_nu)

# Characteristic pressure for TransportVelocityAdami.
# This is related to the fluid's stiffness (rho_0 * c_0^2).
characteristic_pressure = fluid_density * sound_speed^2

# Setup for the cavity: fluid particles and static boundary walls (bottom and sides)
# The top face is initially open as the lid will be a separate boundary system.
cavity = RectangularTank(particle_spacing, cavity_size, cavity_size, fluid_density,
                         n_layers=boundary_layers,
                         faces=(true, true, true, false), # (left, right, bottom, top_open)
                         pressure=characteristic_pressure) # For initializing pressure in TransportVelocity

# Setup for the moving lid (top boundary)
# The lid extends slightly beyond the cavity width to ensure full coverage
# when considering periodic boundary conditions for neighborhood search.
lid_num_particles_y = 3 # Thickness of the lid in particle layers
lid_num_particles_x = cavity.n_particles_per_dimension[1] + 2 * boundary_layers
lid_size = (lid_num_particles_x * particle_spacing, lid_num_particles_y * particle_spacing)

# Position the lid centered above the cavity.
# Lid's bottom edge aligns with the top of the cavity.
lid_center_x = cavity_size[1] / 2
lid_initial_position_x = lid_center_x - lid_size[1] / 2
lid_initial_position_y = cavity_size[2]
lid_initial_position = (lid_initial_position_x, lid_initial_position_y)

lid_particles = RectangularShape(particle_spacing,
                                 (lid_num_particles_x, lid_num_particles_y),
                                 lid_initial_position, density=fluid_density)

# ------------------------------------------------------------------------------
# Fluid System Setup
# ------------------------------------------------------------------------------

smoothing_length = 1.0 * particle_spacing
smoothing_kernel = SchoenbergQuinticSplineKernel{2}()

# Transport velocity formulation requires a characteristic pressure.
transport_velocity_adami = TransportVelocityAdami(characteristic_pressure)

if use_wcsph_formulation
    density_calculator = ContinuityDensity()
    state_equation = StateEquationCole(sound_speed=sound_speed,
                                       reference_density=fluid_density,
                                       exponent=1)

    # Inter-particle averaged pressure gradient (Adami et al. 2013, Eq. 17)
    pressure_acceleration_formulation = TrixiParticles.inter_particle_averaged_pressure

    fluid_system = WeaklyCompressibleSPHSystem(cavity.fluid, density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length,
                                               pressure_acceleration=pressure_acceleration_formulation,
                                               viscosity=viscosity_model,
                                               transport_velocity=transport_velocity_adami,
                                               reference_particle_spacing=particle_spacing)
else # EDSPH formulation
    state_equation = nothing # EDSPH uses sound_speed directly for pressure-like terms.
    density_calculator = ContinuityDensity()

    fluid_system = EntropicallyDampedSPHSystem(cavity.fluid, smoothing_kernel,
                                               smoothing_length,
                                               sound_speed,
                                               density_calculator=density_calculator,
                                               viscosity=viscosity_model,
                                               transport_velocity=transport_velocity_adami,
                                               reference_particle_spacing=particle_spacing)
end

# ------------------------------------------------------------------------------
# Boundary System Setup
# ------------------------------------------------------------------------------

# Lid movement: horizontal motion with constant velocity
lid_movement_function(t) = SVector(lid_velocity_magnitude * t, 0.0)
is_moving_lid(t) = true # Lid is always moving
lid_movement = BoundaryMovement(lid_movement_function, is_moving_lid)

# Boundary model for cavity walls and lid (dummy particles)
# AdamiPressureExtrapolation is used for pressure at the boundary.
boundary_density_calculator = AdamiPressureExtrapolation()

boundary_model_cavity = BoundaryModelDummyParticles(cavity.boundary.density,
                                                    cavity.boundary.mass,
                                                    boundary_density_calculator,
                                                    state_equation=state_equation,
                                                    smoothing_kernel,
                                                    smoothing_length,
                                                    viscosity=viscosity_model,
                                                    reference_particle_spacing=particle_spacing)


boundary_model_lid = BoundaryModelDummyParticles(lid_particles.density,
                                                 lid_particles.mass,
                                                 boundary_density_calculator,
                                                 state_equation=state_equation,
                                                 smoothing_kernel,
                                                 smoothing_length,
                                                 viscosity=viscosity_model,
                                                 reference_particle_spacing=particle_spacing)

boundary_system_cavity = BoundarySPHSystem(cavity.boundary, boundary_model_cavity)


boundary_system_lid = BoundarySPHSystem(lid_particles, boundary_model_lid,
                                        movement=lid_movement)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
# Periodic boundary conditions in x-direction for the neighborhood search.
# This is primarily for the lid particles that move horizontally.
# The domain for periodicity needs to encompass the cavity and the extent of lid movement.
# Define `min_corner` and `max_corner` for the periodic box used by the neighborhood search.
# This allows particles exiting one side (e.g., lid particles) to interact with particles
# from the other side.
boundary_thickness = boundary_layers * particle_spacing

# The periodic box extends beyond the physical cavity to accommodate boundary layers.
periodic_box_min_x = -boundary_thickness
periodic_box_max_x = cavity_size[1] + boundary_thickness
periodic_box_min_y = -boundary_thickness
periodic_box_max_y = cavity_size[2] + boundary_thickness + lid_size[2] # Include lid height

periodic_box = PeriodicBox(min_corner=[periodic_box_min_x, periodic_box_min_y],
                           max_corner=[periodic_box_max_x, periodic_box_max_y])

# Neighborhood search with periodic conditions and an update strategy for moving particles.
neighborhood_search = GridNeighborhoodSearch{2}(; periodic_box=periodic_box)

semi = Semidiscretization(fluid_system, boundary_system_cavity, boundary_system_lid,
                          neighborhood_search=neighborhood_search,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="")

# `UpdateCallback` can be used for tasks like re-evaluating particle lists for periodic boundaries
# or other dynamic updates needed during the simulation. Here, it ensures the neighborhood search
# handles the moving lid and periodic conditions correctly.
extra_callbacks = UpdateCallback()

callbacks = CallbackSet(info_callback, saving_callback, extra_callbacks)

# Time integration using a Runge-Kutta method with adaptive time stepping.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6 (may needs to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may needs to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            maxiters=Int(1e7),
            save_everystep=false, callback=callbacks);
