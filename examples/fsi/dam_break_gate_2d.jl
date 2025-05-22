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
# towards a vertically moving gate and a flexible elastic plate (beam) positioned
# behind the gate.
#
# Note: To accurately reproduce results from the reference paper, a significantly
# higher fluid resolution and a plate thickness closer to the paper's value (0.004m)
# are required. This example uses a coarser resolution and thicker plate for tractability.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Resolution Parameters
# Fluid particle spacing. Higher resolution generally yields more accurate FSI.
fluid_particle_spacing = 0.02 # meters
# Solid (elastic plate) particle parameters
num_particles_plate_thickness = 4 # Number of particles across the plate's thickness

# Tank boundary particle layers and spacing ratio (for SPH fixed boundaries)
tank_boundary_layers = 3
tank_spacing_ratio = 1 # Ratio of fluid particle spacing to tank boundary particle spacing
tank_boundary_particle_spacing = fluid_particle_spacing / tank_spacing_ratio

# Physical Parameters
gravity_magnitude = 9.81
system_acceleration_vec = (0.0, -gravity_magnitude)
simulation_tspan = (0.0, 1.0) # seconds

# Fluid Properties
fluid_density_ref = 997.0 # kg/m^3 (water)
# Sound speed, typically ~10 times max expected fluid velocity (sqrt(2*g*h))
initial_fluid_height = 0.4 # meters (defined later for `initial_fluid_size`)
sound_speed_fluid = 10 * sqrt(2 * gravity_magnitude * initial_fluid_height)
fluid_state_equation = StateEquationCole(sound_speed=sound_speed_fluid,
                                         reference_density=fluid_density_ref,
                                         exponent=7)

# Elastic Plate (Beam) Properties
plate_length = 0.09 # meters
# Plate thickness. Paper uses 0.004m. This example uses a thicker plate for stability
# at coarser resolution.
plate_thickness_original_paper = 0.004
plate_thickness = plate_thickness_original_paper * 10 # Scaled for this example
plate_solid_density = 1161.54 # kg/m^3
# Young's modulus and Poisson's ratio. Scaled from paper for consistency with thicker plate.
youngs_modulus_E_original_paper = 3.5e6 # Pa
youngs_modulus_E = youngs_modulus_E_original_paper / 10 # Scaled
poissons_ratio_nu = 0.45

# ------------------------------------------------------------------------------
# Experiment Setup: Tank, Fluid, Gate, and Elastic Plate
# ------------------------------------------------------------------------------
# Tank and Initial Fluid Column
initial_fluid_width = 0.2
initial_fluid_size = (initial_fluid_width, initial_fluid_height)
tank_width_domain = 0.8
tank_height_domain = 0.8
tank_domain_size = (tank_width_domain, tank_height_domain)

tank_setup = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_domain_size,
                             fluid_density_ref,
                             n_layers=tank_boundary_layers,
                             spacing_ratio=tank_spacing_ratio,
                             acceleration=system_acceleration_vec,
                             state_equation=fluid_state_equation)

# Moving Gate
# Gate is slightly taller than the initial fluid height.
gate_height = initial_fluid_height + 4 * fluid_particle_spacing
gate_num_layers_thickness = tank_boundary_layers # Same thickness as tank walls
gate_shape = RectangularShape(tank_boundary_particle_spacing,
                              (gate_num_layers_thickness,
                               round(Int, gate_height / tank_boundary_particle_spacing)),
                              (initial_fluid_width, 0.0), # Positioned at the end of the initial water column
                              density=fluid_density_ref) # Dummy density for boundary

# Gate movement function (from Sun et al., 2019)
gate_movement_function(t) = SVector(0.0, -285.115 * t^3 + 72.305 * t^2 + 0.1463 * t)
gate_is_moving(t) = t < 0.1 # Gate moves only for the first 0.1 seconds
gate_prescribed_movement = BoundaryMovement(gate_movement_function, gate_is_moving)

# Elastic Plate (Beam)
# Solid particle spacing is derived from plate thickness and number of particles across it.
solid_particle_spacing_plate = plate_thickness / (num_particles_plate_thickness - 1)
num_particles_plate_length = round(Int, plate_length / solid_particle_spacing_plate) + 1

# Position the plate: right end of the plate is 0.2m from the right tank wall.
# The plate is discretized into moving particles and a bottom layer of fixed particles.
# `tlsph=true` for `RectangularShape` ensures particles are placed at the boundary,
# not half a spacing away, which is crucial for solid mechanics discretization.
plate_fixed_end_x_position = tank_width_domain - 0.2 - plate_thickness # Left edge of fixed part
# The plate starts `solid_particle_spacing` above the fixed part for the moving section.
plate_movable_start_y = solid_particle_spacing_plate

plate_movable_particles = RectangularShape(solid_particle_spacing_plate,
                                           (num_particles_plate_thickness, num_particles_plate_length - 1),
                                           (plate_fixed_end_x_position, plate_movable_start_y),
                                           density=plate_solid_density, tlsph=true)
plate_fixed_particles = RectangularShape(solid_particle_spacing_plate,
                                         (num_particles_plate_thickness, 1), # One layer of fixed particles
                                         (plate_fixed_end_x_position, 0.0),  # At the bottom
                                         density=plate_solid_density, tlsph=true)
elastic_plate_particles = union(plate_movable_particles, plate_fixed_particles)
num_fixed_particles_plate = num_particles_plate_thickness # The bottom row is fixed

# ------------------------------------------------------------------------------
# Fluid System Setup (Weakly Compressible SPH)
# ------------------------------------------------------------------------------
fluid_smoothing_length = 1.75 * fluid_particle_spacing # Common choice for Wendland kernels
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
fluid_viscosity_model = ArtificialViscosityMonaghan(alpha=0.1, beta=0.0) # Increased alpha for stability

fluid_system = WeaklyCompressibleSPHSystem(tank_setup.fluid, fluid_density_calculator,
                                           fluid_state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length,
                                           viscosity=fluid_viscosity_model,
                                           acceleration=system_acceleration_vec,
                                           reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup (Tank Walls and Moving Gate)
# ------------------------------------------------------------------------------
boundary_density_calculator_type = AdamiPressureExtrapolation()

# Tank walls
boundary_model_tank_walls = BoundaryModelDummyParticles(tank_setup.boundary.density,
                                                        tank_setup.boundary.mass,
                                                        fluid_state_equation, # Use fluid's EoS
                                                        boundary_density_calculator_type,
                                                        fluid_smoothing_kernel, fluid_smoothing_length,
                                                        reference_particle_spacing=fluid_particle_spacing)
boundary_system_tank_walls = BoundarySPHSystem(tank_setup.boundary, boundary_model_tank_walls)

# Moving gate
boundary_model_gate = BoundaryModelDummyParticles(gate_shape.density, gate_shape.mass,
                                                  fluid_state_equation,
                                                  boundary_density_calculator_type,
                                                  fluid_smoothing_kernel, fluid_smoothing_length,
                                                  reference_particle_spacing=fluid_particle_spacing)
boundary_system_gate = BoundarySPHSystem(gate_shape, boundary_model_gate,
                                         movement=gate_prescribed_movement)

# ------------------------------------------------------------------------------
# Solid System Setup (Elastic Plate - Total Lagrangian SPH)
# ------------------------------------------------------------------------------
# Smoothing length for the solid domain (can differ from fluid).
# `sqrt(2) * solid_particle_spacing` is a common choice for TLSPH.
solid_smoothing_length_plate = sqrt(2) * solid_particle_spacing_plate
solid_smoothing_kernel_plate = WendlandC2Kernel{2}() # Use same kernel type for consistency

# For FSI, the solid's boundary model needs hydrodynamic properties (fluid density/mass)
# to correctly calculate fluid pressure forces on the solid.
hydrodynamic_densities_for_solid = fluid_density_ref .* ones(size(elastic_plate_particles.density))
hydrodynamic_masses_for_solid = hydrodynamic_densities_for_solid .* solid_particle_spacing_plate^2

# Boundary model for the solid plate (for fluid interaction)
# This model defines how fluid pressure is calculated and applied to the solid particles.
solid_fsi_boundary_model = BoundaryModelDummyParticles(hydrodynamic_densities_for_solid,
                                                       hydrodynamic_masses_for_solid,
                                                       fluid_state_equation, # Fluid's EoS for pressure calculation
                                                       boundary_density_calculator_type, # Same as fluid boundaries
                                                       fluid_smoothing_kernel, fluid_smoothing_length, # Fluid's SPH params
                                                       reference_particle_spacing=fluid_particle_spacing)

elastic_plate_system = TotalLagrangianSPHSystem(elastic_plate_particles,
                                                solid_smoothing_kernel_plate,
                                                solid_smoothing_length_plate,
                                                youngs_modulus_E, poissons_ratio_nu,
                                                boundary_model=solid_fsi_boundary_model,
                                                n_fixed_particles=num_fixed_particles_plate,
                                                acceleration=system_acceleration_vec,
                                                reference_particle_spacing=solid_particle_spacing_plate)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
semi = Semidiscretization(fluid_system, boundary_system_tank_walls,
                          boundary_system_gate, elastic_plate_system,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, simulation_tspan)

# Callbacks
info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="dam_break_gate_2d")
callbacks = CallbackSet(info_callback, saving_callback)

# Solve the ODE system.
# Tolerances and dtmax are crucial for FSI stability.
sol = solve(ode, RDPK3SpFSAL49(), # Solver from original file
            abstol=1e-6,
            reltol=1e-4,
            dtmax=1e-3, # Limit stepsize
            save_everystep=false,
            callback=callbacks)
