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

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Resolution Parameters
fluid_particle_spacing = 0.01 # meters
# Solid (elastic plate) particle parameters
num_particles_plate_thickness = 5 # Number of particles across the plate's thickness

# Tank boundary particle layers and spacing ratio
tank_boundary_layers = 4
tank_spacing_ratio = 1 # Ratio of fluid particle spacing to tank boundary particle spacing

# Physical Parameters
gravity_magnitude = 9.81
system_acceleration_vec = (0.0, -gravity_magnitude)
simulation_tspan = (0.0, 1.0) # seconds

# Fluid Properties
fluid_density_ref = 1000.0 # kg/m^3 (water)
initial_fluid_height_H = 0.146 * 2 # H = 2 * 0.146m in paper notation
sound_speed_fluid = 20 * sqrt(gravity_magnitude * initial_fluid_height_H)
# Exponent 1 for state equation is common in FSI with WCSPH.
fluid_state_equation = StateEquationCole(sound_speed=sound_speed_fluid,
                                         reference_density=fluid_density_ref,
                                         exponent=1)

# Elastic Plate (Beam) Properties
plate_length = 0.08   # meters (L_s in paper)
plate_thickness = 0.012 # meters (W_s in paper)
plate_solid_density = 2500.0 # kg/m^3
youngs_modulus_E = 1.0e6  # Pa
poissons_ratio_nu = 0.0   # Paper uses nu=0, implying a simplified material model

# ------------------------------------------------------------------------------
# Experiment Setup: Tank, Fluid, and Elastic Plate
# ------------------------------------------------------------------------------
# Tank and Initial Fluid Column
initial_fluid_width_W = 0.146 # W in paper notation
initial_fluid_size = (initial_fluid_width_W, initial_fluid_height_H)
# Tank size is 4 times the initial fluid dimensions (as per visual in paper)
tank_domain_size = (4 * initial_fluid_width_W, 4 * initial_fluid_height_H)

tank_setup = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_domain_size,
                             fluid_density_ref,
                             n_layers=tank_boundary_layers,
                             spacing_ratio=tank_spacing_ratio,
                             acceleration=system_acceleration_vec,
                             state_equation=fluid_state_equation)

# Elastic Plate (Beam)
solid_particle_spacing_plate = plate_thickness / (num_particles_plate_thickness - 1)
num_particles_plate_length = round(Int, plate_length / solid_particle_spacing_plate) + 1

# Position the plate: fixed base starts at x = 2 * W (initial_fluid_width)
plate_fixed_end_x_position = 2 * initial_fluid_width_W
plate_movable_start_y = solid_particle_spacing_plate # Movable part starts one spacing above base

plate_movable_particles = RectangularShape(solid_particle_spacing_plate,
                                           (num_particles_plate_thickness, num_particles_plate_length - 1),
                                           (plate_fixed_end_x_position, plate_movable_start_y),
                                           density=plate_solid_density, tlsph=true)
plate_fixed_particles = RectangularShape(solid_particle_spacing_plate,
                                         (num_particles_plate_thickness, 1),
                                         (plate_fixed_end_x_position, 0.0),
                                         density=plate_solid_density, tlsph=true)
elastic_plate_particles = union(plate_movable_particles, plate_fixed_particles)
num_fixed_particles_plate = num_particles_plate_thickness

# ------------------------------------------------------------------------------
# Fluid System Setup (Weakly Compressible SPH)
# ------------------------------------------------------------------------------
fluid_smoothing_length = 1.75 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
fluid_viscosity_model = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)

fluid_system = WeaklyCompressibleSPHSystem(tank_setup.fluid, fluid_density_calculator,
                                           fluid_state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length,
                                           viscosity=fluid_viscosity_model,
                                           acceleration=system_acceleration_vec,
                                           reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Boundary System Setup (Tank Walls)
# ------------------------------------------------------------------------------
boundary_density_calculator_type = AdamiPressureExtrapolation()
boundary_model_tank_walls = BoundaryModelDummyParticles(tank_setup.boundary.density,
                                                        tank_setup.boundary.mass,
                                                        fluid_state_equation,
                                                        boundary_density_calculator_type,
                                                        fluid_smoothing_kernel, fluid_smoothing_length,
                                                        reference_particle_spacing=fluid_particle_spacing)
boundary_system_tank_walls = BoundarySPHSystem(tank_setup.boundary, boundary_model_tank_walls)

# ------------------------------------------------------------------------------
# Solid System Setup (Elastic Plate - Total Lagrangian SPH)
# ------------------------------------------------------------------------------
solid_smoothing_length_plate = sqrt(2) * solid_particle_spacing_plate
solid_smoothing_kernel_plate = WendlandC2Kernel{2}()

# Hydrodynamic properties for FSI boundary model on the solid
hydrodynamic_densities_for_solid = fluid_density_ref .* ones(size(elastic_plate_particles.density))
hydrodynamic_masses_for_solid = hydrodynamic_densities_for_solid .* solid_particle_spacing_plate^2

# Choice of FSI boundary model for the solid:
# Option 1: Monaghan-Kajtar (can introduce a gap, but less sensitive to plate thickness)
# k_factor_monaghan_kajtar = gravity_magnitude * initial_fluid_height_H # Characteristic pressure scale
# spacing_ratio_solid_fluid = fluid_particle_spacing / solid_particle_spacing_plate
# solid_fsi_boundary_model = BoundaryModelMonaghanKajtar(k_factor_monaghan_kajtar,
#                                                        spacing_ratio_solid_fluid,
#                                                        solid_particle_spacing_plate,
#                                                        hydrodynamic_masses_for_solid)

# Option 2: Dummy Particles (often better but needs sufficient plate thickness) - Active by default if condition met
# Condition: Plate thickness should be >= 2 * fluid_particle_spacing for full support.
can_use_dummy_particles_for_solid = plate_thickness >= 2 * fluid_particle_spacing
if can_use_dummy_particles_for_solid
    solid_fsi_boundary_model = BoundaryModelDummyParticles(hydrodynamic_densities_for_solid,
                                                           hydrodynamic_masses_for_solid,
                                                           fluid_state_equation,
                                                           boundary_density_calculator_type,
                                                           fluid_smoothing_kernel, fluid_smoothing_length,
                                                           reference_particle_spacing=fluid_particle_spacing)
    println("Using BoundaryModelDummyParticles for solid FSI.")
else
    k_factor_monaghan_kajtar = gravity_magnitude * initial_fluid_height_H
    spacing_ratio_solid_fluid = fluid_particle_spacing / solid_particle_spacing_plate
    solid_fsi_boundary_model = BoundaryModelMonaghanKajtar(k_factor_monaghan_kajtar,
                                                           spacing_ratio_solid_fluid,
                                                           solid_particle_spacing_plate,
                                                           hydrodynamic_masses_for_solid)
    println("Warning: Plate thickness is less than 2x fluid particle spacing. Using BoundaryModelMonaghanKajtar for solid FSI, which might introduce a gap.")
end


# Penalty force for contact between solid particles (if needed, Ganzenmueller type)
penalty_force_solid = PenaltyForceGanzenmueller(alpha=0.01) # From paper

elastic_plate_system = TotalLagrangianSPHSystem(elastic_plate_particles,
                                                solid_smoothing_kernel_plate,
                                                solid_smoothing_length_plate,
                                                youngs_modulus_E, poissons_ratio_nu,
                                                boundary_model=solid_fsi_boundary_model,
                                                n_fixed_particles=num_fixed_particles_plate,
                                                acceleration=system_acceleration_vec,
                                                penalty_force=penalty_force_solid,
                                                reference_particle_spacing=solid_particle_spacing_plate)

# ------------------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------------------
semi = Semidiscretization(fluid_system, boundary_system_tank_walls, elastic_plate_system,
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, simulation_tspan)

# Callbacks
info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="dam_break_plate_2d")
callbacks = CallbackSet(info_callback, saving_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6,
            reltol=1e-4,
            dtmax=1e-3,
            save_everystep=false,
            callback=callbacks)
