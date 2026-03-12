# ==========================================================================================
# 2D Falling Rotating Rigid Squares in Fluid (FSI)
#
# This example simulates two rigid squares with initial angular velocity
# falling into a fluid in a tank.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.01
structure_particle_spacing = fluid_particle_spacing

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (2.0, 1.0)
tank_size = (2.0, 3.0)

fluid_density = 1000.0
sound_speed = 100.0
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

square1_side_length = 0.4
square2_side_length = 0.3

# Material properties [SI units]
wall_material = (; youngs_modulus=3.0e11, poisson_ratio=0.3)
material_properties = (
    wood=(; density=650.0, youngs_modulus=1.0e10, poisson_ratio=0.35,
          restitution=0.35, friction_coefficient=0.5),
    steel=(; density=7850.0, youngs_modulus=2.1e11, poisson_ratio=0.29,
           restitution=0.8, friction_coefficient=0.55)
)

square1_nparticles_side = round(Int, square1_side_length / structure_particle_spacing)
square2_nparticles_side = round(Int, square2_side_length / structure_particle_spacing)

# Lower-left corners of the two squares
square1_bottom_left = (0.4, 1.5)
square2_bottom_left = (1.25, 1.55)

# Initial rigid-body angular velocities [rad/s]
square1_angular_velocity = 5.0
square2_angular_velocity = -7.5

square1 = RectangularShape(structure_particle_spacing,
                           (square1_nparticles_side, square1_nparticles_side),
                           square1_bottom_left,
                           density=material_properties.wood.density)
square2 = RectangularShape(structure_particle_spacing,
                           (square2_nparticles_side, square2_nparticles_side),
                           square2_bottom_left,
                           density=material_properties.steel.density)
square1 = apply_angular_velocity(square1, square1_angular_velocity)
square2 = apply_angular_velocity(square2, square2_angular_velocity)

# ==========================================================================================
# ==== Fluid
fluid_smoothing_length = 1.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()
viscosity = ArtificialViscosityMonaghan(alpha=0.02, beta=0.0)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, fluid_smoothing_kernel,
                                           fluid_smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = BernoulliPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Rigid Structures
# For FSI we need hydrodynamic masses and densities in the structure boundary model.
function make_boundary_model_structure(shape)
    hydrodynamic_densities = fluid_density * ones(size(shape.density))
    hydrodynamic_masses = hydrodynamic_densities *
                          structure_particle_spacing^ndims(fluid_system)

    return BoundaryModelDummyParticles(hydrodynamic_densities,
                                       hydrodynamic_masses,
                                       state_equation=state_equation,
                                       boundary_density_calculator,
                                       fluid_smoothing_kernel,
                                       fluid_smoothing_length)
end

boundary_model_structure_1 = make_boundary_model_structure(square1)
boundary_model_structure_2 = make_boundary_model_structure(square2)

# Use Hertz-Mindlin linearization with equivalent-radius and 2D body-mass proxy
# calibration for non-round bodies.
square1_equivalent_radius = 0.5 * square1_side_length
square2_equivalent_radius = 0.5 * square2_side_length
square1_center = (square1_bottom_left[1] + square1_equivalent_radius,
                  square1_bottom_left[2] + square1_equivalent_radius)
square2_center = (square2_bottom_left[1] + square2_equivalent_radius,
                  square2_bottom_left[2] + square2_equivalent_radius)
drop_height_1 = max(square1_center[2] - square1_equivalent_radius - initial_fluid_size[2],
                    structure_particle_spacing)
drop_height_2 = max(square2_center[2] - square2_equivalent_radius - initial_fluid_size[2],
                    structure_particle_spacing)
impact_velocity_1 = sqrt(2.0 * gravity * drop_height_1)
impact_velocity_2 = sqrt(2.0 * gravity * drop_height_2)
body_mass_1 = material_properties.wood.density * square1_side_length^2
body_mass_2 = material_properties.steel.density * square2_side_length^2

boundary_contact_model_spec_1 = LinearizedHertzMindlinBoundaryContactModel(;
                                                                            material=material_properties.wood,
                                                                            wall_material,
                                                                            radius=square1_equivalent_radius,
                                                                            impact_velocity=impact_velocity_1,
                                                                            body_mass=body_mass_1,
                                                                            particle_spacing=structure_particle_spacing,
                                                                            ndims=2,
                                                                            torque_free=false,
                                                                            resting_contact_projection=true)
boundary_contact_model_spec_2 = LinearizedHertzMindlinBoundaryContactModel(;
                                                                            material=material_properties.steel,
                                                                            wall_material,
                                                                            radius=square2_equivalent_radius,
                                                                            impact_velocity=impact_velocity_2,
                                                                            body_mass=body_mass_2,
                                                                            particle_spacing=structure_particle_spacing,
                                                                            ndims=2,
                                                                            torque_free=false,
                                                                            resting_contact_projection=true)

# RigidBodySystem converts each typed contact specification to runtime
# `RigidBoundaryContactModel` coefficients internally.
structure_system_1 = RigidBodySystem(square1;
                                    boundary_model=boundary_model_structure_1,
                                    boundary_contact_model=boundary_contact_model_spec_1,
                                    acceleration=(0.0, -gravity),
                                    particle_spacing=structure_particle_spacing)
structure_system_2 = RigidBodySystem(square2;
                                    boundary_model=boundary_model_structure_2,
                                    boundary_contact_model=boundary_contact_model_spec_2,
                                    acceleration=(0.0, -gravity),
                                    particle_spacing=structure_particle_spacing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system,
                          structure_system_1, structure_system_2)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01,
                                         output_directory="out",
                                         prefix="")
update_callback = UpdateCallback()

callbacks = CallbackSet(info_callback, saving_callback, update_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6
            reltol=1e-5, # Default reltol is 1e-3
            save_everystep=false, callback=callbacks);
