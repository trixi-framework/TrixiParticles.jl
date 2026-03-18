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
fluid_particle_spacing = 0.02
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
tank_size = (2.0, 5.0)

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

# Approximate densities for a floating and a sinking square
square1_density = 600.0
square2_density = 2000.0

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
                           density=square1_density)
square2 = RectangularShape(structure_particle_spacing,
                           (square2_nparticles_side, square2_nparticles_side),
                           square2_bottom_left,
                           density=square2_density)
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
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Rigid Structures
# For FSI we need hydrodynamic masses and densities in the structure boundary model.
function structure_boundary_model(shape)
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

boundary_model_structure_1 = structure_boundary_model(square1)
boundary_model_structure_2 = structure_boundary_model(square2)

# Use a less dissipative wall contact for the denser square so its rebound is more visible.
contact_model_1 = RigidContactModel(; normal_stiffness=2.0e5,
                                    normal_damping=200.0,
                                    contact_distance=2.0 *
                                                     structure_particle_spacing)
contact_model_2 = RigidContactModel(; normal_stiffness=2.0e5,
                                    normal_damping=80.0,
                                    contact_distance=2.0 *
                                                     structure_particle_spacing)

structure_system_1 = RigidBodySystem(square1;
                                     boundary_model=boundary_model_structure_1,
                                     contact_model=contact_model_1,
                                     acceleration=(0.0, -gravity),
                                     particle_spacing=structure_particle_spacing)
structure_system_2 = RigidBodySystem(square2;
                                     boundary_model=boundary_model_structure_2,
                                     contact_model=contact_model_2,
                                     acceleration=(0.0, -gravity),
                                     particle_spacing=structure_particle_spacing)

extra_structure_systems = (nothing,)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, boundary_system,
                          structure_system_1, structure_system_2,
                          extra_structure_systems...)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01,
                                         output_directory="out",
                                         prefix="")

callbacks = CallbackSet(info_callback, saving_callback)

# Use a Runge-Kutta method with automatic (error based) time step size control.
# To prevent penetration of fluid particles through the rigid bodies or the boundary
# the values of `abstol` and `reltol` might need to be adjusted.
# `dtmax` is only necessary to prevent the time step from getting too large
# during freefall, which can lead to penetration on impact.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-6, # Default abstol is 1e-6
            reltol=1e-4, # Default reltol is 1e-3
            dtmax=1e-3,
            save_everystep=false, callback=callbacks);
