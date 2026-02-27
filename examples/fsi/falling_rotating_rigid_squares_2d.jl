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
tank_size = (2.0, 2.0)

fluid_density = 1000.0
sound_speed = 10 * sqrt(gravity * initial_fluid_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

square1_side_length = 0.4
square2_side_length = 0.3

# Material properties [SI units]
wall_material = (; youngs_modulus=3.0e10, poisson_ratio=0.2)
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

square1 = RectangularShape(structure_particle_spacing,
                           (square1_nparticles_side, square1_nparticles_side),
                           square1_bottom_left,
                           density=material_properties.wood.density)
square2 = RectangularShape(structure_particle_spacing,
                           (square2_nparticles_side, square2_nparticles_side),
                           square2_bottom_left,
                           density=material_properties.steel.density)

# Initial rigid-body angular velocities [rad/s]
square1_angular_velocity = 5.0
square2_angular_velocity = -7.5

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

@inline shear_modulus(youngs_modulus, poisson_ratio) = youngs_modulus / (2.0 *
                                                                          (1.0 + poisson_ratio))

function effective_youngs_modulus(material, wall)
    return 1.0 / ((1.0 - material.poisson_ratio^2) / material.youngs_modulus +
                  (1.0 - wall.poisson_ratio^2) / wall.youngs_modulus)
end

function effective_shear_modulus(material, wall)
    shear_material = shear_modulus(material.youngs_modulus, material.poisson_ratio)
    shear_wall = shear_modulus(wall.youngs_modulus, wall.poisson_ratio)

    return 1.0 / ((2.0 - material.poisson_ratio) / shear_material +
                  (2.0 - wall.poisson_ratio) / shear_wall)
end

function damping_ratio_from_restitution(restitution)
    restitution_clamped = clamp(restitution, 0.0, 1.0)
    restitution_clamped >= 1.0 && return 0.0
    restitution_clamped <= eps(restitution_clamped) && return 1.0

    log_restitution = log(restitution_clamped)
    return -log_restitution / sqrt(pi^2 + log_restitution^2)
end

function make_material_contact_model(material, side_length, bottom_left)
    # For non-round bodies we use an equivalent local contact radius based on the half side.
    equivalent_radius = 0.5 * side_length
    center = (bottom_left[1] + 0.5 * side_length, bottom_left[2] + 0.5 * side_length)
    drop_height = max(center[2] - 0.5 * side_length - initial_fluid_size[2],
                      structure_particle_spacing)
    impact_velocity = sqrt(2.0 * gravity * drop_height)
    effective_E = effective_youngs_modulus(material, wall_material)
    effective_G = effective_shear_modulus(material, wall_material)

    # 2D mass proxy used by this setup.
    particle_mass = material.density * side_length^2

    # Linearize Hertz-Mindlin contact around expected peak compression from impact energy.
    hertz_coefficient = (4.0 / 3.0) * effective_E * sqrt(equivalent_radius)
    reference_penetration = ((5.0 / 4.0) * particle_mass * impact_velocity^2 /
                             hertz_coefficient)^(2.0 / 5.0)
    reference_penetration = max(reference_penetration,
                                0.01 * structure_particle_spacing)
    normal_stiffness = (3.0 / 2.0) * hertz_coefficient * sqrt(reference_penetration)

    contact_radius = sqrt(equivalent_radius * reference_penetration)
    tangential_stiffness = 8.0 * effective_G * contact_radius

    damping_ratio = damping_ratio_from_restitution(material.restitution)
    normal_damping = 2.0 * damping_ratio * sqrt(normal_stiffness * particle_mass)
    tangential_damping = 2.0 * damping_ratio * sqrt(tangential_stiffness * particle_mass)

    static_friction = material.friction_coefficient
    kinetic_friction = 0.9 * static_friction

    if static_friction <= eps(static_friction)
        tangential_stiffness = 0.0
        tangential_damping = 0.0
    end

    return RigidBoundaryContactModel(; normal_stiffness,
                                     normal_damping,
                                     static_friction_coefficient=static_friction,
                                     kinetic_friction_coefficient=kinetic_friction,
                                     tangential_stiffness,
                                     tangential_damping,
                                     contact_distance=2.0 * structure_particle_spacing,
                                     stick_velocity_tolerance=max(1e-5,
                                                                  0.01 * impact_velocity),
                                     penetration_slop=0.0)
end

boundary_model_structure_1 = make_boundary_model_structure(square1)
boundary_model_structure_2 = make_boundary_model_structure(square2)

boundary_contact_model_1 = make_material_contact_model(material_properties.wood,
                                                       square1_side_length,
                                                       square1_bottom_left)
boundary_contact_model_2 = make_material_contact_model(material_properties.steel,
                                                       square2_side_length,
                                                       square2_bottom_left)

structure_system_1 = RigidSPHSystem(square1;
                                    boundary_model=boundary_model_structure_1,
                                    boundary_contact_model=boundary_contact_model_1,
                                    acceleration=(0.0, -gravity),
                                    angular_velocity=square1_angular_velocity,
                                    particle_spacing=structure_particle_spacing)
structure_system_2 = RigidSPHSystem(square2;
                                    boundary_model=boundary_model_structure_2,
                                    boundary_contact_model=boundary_contact_model_2,
                                    acceleration=(0.0, -gravity),
                                    angular_velocity=square2_angular_velocity,
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
            reltol=1e-3, # Default reltol is 1e-3
            save_everystep=false, callback=callbacks);
