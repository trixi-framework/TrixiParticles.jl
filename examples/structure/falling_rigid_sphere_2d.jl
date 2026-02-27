# ==========================================================================================
# 2D Rigid Spheres Elastic Collision
#
# This example simulates rigid spheres colliding elastically with tank walls.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.01
fluid_smoothing_length = 1.5 * fluid_particle_spacing
fluid_smoothing_kernel = WendlandC2Kernel{2}()
structure_particle_spacing = fluid_particle_spacing

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 3
spacing_ratio = 1

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 10.0)

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (0.0, 0.0)
tank_size = (3.2, 2.0)

fluid_density = 1000.0
# Use a tank-based length scale to avoid zero sound speed in the rigid-only setup.
sound_speed = 10 * sqrt(gravity * tank_size[2])
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       faces=(true, true, true, false),
                       acceleration=(0.0, -gravity), state_equation=state_equation)

sphere_radius = 0.16

# Material properties [SI units]
wall_material = (; youngs_modulus=3.0e10, poisson_ratio=0.2)
material_properties = (
    elastic=(; density=1200.0, youngs_modulus=2.0e9, poisson_ratio=0.35,
             restitution=1.0, friction_coefficient=0.0),
    wood=(; density=650.0, youngs_modulus=1.0e10, poisson_ratio=0.35,
          restitution=0.35, friction_coefficient=0.5),
    steel=(; density=7850.0, youngs_modulus=2.1e11, poisson_ratio=0.29,
           restitution=0.8, friction_coefficient=0.55),
    rubber=(; density=1100.0, youngs_modulus=1.0e7, poisson_ratio=0.49,
            restitution=0.7, friction_coefficient=0.9)
)

wood_center = (0.45, 1.5)
elastic_center = (1.35, 1.5)
steel_center = (2.25, 1.5)
rubber_center = (2.95, 1.5)

# Use a round layout for all spheres and calibrate the perfect-elastic contact
# model below to keep rebound close to wall-normal in this multi-sphere setup.
elastic_sphere = SphereShape(structure_particle_spacing, sphere_radius, elastic_center,
                             material_properties.elastic.density,
                             sphere_type=RoundSphere())
wood_sphere = SphereShape(structure_particle_spacing, sphere_radius, wood_center,
                          material_properties.wood.density,
                          sphere_type=RoundSphere())
steel_sphere = SphereShape(structure_particle_spacing, sphere_radius, steel_center,
                           material_properties.steel.density,
                           sphere_type=RoundSphere())
rubber_sphere = SphereShape(structure_particle_spacing, sphere_radius, rubber_center,
                            material_properties.rubber.density,
                            sphere_type=RoundSphere())

# ==========================================================================================
# ==== Boundary
# Use pressure-zeroing dummy particles for pure rigid-wall contact weighting.
boundary_density_calculator = PressureZeroing()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             boundary_density_calculator,
                                             fluid_smoothing_kernel, fluid_smoothing_length)

boundary_system = WallBoundarySystem(tank.boundary, boundary_model)

# ==========================================================================================
# ==== Rigid Structures

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

function make_material_contact_model(material, center; resting_contact_projection=true)
    drop_height = max(center[2] - sphere_radius, structure_particle_spacing)
    impact_velocity = sqrt(2.0 * gravity * drop_height)
    effective_radius = sphere_radius
    effective_E = effective_youngs_modulus(material, wall_material)
    effective_G = effective_shear_modulus(material, wall_material)

    particle_mass = material.density * pi * sphere_radius^2

    # 3D Hertz-Mindlin proxy used to parameterize this 2D linear spring-damper contact model.
    # This gives plausible relative material behavior, but not exact 2D continuum contact units.
    hertz_proxy_coefficient = (4.0 / 3.0) * effective_E * sqrt(effective_radius)
    reference_penetration = ((5.0 / 4.0) * particle_mass * impact_velocity^2 /
                             hertz_proxy_coefficient)^(2.0 / 5.0)
    reference_penetration = max(reference_penetration,
                                0.01 * structure_particle_spacing)
    normal_stiffness = (3.0 / 2.0) * hertz_proxy_coefficient *
                       sqrt(reference_penetration)

    contact_radius = sqrt(effective_radius * reference_penetration)
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
                                     contact_distance=2.0 *
                                                      structure_particle_spacing,
                                     stick_velocity_tolerance=max(1e-5,
                                                                  0.01 *
                                                                  impact_velocity),
                                     penetration_slop=0.0,
                                     torque_free=true,
                                     resting_contact_projection)
end

# Use a calibrated contact model for the perfect-elastic reference sphere to
# keep rebound close to wall-normal in this multi-sphere setup.
elastic_contact_model = RigidBoundaryContactModel(; normal_stiffness=2.0e5,
                                                  normal_damping=0.0,
                                                  static_friction_coefficient=0.0,
                                                  kinetic_friction_coefficient=0.0,
                                                  tangential_stiffness=0.0,
                                                  tangential_damping=0.0,
                                                  contact_distance=2.0 *
                                                                   structure_particle_spacing,
                                                  stick_velocity_tolerance=1e-5,
                                                  penetration_slop=0.0,
                                                  torque_free=true,
                                                  resting_contact_projection=false)

# Keep impact rebound physically consistent for dissipative materials by
# disabling the resting-contact projection fallback in this impact example.
wood_contact_model = make_material_contact_model(material_properties.wood, wood_center;
                                                 resting_contact_projection=false)
steel_contact_model = make_material_contact_model(material_properties.steel, steel_center;
                                                  resting_contact_projection=false)
rubber_contact_model = make_material_contact_model(material_properties.rubber, rubber_center;
                                                   resting_contact_projection=false)

function make_rigid_structure_system(shape, boundary_contact_model)
    return RigidSPHSystem(shape;
                          acceleration=(0.0, -gravity),
                          boundary_contact_model=boundary_contact_model,
                          particle_spacing=structure_particle_spacing)
end

structure_system_elastic = make_rigid_structure_system(elastic_sphere,
                                                       elastic_contact_model)
structure_system_wood = make_rigid_structure_system(wood_sphere, wood_contact_model)
structure_system_steel = make_rigid_structure_system(steel_sphere, steel_contact_model)
structure_system_rubber = make_rigid_structure_system(rubber_sphere, rubber_contact_model)
# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(boundary_system,
                          structure_system_elastic,
                          structure_system_wood,
                          structure_system_steel,
                          structure_system_rubber)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out", prefix="")

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

# Use adaptive RK with conservative controller settings to resolve impacts
# without imposing a hard `dtmax`.
sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-7,
            reltol=5e-4,
            dt=1e-4,
            qmax=1.1,
            save_everystep=false, callback=callbacks);
