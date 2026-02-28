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
# Use a phase position with lower wall-discretization bias for the elastic sphere.
elastic_center = (1.20, 1.5)
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

# Use a calibrated contact model for the perfect-elastic reference sphere to
# keep rebound close to wall-normal in this multi-sphere setup.
elastic_contact_model_spec = PerfectElasticBoundaryContactModel(; normal_stiffness=2.0e5,
                                                                contact_distance=2.0 *
                                                                                 structure_particle_spacing,
                                                                stick_velocity_tolerance=1e-5,
                                                                torque_free=true)

# Keep all materials in one solve, but enable resting-contact projection for
# dissipative materials to prevent adaptive-step collapse once they settle.
# The factory uses Hertz-Mindlin linearization with a 2D per-unit-thickness
# mass proxy in this example setup.
wood_contact_model_spec = LinearizedHertzMindlinBoundaryContactModel(; material=material_properties.wood,
                                                                     wall_material,
                                                                     radius=sphere_radius,
                                                                     center=wood_center,
                                                                     gravity,
                                                                     particle_spacing=structure_particle_spacing,
                                                                     ndims=2,
                                                                     torque_free=true,
                                                                     resting_contact_projection=true)
steel_contact_model_spec = LinearizedHertzMindlinBoundaryContactModel(; material=material_properties.steel,
                                                                      wall_material,
                                                                      radius=sphere_radius,
                                                                      center=steel_center,
                                                                      gravity,
                                                                      particle_spacing=structure_particle_spacing,
                                                                      ndims=2,
                                                                      torque_free=true,
                                                                      resting_contact_projection=true)
rubber_contact_model_spec = LinearizedHertzMindlinBoundaryContactModel(; material=material_properties.rubber,
                                                                       wall_material,
                                                                       radius=sphere_radius,
                                                                       center=rubber_center,
                                                                       gravity,
                                                                       particle_spacing=structure_particle_spacing,
                                                                       ndims=2,
                                                                       torque_free=true,
                                                                       resting_contact_projection=true)

function make_rigid_structure_system(shape, boundary_contact_model_spec)
    # RigidSPHSystem converts the typed contact specification to the runtime
    # `RigidBoundaryContactModel` coefficients internally.
    return RigidSPHSystem(shape;
                          acceleration=(0.0, -gravity),
                          boundary_contact_model=boundary_contact_model_spec,
                          particle_spacing=structure_particle_spacing)
end

structure_system_elastic = make_rigid_structure_system(elastic_sphere,
                                                       elastic_contact_model_spec)
structure_system_wood = make_rigid_structure_system(wood_sphere, wood_contact_model_spec)
structure_system_steel = make_rigid_structure_system(steel_sphere, steel_contact_model_spec)
structure_system_rubber = make_rigid_structure_system(rubber_sphere,
                                                      rubber_contact_model_spec)
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
