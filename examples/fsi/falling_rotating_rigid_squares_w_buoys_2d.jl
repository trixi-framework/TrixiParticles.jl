# ==========================================================================================
# 2D Falling Rotating Rigid Squares in Fluid with buoys (FSI)
#
# This example simulates two rigid squares with initial angular velocity
# falling into a fluid with buoys ontop in a tank.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

tspan = (0.0, 2.0)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "falling_rotating_rigid_squares_2d.jl"),
              sol=nothing);

small_sphere_radius = 0.15
small_sphere_density = 500.0
small_sphere_y = initial_fluid_size[2] + small_sphere_radius
small_sphere_x_positions = 0.2:(3 * small_sphere_radius):1.8
small_sphere_contact_model = RigidContactModel(; normal_stiffness=2.0e5,
                                               normal_damping=120.0,
                                               static_friction_coefficient=0.6,
                                               kinetic_friction_coefficient=0.4,
                                               tangential_stiffness=1.0e5,
                                               tangential_damping=150.0,
                                               contact_distance=2.0 *
                                                                structure_particle_spacing)
extra_structure_systems = [begin
                               sphere = SphereShape(structure_particle_spacing,
                                                    small_sphere_radius,
                                                    (x, small_sphere_y),
                                                    small_sphere_density,
                                                    sphere_type=RoundSphere())
                               sphere_boundary_model = structure_boundary_model(sphere)
                               RigidBodySystem(sphere;
                                               boundary_model=sphere_boundary_model,
                                               contact_model=small_sphere_contact_model,
                                               acceleration=(0.0, -gravity),
                                               particle_spacing=structure_particle_spacing)
                           end
                           for x in small_sphere_x_positions]

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "falling_rotating_rigid_squares_2d.jl"),
              extra_structure_systems=extra_structure_systems, tspan=tspan);
