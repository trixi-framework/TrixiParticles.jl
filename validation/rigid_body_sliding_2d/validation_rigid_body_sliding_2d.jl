# ==========================================================================================
# 2D Rigid-Body Sliding Validation
#
# A rigid square slides on a horizontal wall under Coulomb friction. While it is slipping,
# its center-of-mass acceleration is -mu_k * g, so its analytical stopping distance is
# v_0^2 / (2 * mu_k * g). Running the case at two wall resolutions also verifies that
# geometry-normal wall contact is independent of tangential wall-particle sampling.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution
particle_spacing = 0.03
wall_particle_spacings = (particle_spacing, particle_spacing / 2)
boundary_layers = 3
contact_distance = 2.0 * particle_spacing

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 0.3)
gravity = 9.81
initial_velocity = 1.0
kinetic_friction_coefficient = 0.4
analytical_stopping_distance = initial_velocity^2 /
                               (2 * kinetic_friction_coefficient * gravity)

square_side_length = 0.18
square_density = 1000.0
square_particles_per_side = round(Int, square_side_length / particle_spacing)

# ==========================================================================================
# ==== Run Simulations and Compute Errors
function run_rigid_body_sliding_validation(wall_particle_spacing; tspan)
    # Align the bottom rigid particles one contact distance above the top wall particles.
    square_bottom_y = contact_distance -
                      (particle_spacing + wall_particle_spacing) / 2
    square = RectangularShape(particle_spacing,
                              (square_particles_per_side, square_particles_per_side),
                              (0.0, square_bottom_y), density=square_density,
                              velocity=(initial_velocity, 0.0))

    floor = RectangularTank(wall_particle_spacing, (0.0, 0.0),
                            (0.99, particle_spacing), square_density;
                            n_layers=boundary_layers, min_coordinates=(-0.35, 0.0),
                            faces=(false, false, true, false))
    boundary_model = BoundaryModelMonaghanKajtar(10.0, 1.0, wall_particle_spacing,
                                                 floor.boundary.mass)
    boundary_system = WallBoundarySystem(floor.boundary, boundary_model)

    contact_model = RigidContactModel(; normal_stiffness=2.0e5,
                                      normal_damping=100.0,
                                      static_friction_coefficient=0.6,
                                      kinetic_friction_coefficient,
                                      tangential_stiffness=1.0e5,
                                      tangential_damping=150.0,
                                      contact_distance)
    rigid_system = RigidBodySystem(square; contact_model,
                                   acceleration=(0.0, -gravity),
                                   particle_spacing)

    semi = Semidiscretization(rigid_system, boundary_system)
    ode = semidiscretize(semi, tspan)
    callbacks = CallbackSet(UpdateCallback(), StepsizeCallback(cfl=0.5))
    sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-4,
                save_everystep=false, callback=callbacks)

    initial_v_ode, initial_u_ode = sol.u[begin].x
    final_v_ode, final_u_ode = sol.u[end].x
    initial_velocity_state = TrixiParticles.wrap_v(initial_v_ode, rigid_system, semi)
    initial_coordinates_state = TrixiParticles.wrap_u(initial_u_ode, rigid_system, semi)
    final_velocity_state = TrixiParticles.wrap_v(final_v_ode, rigid_system, semi)
    final_coordinates_state = TrixiParticles.wrap_u(final_u_ode, rigid_system, semi)

    initial_center_of_mass,
    _ = TrixiParticles.rigid_center_of_mass_kinematics(rigid_system,
                                                       initial_coordinates_state,
                                                       initial_velocity_state)
    final_center_of_mass,
    final_center_of_mass_velocity = TrixiParticles.rigid_center_of_mass_kinematics(rigid_system,
                                                                                   final_coordinates_state,
                                                                                   final_velocity_state)

    stopping_distance = final_center_of_mass[1] - initial_center_of_mass[1]
    relative_error = abs(stopping_distance - analytical_stopping_distance) /
                     analytical_stopping_distance

    return (; wall_particle_spacing, sol, stopping_distance, relative_error,
            final_horizontal_velocity=final_center_of_mass_velocity[1])
end

validation_results = [run_rigid_body_sliding_validation(wall_particle_spacing; tspan)
                      for wall_particle_spacing in wall_particle_spacings]
solutions = [result.sol for result in validation_results]
stopping_distance_errors = [result.relative_error for result in validation_results]
wall_resolution_error = abs(validation_results[1].stopping_distance -
                            validation_results[2].stopping_distance) /
                        analytical_stopping_distance

println("Validation results for 2D rigid-body sliding:")
println("  Analytical stopping distance: $analytical_stopping_distance")
for result in validation_results
    println("  Wall spacing $(result.wall_particle_spacing): " *
            "distance=$(result.stopping_distance), " *
            "relative error=$(result.relative_error), " *
            "final velocity=$(result.final_horizontal_velocity)")
end
println("  Relative wall-resolution difference: $wall_resolution_error")
