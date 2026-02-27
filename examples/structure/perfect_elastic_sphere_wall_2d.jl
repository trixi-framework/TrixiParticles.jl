# ==========================================================================================
# 2D Perfect Elastic Sphere-Wall Collision
#
# Rigid sphere falls under gravity and impacts a flat wall with a frictionless and
# damping-free contact model.
# The script reports impact/rebound consistency after wall separation.
# ==========================================================================================

using LinearAlgebra
using OrdinaryDiffEq
using TrixiParticles

# ==========================================================================================
# ==== Resolution and setup
structure_particle_spacing = 0.01
structure_smoothing_length = 1.5 * structure_particle_spacing
structure_smoothing_kernel = WendlandC2Kernel{2}()

tspan = (0.0, 2.0)
gravity = 9.81

sphere_radius = 0.1
sphere_density = 1200.0
sphere_center = (0.0, 0.25)
sphere_velocity = (0.0, 0.0)

sphere = SphereShape(structure_particle_spacing, sphere_radius, sphere_center,
                     sphere_density;
                     sphere_type=RoundSphere(),
                     velocity=sphere_velocity)

wall_x = collect(-0.4:structure_particle_spacing:0.4)
wall_coordinates = zeros(2, length(wall_x))
wall_coordinates[1, :] .= wall_x
wall_mass = fill(1000.0 * structure_particle_spacing^2, length(wall_x))
wall_density = fill(1000.0, length(wall_x))
wall_ic = InitialCondition(; coordinates=wall_coordinates,
                           mass=wall_mass,
                           density=wall_density,
                           particle_spacing=structure_particle_spacing)

wall_model = BoundaryModelDummyParticles(wall_density, wall_mass,
                                         PressureZeroing(),
                                         structure_smoothing_kernel,
                                         structure_smoothing_length)
wall_system = WallBoundarySystem(wall_ic, wall_model)

# Perfect-elastic reference model:
# - no normal damping
# - no tangential stiffness/damping
# - no friction
# - disable resting-contact projection so rebound is not suppressed
boundary_contact_model = PerfectElasticBoundaryContactModel(; normal_stiffness=2.0e5,
                                                            # Use >= 2h to avoid tunneling and preserve
                                                            # a stable, symmetric rebound at fine
                                                            # resolutions.
                                                            contact_distance=2.0 *
                                                                             structure_particle_spacing,
                                                            torque_free=true,
                                                            stick_velocity_tolerance=1e-8)

rigid_system = RigidSPHSystem(sphere;
                              acceleration=(0.0, -gravity),
                              boundary_contact_model=boundary_contact_model,
                              particle_spacing=structure_particle_spacing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(rigid_system, wall_system)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=50)
saving_callback = SolutionSavingCallback(dt=0.01, output_directory="out", prefix="")
callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

sol = solve(ode, RDPK3SpFSAL35();
            abstol=1e-8,
            reltol=1e-6,
            save_everystep=true,
            callback=callbacks)

# ==========================================================================================
# ==== Diagnostics
function total_kinetic_energy(v_state, rigid_system, semi)
    v_rigid = TrixiParticles.wrap_v(v_state, rigid_system, semi)
    velocity = TrixiParticles.current_velocity(v_rigid, rigid_system)
    kinetic_energy = zero(eltype(rigid_system))

    for particle in eachindex(rigid_system.mass)
        particle_velocity = velocity[:, particle]
        kinetic_energy += 0.5 * rigid_system.mass[particle] *
                          dot(particle_velocity, particle_velocity)
    end

    return kinetic_energy
end

function total_mechanical_energy(v_state, u_state, rigid_system, semi, gravity)
    mechanical_energy = total_kinetic_energy(v_state, rigid_system, semi)
    u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
    coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)

    for particle in eachindex(rigid_system.mass)
        mechanical_energy += rigid_system.mass[particle] * gravity * coordinates[2, particle]
    end

    return mechanical_energy
end

function mean_velocity(v_state, rigid_system, semi)
    v_rigid = TrixiParticles.wrap_v(v_state, rigid_system, semi)
    velocity = TrixiParticles.current_velocity(v_rigid, rigid_system)

    return vec(sum(velocity, dims=2) ./ size(velocity, 2))
end

function mean_vertical_velocity(v_state, rigid_system, semi)
    return mean_velocity(v_state, rigid_system, semi)[2]
end

function center_of_mass(u_state, rigid_system, semi)
    u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
    coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)
    com = zeros(eltype(rigid_system), ndims(rigid_system))
    total_mass = sum(rigid_system.mass)

    for particle in eachindex(rigid_system.mass)
        com .+= rigid_system.mass[particle] .* coordinates[:, particle]
    end
    com ./= total_mass

    return com
end

function minimum_wall_clearance(u_state, rigid_system, semi, wall_top)
    u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
    coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)

    return minimum(coordinates[2, :]) - wall_top
end

initial_kinetic_energy = total_kinetic_energy(sol.u[1].x[1], rigid_system, semi)
vertical_velocity_history = [mean_vertical_velocity(state.x[1], rigid_system, semi)
                             for state in sol.u]
wall_top = maximum(wall_coordinates[2, :])
clearance_history = [minimum_wall_clearance(state.x[2], rigid_system, semi, wall_top)
                     for state in sol.u]

contact_clearance_threshold = 1.2 * boundary_contact_model.contact_distance
impact_start = findfirst(clearance -> clearance <= contact_clearance_threshold,
                         clearance_history)
pre_impact_index = isnothing(impact_start) ? nothing : max(impact_start - 1, 1)
rebound_start = isnothing(impact_start) ? nothing :
                findfirst(i -> i >= impact_start && vertical_velocity_history[i] > 0,
                          eachindex(sol.u))
separation_after_rebound = isnothing(rebound_start) ? nothing :
                           findfirst(i -> i >= rebound_start &&
                                          clearance_history[i] > contact_clearance_threshold,
                                     eachindex(sol.u))

diagnostic_index = something(separation_after_rebound, length(sol.u))
reference_index = something(pre_impact_index, 1)
diagnostic_kinetic_energy = total_kinetic_energy(sol.u[diagnostic_index].x[1],
                                                 rigid_system, semi)
reference_kinetic_energy = total_kinetic_energy(sol.u[reference_index].x[1], rigid_system, semi)
kinetic_energy_relative_error = abs(diagnostic_kinetic_energy - reference_kinetic_energy) /
                                max(abs(reference_kinetic_energy),
                                    sqrt(eps(eltype(rigid_system))))
diagnostic_mechanical_energy = total_mechanical_energy(sol.u[diagnostic_index].x[1],
                                                       sol.u[diagnostic_index].x[2],
                                                       rigid_system, semi, gravity)
reference_mechanical_energy = total_mechanical_energy(sol.u[reference_index].x[1],
                                                      sol.u[reference_index].x[2],
                                                      rigid_system, semi, gravity)
mechanical_energy_relative_error = abs(diagnostic_mechanical_energy -
                                       reference_mechanical_energy) /
                                   max(abs(reference_mechanical_energy),
                                       sqrt(eps(eltype(rigid_system))))

incoming_velocity = mean_velocity(sol.u[reference_index].x[1], rigid_system, semi)
diagnostic_velocity = mean_velocity(sol.u[diagnostic_index].x[1], rigid_system, semi)
incoming_normal_speed = incoming_velocity[2]
incoming_tangential_speed = incoming_velocity[1]
diagnostic_normal_speed = diagnostic_velocity[2]
diagnostic_tangential_speed = diagnostic_velocity[1]
normal_speed_relative_error = abs(abs(diagnostic_normal_speed) - abs(incoming_normal_speed)) /
                              max(abs(incoming_normal_speed), sqrt(eps(eltype(rigid_system))))
tangential_speed_tolerance = max(1e-6,
                                 1e-2 * max(abs(diagnostic_normal_speed),
                                            abs(incoming_normal_speed)))
tangential_to_normal_ratio = abs(diagnostic_tangential_speed) /
                             max(abs(diagnostic_normal_speed),
                                 sqrt(eps(eltype(rigid_system))))
velocity_normality_passed = !isnothing(pre_impact_index) &&
                            !isnothing(separation_after_rebound) &&
                            diagnostic_normal_speed > 0 &&
                            abs(diagnostic_tangential_speed) <=
                            tangential_speed_tolerance &&
                            abs(diagnostic_tangential_speed -
                                incoming_tangential_speed) <=
                            tangential_speed_tolerance &&
                            tangential_to_normal_ratio < 1e-2 &&
                            normal_speed_relative_error < 1e-2

energy_check_passed = !isnothing(pre_impact_index) &&
                      !isnothing(rebound_start) &&
                      !isnothing(separation_after_rebound) &&
                      mechanical_energy_relative_error < 1e-3
com_history = [center_of_mass(state.x[2], rigid_system, semi) for state in sol.u]
com_reference = com_history[reference_index]
com_diagnostic = com_history[diagnostic_index]
com_x_drift = com_diagnostic[1] - com_reference[1]
max_abs_com_x_drift = maximum(abs(com[1] - com_reference[1]) for com in com_history)
com_x_tolerance = max(1e-6, 1e-2 * structure_particle_spacing)
com_x_check_passed = abs(com_x_drift) <= com_x_tolerance &&
                     max_abs_com_x_drift <= com_x_tolerance

overall_check_passed = energy_check_passed &&
                       velocity_normality_passed &&
                       com_x_check_passed

println("Perfect-elastic sphere-wall diagnostics:")
println("  retcode: ", sol.retcode)
println("  initial_kinetic_energy: ", initial_kinetic_energy)
println("  reference_index: ", reference_index)
println("  diagnostic_index: ", diagnostic_index)
println("  impact_detected: ", !isnothing(impact_start))
println("  diagnostic_kinetic_energy: ", diagnostic_kinetic_energy)
println("  reference_kinetic_energy: ", reference_kinetic_energy)
println("  kinetic_energy_relative_error: ", kinetic_energy_relative_error)
println("  diagnostic_mechanical_energy: ", diagnostic_mechanical_energy)
println("  reference_mechanical_energy: ", reference_mechanical_energy)
println("  mechanical_energy_relative_error: ", mechanical_energy_relative_error)
println("  incoming_velocity: ", incoming_velocity)
println("  diagnostic_velocity: ", diagnostic_velocity)
println("  normal_speed_relative_error: ", normal_speed_relative_error)
println("  tangential_speed_tolerance: ", tangential_speed_tolerance)
println("  tangential_to_normal_ratio: ", tangential_to_normal_ratio)
println("  rebound_detected: ", !isnothing(rebound_start))
println("  separated_after_rebound: ", !isnothing(separation_after_rebound))
println("  energy_check_passed: ", energy_check_passed)
println("  velocity_normality_passed: ", velocity_normality_passed)
println("  com_reference: ", com_reference)
println("  com_diagnostic: ", com_diagnostic)
println("  com_x_drift: ", com_x_drift)
println("  max_abs_com_x_drift: ", max_abs_com_x_drift)
println("  com_x_tolerance: ", com_x_tolerance)
println("  com_x_check_passed: ", com_x_check_passed)
println("  overall_check_passed: ", overall_check_passed)

if !overall_check_passed
    println("WARNING: Elastic-collision checks did not pass. Increase `tspan` or adjust contact resolution/parameters.")
end
