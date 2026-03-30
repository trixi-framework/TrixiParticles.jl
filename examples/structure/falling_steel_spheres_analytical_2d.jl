# ==========================================================================================
# 2D Falling Steel Spheres vs Analytical Bounce
#
# This structure-only example drops several identical steel spheres onto a flat wall.
# It compares each sphere's center-of-mass trajectory against a piecewise analytical
# bounce solution using the same restitution coefficient as the steel contact model.
# The setup is intended to diagnose wall-phase-dependent drift or rebound variation.
# ==========================================================================================

using OrdinaryDiffEq
using TrixiParticles

# ==========================================================================================
# ==== Resolution and setup
# Use a coarse particle layout so this investigation example remains quick to run
# while still exercising the steel wall-contact model.
structure_particle_spacing = 0.02
structure_smoothing_length = 1.5 * structure_particle_spacing
structure_smoothing_kernel = WendlandC2Kernel{2}()

gravity = 9.81
tspan = (0.0, 0.45)

sphere_radius = 0.04
initial_center_y = 0.26
sphere_centers_x = (-0.12, 0.0, 0.12)

wall_material = (; youngs_modulus=3.0e11, poisson_ratio=0.2)
steel_material = (; density=7850.0, youngs_modulus=2.1e11, poisson_ratio=0.29,
                  restitution=0.8, friction_coefficient=0.55)

wall_x = collect(-0.22:structure_particle_spacing:0.22)
boundary_layers = 3
n_wall_particles = boundary_layers * length(wall_x)
wall_coordinates = zeros(2, n_wall_particles)
wall_mass = fill(1000.0 * structure_particle_spacing^2, n_wall_particles)
wall_density = fill(1000.0, n_wall_particles)

let index = 1
    for layer in 0:(boundary_layers - 1), x in wall_x
        wall_coordinates[1, index] = x
        wall_coordinates[2, index] = -layer * structure_particle_spacing
        index += 1
    end
end

wall_ic = InitialCondition(; coordinates=wall_coordinates,
                           mass=wall_mass,
                           density=wall_density,
                           particle_spacing=structure_particle_spacing)

wall_model = BoundaryModelDummyParticles(wall_density, wall_mass,
                                         PressureZeroing(),
                                         structure_smoothing_kernel,
                                         structure_smoothing_length)
wall_system = WallBoundarySystem(wall_ic, wall_model)

function make_steel_sphere(center_x)
    center = (center_x, initial_center_y)
    shape = SphereShape(structure_particle_spacing, sphere_radius, center,
                        steel_material.density, sphere_type=RoundSphere())
    contact_model_spec = LinearizedHertzMindlinBoundaryContactModel(;
                                                                    material=steel_material,
                                                                    wall_material,
                                                                    radius=sphere_radius,
                                                                    center,
                                                                    gravity,
                                                                    particle_spacing=structure_particle_spacing,
                                                                    ndims=2,
                                                                    torque_free=true,
                                                                    resting_contact_projection=true)

    rigid_system = RigidBodySystem(shape;
                                   acceleration=(0.0, -gravity),
                                   boundary_contact_model=contact_model_spec,
                                   particle_spacing=structure_particle_spacing)

    return (; center, rigid_system)
end

sphere_cases = map(make_steel_sphere, sphere_centers_x)
rigid_systems = map(case -> case.rigid_system, sphere_cases)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(wall_system, rigid_systems...)
ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=10)
saving_callback = SolutionSavingCallback(dt=0.02, output_directory="out", prefix="")
callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback())

sol = solve(ode, RDPK3SpFSAL49();
            abstol=1e-7,
            reltol=5e-4,
            dt=1e-4,
            qmax=1.1,
            saveat=0.001,
            save_everystep=false,
            callback=callbacks)

# ==========================================================================================
# ==== Diagnostics
function center_of_mass(u_state, rigid_system, semi)
    u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
    coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)
    center = zeros(eltype(rigid_system), ndims(rigid_system))

    for particle in eachindex(rigid_system.mass)
        center .+= rigid_system.mass[particle] .* coordinates[:, particle]
    end

    center ./= rigid_system.total_mass

    return center
end

function center_of_mass_velocity(v_state, rigid_system, semi)
    v_rigid = TrixiParticles.wrap_v(v_state, rigid_system, semi)
    velocity = TrixiParticles.current_velocity(v_rigid, rigid_system)
    center_velocity = zeros(eltype(rigid_system), ndims(rigid_system))

    for particle in eachindex(rigid_system.mass)
        center_velocity .+= rigid_system.mass[particle] .* velocity[:, particle]
    end

    center_velocity ./= rigid_system.total_mass

    return center_velocity
end

function minimum_y(u_state, rigid_system, semi)
    u_rigid = TrixiParticles.wrap_u(u_state, rigid_system, semi)
    coordinates = TrixiParticles.current_coordinates(u_rigid, rigid_system)

    return minimum(coordinates[2, :])
end

function analytical_bounce_positions(times, initial_y, contact_y, gravity, restitution)
    analytical_y = similar(times)
    initial_drop_height = max(initial_y - contact_y, zero(eltype(times)))

    if initial_drop_height <= eps(eltype(times))
        analytical_y .= contact_y
        return analytical_y
    end

    impact_speed = sqrt(2 * gravity * initial_drop_height)
    first_impact_time = impact_speed / gravity

    for i in eachindex(times)
        t = times[i]

        if t <= first_impact_time
            analytical_y[i] = initial_y - 0.5 * gravity * t^2
            continue
        end

        time_since_impact = t - first_impact_time
        rebound_speed = restitution * impact_speed

        while rebound_speed > sqrt(eps(eltype(times)))
            flight_time = 2 * rebound_speed / gravity

            if time_since_impact <= flight_time
                analytical_y[i] = contact_y + rebound_speed * time_since_impact -
                                  0.5 * gravity * time_since_impact^2
                break
            end

            time_since_impact -= flight_time
            rebound_speed *= restitution
        end

        if rebound_speed <= sqrt(eps(eltype(times)))
            analytical_y[i] = contact_y
        end
    end

    return analytical_y
end

function first_rebound_height(center_y_history, impact_index)
    isnothing(impact_index) && return NaN

    return maximum(center_y_history[impact_index:end])
end

wall_top = maximum(wall_coordinates[2, :])
runtime_contact_model = first(rigid_systems).contact_model
contact_distance = runtime_contact_model.contact_distance
geometric_contact_height = wall_top + sphere_radius
effective_contact_height = geometric_contact_height + contact_distance

all_diagnostics = map(enumerate(sphere_cases)) do (sphere_index, case)
    rigid_system = case.rigid_system
    center_x_history = similar(sol.t)
    center_y_history = similar(sol.t)
    minimum_y_history = similar(sol.t)
    velocity_y_history = similar(sol.t)

    for i in eachindex(sol.t)
        v_state, u_state = sol.u[i].x
        center = center_of_mass(u_state, rigid_system, semi)
        center_velocity = center_of_mass_velocity(v_state, rigid_system, semi)
        center_x_history[i] = center[1]
        center_y_history[i] = center[2]
        minimum_y_history[i] = minimum_y(u_state, rigid_system, semi)
        velocity_y_history[i] = center_velocity[2]
    end

    analytical_y = analytical_bounce_positions(sol.t, case.center[2],
                                               effective_contact_height,
                                               gravity,
                                               steel_material.restitution)
    analytical_x = fill(case.center[1], length(sol.t))
    y_error = center_y_history .- analytical_y
    x_error = center_x_history .- analytical_x

    contact_threshold = wall_top + 1.1 * contact_distance
    free_flight_indices = findall(y -> y > contact_threshold, minimum_y_history)
    free_flight_rms_y_error = isempty(free_flight_indices) ? NaN :
                              sqrt(sum(y_error[i]^2 for i in free_flight_indices) /
                                   length(free_flight_indices))

    impact_index = findfirst(y -> y <= wall_top + contact_distance, minimum_y_history)
    rebound_height_numerical = first_rebound_height(center_y_history, impact_index)
    rebound_height_analytical = effective_contact_height +
                                steel_material.restitution^2 *
                                max(case.center[2] - effective_contact_height, 0.0)
    rebound_height_ratio = rebound_height_numerical / rebound_height_analytical
    separated_after_impact = !isnothing(impact_index) &&
                             any(y -> y > wall_top + contact_distance +
                                      0.25 * structure_particle_spacing,
                                 minimum_y_history[(impact_index + 1):end])

    return (; sphere_index,
            initial_center=case.center,
            max_abs_x_error=maximum(abs, x_error),
            max_abs_y_error=maximum(abs, y_error),
            free_flight_rms_y_error,
            rebound_height_numerical,
            rebound_height_analytical,
            rebound_height_error=rebound_height_numerical - rebound_height_analytical,
            rebound_height_ratio,
            minimum_center_y=minimum(center_y_history),
            minimum_particle_y=minimum(minimum_y_history),
            impact_index,
            separated_after_impact,
            peak_upward_velocity=maximum(max(vy, 0.0) for vy in velocity_y_history))
end

rebound_height_spread = maximum(d.rebound_height_numerical for d in all_diagnostics) -
                        minimum(d.rebound_height_numerical for d in all_diagnostics)
max_horizontal_drift = maximum(d.max_abs_x_error for d in all_diagnostics)
max_free_flight_rms_y_error = maximum(d.free_flight_rms_y_error for d in all_diagnostics)
max_rebound_height_relative_error = maximum(abs(1 - d.rebound_height_ratio)
                                            for d in all_diagnostics)

println("Steel spheres analytical comparison:")
println("  retcode: ", sol.retcode)
println("  n_spheres: ", length(rigid_systems))
println("  steel_material: ", steel_material)
println("  runtime_contact_model: ", runtime_contact_model)
println("  wall_top: ", wall_top)
println("  geometric_contact_height: ", geometric_contact_height)
println("  effective_contact_height: ", effective_contact_height)
println("  contact_distance: ", contact_distance)
println("  rebound_height_spread: ", rebound_height_spread)
println("  max_rebound_height_relative_error: ", max_rebound_height_relative_error)
println("  max_horizontal_drift: ", max_horizontal_drift)
println("  max_free_flight_rms_y_error: ", max_free_flight_rms_y_error)

for diagnostic in all_diagnostics
    println("  sphere_", diagnostic.sphere_index, ":")
    println("    initial_center: ", diagnostic.initial_center)
    println("    max_abs_x_error: ", diagnostic.max_abs_x_error)
    println("    max_abs_y_error: ", diagnostic.max_abs_y_error)
    println("    free_flight_rms_y_error: ", diagnostic.free_flight_rms_y_error)
    println("    rebound_height_numerical: ", diagnostic.rebound_height_numerical)
    println("    rebound_height_analytical: ", diagnostic.rebound_height_analytical)
    println("    rebound_height_error: ", diagnostic.rebound_height_error)
    println("    rebound_height_ratio: ", diagnostic.rebound_height_ratio)
    println("    minimum_center_y: ", diagnostic.minimum_center_y)
    println("    minimum_particle_y: ", diagnostic.minimum_particle_y)
    println("    impact_detected: ", !isnothing(diagnostic.impact_index))
    println("    separated_after_impact: ", diagnostic.separated_after_impact)
    println("    peak_upward_velocity: ", diagnostic.peak_upward_velocity)
end

if max_horizontal_drift > 5 * structure_particle_spacing ||
   rebound_height_spread > 0.05 ||
   max_free_flight_rms_y_error > 0.02 ||
   max_rebound_height_relative_error > 0.1
    println("WARNING: Steel-sphere motion deviates noticeably from the analytical reference.")
end
