# ==========================================================================================
# 2D Rigid-Body Sliding Validation
#
# A rigid square slides on a horizontal wall under Coulomb friction. While it is slipping,
# its center-of-mass acceleration is -mu_k * g, so its analytical stopping distance is
# v_0^2 / (2 * mu_k * g). Running the example at two wall resolutions also verifies that
# geometry-normal wall contact is independent of tangential wall-particle sampling.
# ==========================================================================================

using TrixiParticles
using TrixiParticles.JSON
using OrdinaryDiffEqLowStorageRK

# ==========================================================================================
# ==== Resolution and Experiment Setup
particle_spacing = 0.03
wall_particle_spacings = (particle_spacing, particle_spacing / 2)
tspan = (0.0, 0.3)
gravity = 9.81
initial_velocity = 1.0
kinetic_friction_coefficient = 0.4
analytical_stopping_time = initial_velocity / (kinetic_friction_coefficient * gravity)
analytical_stopping_distance = initial_velocity^2 /
                               (2 * kinetic_friction_coefficient * gravity)

# ==========================================================================================
# ==== Sensors
# Only the frictional rigid body contributes data. The same callback is applied to every
# system in the example, so all other systems return `nothing`.
function frictional_center_of_mass_x(system::RigidBodySystem, data, t)
    contact_model = system.contact_model
    (isnothing(contact_model) || iszero(contact_model.kinetic_friction_coefficient)) &&
        return nothing
    return data.center_of_mass[1]
end

function frictional_horizontal_velocity(system::RigidBodySystem, data, t)
    contact_model = system.contact_model
    (isnothing(contact_model) || iszero(contact_model.kinetic_friction_coefficient)) &&
        return nothing
    return data.center_of_mass_velocity[1]
end

frictional_center_of_mass_x(system, data, t) = nothing
frictional_horizontal_velocity(system, data, t) = nothing

# ==========================================================================================
# ==== Run Simulations and Compute Errors
function run_rigid_body_sliding_validation(wall_particle_spacing; tspan)
    spacing_millimeters = round(Int, 1000 * wall_particle_spacing)
    filename = "validation_result_rigid_body_sliding_2d_wall_spacing_$(spacing_millimeters)"
    output_directory = "out"

    postprocess_callback = PostprocessCallback(; dt=0.005, output_directory, filename,
                                               write_csv=true, write_file_interval=0,
                                               frictional_center_of_mass_x,
                                               frictional_horizontal_velocity)

    # Reuse the public example setup, but replace its solve so validation can install a
    # postprocess callback and vary only the wall resolution.
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "structure",
                           "sliding_rigid_squares_friction_2d.jl");
                  wall_particle_spacing, tspan, sol=nothing)

    callbacks = CallbackSet(UpdateCallback(), StepsizeCallback(cfl=0.5),
                            postprocess_callback)
    sol = solve((@__MODULE__).ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-4,
                save_everystep=false, callback=callbacks)

    run_filename = joinpath(output_directory, filename * ".json")
    run_data = JSON.parsefile(run_filename)
    position_key = only(filter(key -> startswith(key, "frictional_center_of_mass_x_"),
                               keys(run_data)))
    velocity_key = only(filter(key -> startswith(key, "frictional_horizontal_velocity_"),
                               keys(run_data)))

    time_values = Float64.(run_data[position_key]["time"])
    position_values = Float64.(run_data[position_key]["values"])
    velocity_values = Float64.(run_data[velocity_key]["values"])
    stopping_distance = last(position_values) - first(position_values)
    relative_error = abs(stopping_distance - analytical_stopping_distance) /
                     analytical_stopping_distance

    # Store the analytical trajectory beside the numerical series so the plotting script does
    # not need to reconstruct simulation-specific initial coordinates.
    initial_position = first(position_values)
    analytical_positions = map(time_values) do time
        elapsed_time = time - first(time_values)
        displacement = elapsed_time < analytical_stopping_time ?
                       initial_velocity * elapsed_time -
                       0.5 * kinetic_friction_coefficient * gravity * elapsed_time^2 :
                       analytical_stopping_distance
        return initial_position + displacement
    end
    run_data["analytical_center_of_mass_x"] = Dict(
        "type" => "series",
        "datatype" => "Float64",
        "n_values" => length(analytical_positions),
        "system_name" => "analytical",
        "values" => analytical_positions,
        "time" => time_values
    )
    open(run_filename, "w") do io
        JSON.print(io, run_data, 4)
    end

    return (; wall_particle_spacing, sol, run_filename, stopping_distance, relative_error,
            final_horizontal_velocity=last(velocity_values))
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
