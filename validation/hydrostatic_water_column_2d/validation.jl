include("../validation_util.jl")

# ==========================================================================================
# 2D Hydrostatic Water Column on an Elastic Plate Validation
#
# Case "Elastic plate under a hydrostatic water column" as described in
# "A fluid–structure interaction model for free-surface flows and flexible structures
# using smoothed particle hydrodynamics on a GPU" by J. O'Connor and B.D. Rogers
# published in Journal of Fluids and Structures
# https://doi.org/10.1016/j.jfluidstructs.2021.103312
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq
using JSON

# ==========================================================================================
# ==== Resolution
# Reference data is available for n_particles_plate_y = 3, 5, 11, and 13.
# Computational time on a single core:
# n_particles_plate_y = 3: ~0.25h
# n_particles_plate_y = 5: ~1.5h
# n_particles_plate_y = 11: ~10h
# n_particles_plate_y = 13: ~15h

n_particles_plate_y = 5

# ==========================================================================================
# ==== Experiment Setup
# For better results this should be increased to at least 0.5.
tspan = (0.0, 0.5)

# ==========================================================================================
# ==== Sensor (Postprocessing)
function y_deflection(system::TotalLagrangianSPHSystem, dv_ode, du_ode, v_ode, u_ode, semi,
                      t)
    # Choose the particle in the middle of the beam along x.
    particle_id = Int(n_particles_per_dimension[1] * (n_particles_plate_y + 1) / 2 -
                      (n_particles_per_dimension[1] + 1) / 2 + 1)
    return TrixiParticles.current_coords(u_ode, system, particle_id)[2] + plate_size[2] / 2
end
y_deflection(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing

# ==========================================================================================
# ==== Run Simulations and Compute Errors
function run_simulation(method; n_particles_plate_y, tspan)
    method_label = String(method.name)
    pp_filename = "validation_result_hyd_" * method_label * "_" *
                  string(n_particles_plate_y)
    pp = PostprocessCallback(; dt=0.0025, filename=pp_filename, y_deflection,
                             kinetic_energy)

    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "fsi", "hydrostatic_water_column_2d.jl"),
                  use_edac=method.use_edac,
                  n_particles_plate_y=n_particles_plate_y,
                  update_strategy=SerialUpdate(),
                  tspan=tspan,
                  prefix=pp_filename,
                  extra_callback=pp)

    return pp_filename, sol
end

methods = ((name=:edac, use_edac=true),
           (name=:wcsph, use_edac=false))
errors = Dict{Symbol, Tuple{Float64, Float64}}()

for method in methods
    pp_filename,
    _ = run_simulation(method;
                       n_particles_plate_y=n_particles_plate_y,
                       tspan=tspan)

    # Load the run JSON file and add the analytical solution as a single point.
    run_filename = joinpath("out", pp_filename * ".json")
    run_data = nothing
    open(run_filename, "r") do io
        run_data = JSON.parse(io)
    end

    run_data["analytical_solution"] = Dict(
        "n_values" => 1,
        "time" => [tspan[2]],
        "values" => [analytical_value],
        "datatype" => "Float64",
        "type" => "point"
    )

    open(run_filename, "w") do io
        JSON.print(io, run_data, 2)
    end

    # Compute errors using the average over t in [0.25, 0.5]
    # for the sensor starting with "y_deflection".
    sensor_key = first(filter(k -> startswith(k, "y_deflection"), keys(run_data)))
    time_vals = run_data[sensor_key]["time"]
    sim_vals = run_data[sensor_key]["values"]
    inds = findall(t -> 0.25 <= t <= 0.5, time_vals)
    avg_sim = sum(sim_vals[inds]) / length(sim_vals[inds])
    abs_error = abs(avg_sim - analytical_value)
    rel_error = abs_error / abs(analytical_value)
    errors[method.name] = (abs_error, rel_error)
end

println("Errors for hydrostatic water column 2D:", errors)
