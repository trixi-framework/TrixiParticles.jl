include("../validation_util.jl")

############################################################################################
# Case "Elastic plate under a hydrostatic water column" as described in
# "A fluid–structure interaction model for free-surface flows and flexible structures
# using smoothed particle hydrodynamics on a GPU" by J. O'Connor and B.D. Rogers
# published in Journal of Fluids and Structures
# https://doi.org/10.1016/j.jfluidstructs.2021.103312
############################################################################################

using TrixiParticles
using OrdinaryDiffEq
using JSON

n_particles_plate_y = 3
tspan=(0.0, 0.3)

# ============================================================================
# Sensor Function (for Postprocessing)
# ============================================================================
function y_deflection(system::TotalLagrangianSPHSystem, v, u, semi, t)
    # Choose the particle in the middle of the beam along x.
    particle_id = Int(n_particles_per_dimension[1] * (n_particles_plate_y + 1) / 2 -
                      (n_particles_per_dimension[1] + 1) / 2 + 1)
    return TrixiParticles.current_coords(u, system, particle_id)[2] + plate_size[2] / 2
end
y_deflection(system, v, u, semi, t) = nothing

# ============================================================================
# Run Simulations and Compute Errors
# ============================================================================
errors = Dict{String, Tuple{Float64, Float64}}()

for method in ["edac", "wcsph"]
    pp_filename = "validation_result_hyd_" * method * "_" * string(n_particles_plate_y)
    pp = PostprocessCallback(; dt=0.0025, filename=pp_filename, y_deflection,
                             kinetic_energy)

   trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fsi", "hydrostatic_water_column_2d.jl"),
              use_edac=(method == "edac" ? true : false),
              n_particles_plate_y=n_particles_plate_y,
              tspan=(0.0, 0.3),
              update_strategy=SerialUpdate(),
              dt=0.5,
              tspan=tspan,
              prefix=pp_filename,
              extra_callback=pp,
              sol=nothing)

    sol = solve(ode, RDPK3SpFSAL49(), dt=1e-8, reltol=1e-5, abstol=1e-7, maxiters=1e6,
                save_everystep=false, callback=callbacks)

    # Load the run JSON file and add the analytical solution as a single point
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

    # Compute errors using data from t ∈ [0.25, 0.5] for the sensor starting with "y_deflection"
    sensor_key = first(filter(k -> startswith(k, "y_deflection"), keys(run_data)))
    time_vals = run_data[sensor_key]["time"]
    sim_vals = run_data[sensor_key]["values"]
    inds = findall(t -> 0.25 <= t <= 0.5, time_vals)
    avg_sim = sum(sim_vals[inds]) / length(sim_vals[inds])
    abs_error = abs(avg_sim - analytical_value)
    rel_error = abs_error / abs(analytical_value)
    errors[method] = (abs_error, rel_error)
end

println("Errors for hydrostatic water column 2D:", errors)
