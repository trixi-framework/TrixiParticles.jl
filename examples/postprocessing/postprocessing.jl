# ==========================================================================================
# Postprocessing Callback Example
#
# This example demonstrates how to use the `PostprocessCallback` in TrixiParticles.jl
# to compute and save custom quantities during a simulation, or to execute
# arbitrary user-defined functions at specified intervals.
# A hydrostatic water column simulation is used as the base.
# ==========================================================================================

using TrixiParticles
using Plots    # For plotting results from CSV/JSON
using CSV      # For reading CSV files
using DataFrames # For handling data from CSV
using JSON     # For reading JSON files

# ------------------------------------------------------------------------------
# Part 1: Custom User-Defined Function with PostprocessCallback
# ------------------------------------------------------------------------------
# Define a custom function to be called by the `PostprocessCallback`.
# Arguments: `system` (the SPH system, e.g., fluid_system),
#            `v_ode`, `u_ode` (current state vectors from OrdinaryDiffEq.jl),
#            `semi` (the semidiscretization object),
#            `t` (current simulation time).
# The return value of this function can be recorded by the callback if its name
# (as a Symbol) is passed to `PostprocessCallback`.
function my_custom_hello_function(system, v_ode, u_ode, semi, t)
    println("Executing custom_hello_function at t = $t")
    # This function could perform complex calculations or I/O operations.
    # The value returned here (e.g., `2 * t`) will be stored by the callback
    # if `PostprocessCallback` is initialized with `my_custom_hello_function` as a tracked quantity.
    return 2 * t # Example: return twice the current time
end

# Create a PostprocessCallback to execute `my_custom_hello_function`.
# `interval=10` means the function is called every 10 simulation steps.
# `write_file_interval=0` disables writing of a summary CSV/JSON by this specific callback instance.
# If we wanted to save the return value of `my_custom_hello_function`, we'd pass
# `user_defined_outputs=(my_custom_hello_function_symbol=my_custom_hello_function,)`
# or simply `my_custom_hello_function` if the name itself is the symbol.
# Here, we use `user_defined_callback` to just execute it.
custom_function_cb = PostprocessCallback(; interval=10, write_file_interval=0,
                                         user_defined_callback=my_custom_hello_function)

println("Running simulation with custom_hello_function callback...")
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=custom_function_cb, tspan=(0.0, 0.1)) # Short simulation

# ------------------------------------------------------------------------------
# Part 2: Using Built-in Quantities with PostprocessCallback
# ------------------------------------------------------------------------------
# Create a PostprocessCallback to compute and save average pressure and kinetic energy.
# `dt=0.005` means these quantities are computed approximately every 0.005 simulation time units.
# `filename="out/hydrostatic_quantities"` specifies the base name for output files (CSV/JSON).
# `avg_pressure` and `kinetic_energy` are built-in functions in TrixiParticles.
# `write_file_interval=1` (default) enables writing to CSV/JSON. If set to 0, data is computed but not written.
quantities_cb = PostprocessCallback(; dt=0.005, filename="out/hydrostatic_quantities",
                                    avg_pressure, kinetic_energy)
# Output will be `out/hydrostatic_quantities.csv` and `out/hydrostatic_quantities.json`

println("\nRunning simulation to record average pressure and kinetic energy...")
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=quantities_cb, tspan=(0.0, 0.1)) # Short simulation

# ------------------------------------------------------------------------------
# Part 3: Reading and Plotting Saved Data
# ------------------------------------------------------------------------------
# --- Reading from CSV file ---
output_csv_file = "out/hydrostatic_quantities.csv"
println("\nReading data from CSV: $output_csv_file")
if isfile(output_csv_file)
    data_df = CSV.read(output_csv_file, DataFrame)

    # Column names in the CSV typically are: `time`, `avg_pressure_fluid_1`, `kinetic_energy_fluid_1`
    # (assuming the fluid system is the first one and named `fluid_1` internally).
    # Check `names(data_df)` if unsure.
    time_csv = data_df.time
    avg_pressure_csv = data_df.avg_pressure_fluid_1
    kinetic_energy_csv = data_df.kinetic_energy_fluid_1

    # Plotting data from CSV
    plot_pressure_csv = Plots.plot(time_csv, avg_pressure_csv,
                                   label="Avg. Pressure (CSV)", color=:blue,
                                   xlabel="Time (s)", ylabel="Pressure (Pa)",
                                   title="Average Fluid Pressure")
    plot_energy_csv = Plots.plot(time_csv, kinetic_energy_csv,
                                 label="Kinetic Energy (CSV)", color=:red,
                                 xlabel="Time (s)", ylabel="Energy (J)",
                                 title="Total Fluid Kinetic Energy")

    combined_plot_csv = Plots.plot(plot_pressure_csv, plot_energy_csv, layout=(2, 1),
                                   legend=:outertopright, size=(800, 700))
    println("Displaying plots from CSV data...")
    display(combined_plot_csv)
else
    println("Warning: CSV file $output_csv_file not found. Skipping CSV plotting.")
end

# --- Reading from JSON file (alternative) ---
output_json_file = "out/hydrostatic_quantities.json"
println("\nReading data from JSON: $output_json_file")
if isfile(output_json_file)
    file_content_json = read(output_json_file, String)
    data_json = JSON.parse(file_content_json)

    # Accessing data from the JSON structure
    # The structure is typically: Dict{String, Dict{"time" => Vector, "values" => Vector}}
    time_json = data_json["kinetic_energy_fluid_1"]["time"] # Assuming key exists
    values_kinetic_energy_json = data_json["kinetic_energy_fluid_1"]["values"]
    values_avg_pressure_json = data_json["avg_pressure_fluid_1"]["values"]

    # Plotting data from JSON (similar to CSV plotting)
    plot_pressure_json = Plots.plot(time_json, values_avg_pressure_json,
                                    label="Avg. Pressure (JSON)", color=:green,
                                    xlabel="Time (s)", ylabel="Pressure (Pa)",
                                    title="Average Fluid Pressure (from JSON)")
    plot_energy_json = Plots.plot(time_json, values_kinetic_energy_json,
                                  label="Kinetic Energy (JSON)", color=:purple,
                                  xlabel="Time (s)", ylabel="Energy (J)",
                                  title="Total Fluid Kinetic Energy (from JSON)")

    combined_plot_json = Plots.plot(plot_pressure_json, plot_energy_json, layout=(2, 1),
                                    legend=:outertopright, size=(800, 700))
    println("Displaying plots from JSON data...")
    display(combined_plot_json)
else
    println("Warning: JSON file $output_json_file not found. Skipping JSON plotting.")
end

println("\nPostprocessing callback example finished.")
