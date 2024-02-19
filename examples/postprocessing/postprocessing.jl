using TrixiParticles
using Plots
using CSV
using DataFrames
using JSON

# Any custom function with the arguments `v, u, t, system` can be passed to the callback
# to be called every 10th timestep. See example below:
function hello(v, u, t, system)
    # Will write "hello" and the current simulation time
    println("hello at ", t)

    # Value stored for output in the postprocessing output file
    return 2 * t
end
example_cb = PostprocessCallback(; interval=10, hello)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=example_cb, tspan=(0.0, 0.1));

# Lets write the average pressure and kinetic energy every 0.01s
pp = PostprocessCallback(; dt=0.005, filename="example_pressure_ekin", avg_pressure,
                         kinetic_energy)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=pp, tspan=(0.0, 0.1));

data = CSV.read("out/example_pressure_ekin.csv", DataFrame)

# Or, alternatively using JSON (data is not used in the plot)
file_content = read("out/example_pressure_ekin.json", String)
data_json = JSON.parse(file_content)
time = data_json["kinetic_energy_fluid_1"]["time"]
values_ekin = data_json["kinetic_energy_fluid_1"]["values"]
values_avg_p = data_json["avg_pressure_fluid_1"]["values"]

# Create side-by-side subplots
p1 = plot(data.time, data.kinetic_energy_fluid_1, color=:blue,
          title="Kinetic Energy", xlabel="Time", ylabel="Kinetic Energy")
p2 = plot(data.time, data.avg_pressure_fluid_1, color=:red,
          title="Average Pressure", xlabel="Time", ylabel="Pressure")

# Combine plots into a single figure
plot(p1, p2, legend=false)
