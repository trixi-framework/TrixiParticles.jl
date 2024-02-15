using TrixiParticles
using Plots
using CSV
using DataFrames
using JSON

# Any function can be implemented and will be called every 10th timestep! See example below:
function hello(v, u, t, system)
    # will write "hello" and the current simulation time
    println("hello at ", t)

    # value stored for output in the postprocessing output file
    return 2 * t
end
example_cb = PostprocessCallback(hello; interval=10)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=example_cb, tspan=(0.0, 0.1));

# Lets write the average pressure and kinetic energy every 0.01s
pp = PostprocessCallback(avg_pressure, kinetic_energy; dt=0.005, filename="example_pressure_ekin")

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              extra_callback=pp, tspan=(0.0, 0.1));

data = CSV.read("out/example_pressure_ekin.csv", DataFrame)

# or alternatively using JSON (data is not used in the plot)
file_content = read("out/example_pressure_ekin.json", String)
data_json = JSON.parse(file_content)
time = data_json["kinetic_energy_fluid_1"]["time"]
values_ekin = data_json["kinetic_energy_fluid_1"]["values"]
values_avg_p = data_json["avg_pressure_fluid_1"]["values"]

# Create side-by-side subplots
p1 = plot(data.time, data.kinetic_energy_fluid_1, label="kinetic energy", color=:blue,
          title="Kin. Energy over Time", xlabel="Time", ylabel="Kin. E")
p2 = plot(data.time, data.avg_pressure_fluid_1, label="average pressure", color=:red,
          title="avg P over Time", xlabel="Time", ylabel="Pressure")

# Combine plots into a single figure
plot(p1, p2, layout=(1, 2), legend=false)
