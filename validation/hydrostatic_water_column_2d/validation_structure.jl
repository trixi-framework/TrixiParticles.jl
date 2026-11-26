using TrixiParticles
using CSV
using DataFrames
using JSON

include("structure_util.jl")

resolutions = (3, 5, 11, 21, 41)
output_directory = "out_hydrostatic_structure"
verify_rhs = true

mkpath(output_directory)
results = [HydrostaticPlateValidation.static_plate(resolution; verify_rhs)
           for resolution in resolutions]
csv_file = joinpath(output_directory, "hydrostatic_structure.csv")
CSV.write(csv_file, DataFrame(results))
parameters = (; resolutions, verify_rhs, plate_size=(1.0, 0.05), structure_density=2700.0,
              fluid_density=1000.0, water_height=2.0, gravity=9.81, young_modulus=67.5e9,
              poisson_ratio=0.3, smoothing_length_ratio=sqrt(2), clamped_columns=3,
              place_on_shell=true, load="top traction + material gravity",
              model="stress-free linearized TLSPH, prescribed load, no fluid or time integration")
open(joinpath(output_directory, "parameters.json"), "w") do io
    JSON.print(io, parameters, 2)
end
println("Wrote $csv_file ($(length(results)) isolated-plate controls)")
