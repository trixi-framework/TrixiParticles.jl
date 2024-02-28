include("../validation_util.jl")

using TrixiParticles
using JSON

trixi_include(@__MODULE__,
              joinpath(validation_dir(), "oscillating_beam_2d",
                       "validation_oscillating_beam_2d.jl"),
              resolution=[5], tspan=(0, 2))

reference_file_name = joinpath(validation_dir(),
                               "oscillating_beam_2d/validation_reference_5.json")
run_file_name = joinpath(pkgdir(TrixiParticles),
                         "out/validation_run_oscillating_beam_2d_5.json")

reference_data = JSON.parsefile(reference_file_name)
run_data = JSON.parsefile(run_file_name)

error = calculate_error(reference_data, run_data)
error
