include("../validation_util.jl")

using TrixiParticles
using JSON

trixi_include(@__MODULE__,
              joinpath(validation_dir(), "oscillating_beam_2d",
                       "validation_oscillating_beam_2d.jl"),
              resolution=[9])

reference_file_name = "validation/oscillating_beam_2d/validation_reference_oscillating_beam_2d_9.json"
run_file_name = "out/validation_run_oscillating_beam_2d_9.json"

reference_data = JSON.parsefile(reference_file_name)
run_data = JSON.parsefile(run_file_name)

error = calculate_error(reference_data, run_data)
