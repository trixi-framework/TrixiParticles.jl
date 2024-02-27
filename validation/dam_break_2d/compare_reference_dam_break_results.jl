include("../validation_util.jl")

using TrixiParticles
using JSON

H = 0.6

trixi_include(@__MODULE__,
              joinpath(validation_dir(), "dam_break_2d",
                       "validation_dam_break_2d.jl"), resolutions=[H / 40])

reference_file_edac_name = "validation/dam_break_2d/validation_reference_dam_break_edac_0015.json"
run_file_edac_name = "out/validation_result_dam_break_edac_0015.json"

reference_data = JSON.parsefile(reference_file_edac_name)
run_data = JSON.parsefile(run_file_edac_name)

error_edac = calculate_error(reference_data, run_data)

if error_edac > 7e-19
    error("The test failed for EDAC with an error of:", error_edac)
    return -1
end

reference_file_wcsph_name = "validation/dam_break_2d/validation_reference_dam_break_wcsph_0015.json"
run_file_wcsph_name = "out/validation_result_dam_break_wcsph_0015.json"

reference_data = JSON.parsefile(reference_file_wcsph_name)
run_data = JSON.parsefile(run_file_wcsph_name)

error_wcsph = calculate_error(reference_data, run_data)

if error_wcsph > 7e-19
    error("The test failed for WCSPH with an error of:", error_wcsph)
    return -1
end
