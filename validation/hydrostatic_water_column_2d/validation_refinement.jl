using TrixiParticles

include("refinement_util.jl")

resolutions = (3, 5, 9, 17)
methods = (:wcsph, :edac)
density_modes = (:continuity, :summation)
workers = 12
execute = true
reuse_directories = ()
output_root = "out_hydrostatic_refinement"
operator_file = isfile(joinpath("out", "correction_convergence.csv")) ?
                abspath(joinpath("out", "correction_convergence.csv")) : nothing
structure_file = isfile(joinpath("out_hydrostatic_structure", "hydrostatic_structure.csv")) ?
                 abspath(joinpath("out_hydrostatic_structure", "hydrostatic_structure.csv")) :
                 nothing
report_file = joinpath(output_root, "correction_validation.html")
completion_command = String[]

study = HydrostaticRefinement.run_refinement(; resolutions, methods, density_modes, workers,
                                             reuse_directories, output_root, operator_file,
                                             structure_file, report_file,
                                             completion_command, execute)
println("Four-level refinement study: $(study["status"])")
