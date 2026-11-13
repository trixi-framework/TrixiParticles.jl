using TrixiParticles

include("refinement_util.jl")

length(ARGS) <= 2 ||
    throw(ArgumentError("usage: publish_refinement.jl [study_root] [diagnostic_root]"))
output_root = isempty(ARGS) ? "out_hydrostatic_refinement" : first(ARGS)
diagnostic_directories = length(ARGS) < 2 ? String[] :
                         filter(directory -> isfile(joinpath(directory,
                                                             "diagnostic_variant.json")),
                                readdir(ARGS[2]; join=true))
snapshot = HydrostaticRefinement.publish_refinement(output_root; diagnostic_directories)
println("Published available results from $(snapshot["status"]) campaign: $(snapshot["report_file"])")
