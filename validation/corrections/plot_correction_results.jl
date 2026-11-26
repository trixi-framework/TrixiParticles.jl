using TrixiParticles

include("plot_util.jl")

output_directory = "out"
hydrostatic_directories = isdir("out_hydrostatic_long") ? ("out_hydrostatic_long",) :
                          Tuple(filter(isdir,
                                       ("out_hydrostatic_corrections",
                                        "out_hydrostatic_summation_corrections")))
operators = nothing # Include every operator, including constant-pressure responses.
expanded = true
structure_file = isfile(joinpath("out_hydrostatic_structure", "hydrostatic_structure.csv")) ?
                 joinpath("out_hydrostatic_structure", "hydrostatic_structure.csv") :
                 nothing
report_file = joinpath(output_directory, "correction_validation.html")

report = CorrectionValidationPlots.build_report(;
                                                operator_file=joinpath(output_directory,
                                                                       "correction_convergence.csv"),
                                                hydrostatic_directories, structure_file,
                                                operators, expanded)
CorrectionValidationPlots.write_report(report_file, report)
println("Wrote $report_file ($(report.operator_rows) operator measurements, $(report.coupled_runs) coupled histories)")
