using TrixiParticles

include(joinpath("..", "corrections", "plot_util.jl"))

output_directory = "out_hydrostatic_corrections"
operator_file = isfile(joinpath("out", "correction_convergence.csv")) ?
                joinpath("out", "correction_convergence.csv") : nothing
hydrostatic_directories = isdir("out_hydrostatic_summation_corrections") ?
                          (output_directory, "out_hydrostatic_summation_corrections") :
                          (output_directory,)
report_file = joinpath(output_directory, "correction_validation.html")
expanded = true
structure_file = isfile(joinpath("out_hydrostatic_structure", "hydrostatic_structure.csv")) ?
                 joinpath("out_hydrostatic_structure", "hydrostatic_structure.csv") :
                 nothing

report = CorrectionValidationPlots.build_report(; operator_file, hydrostatic_directories,
                                                structure_file, expanded)
CorrectionValidationPlots.write_report(report_file, report)
println("Wrote $report_file ($(report.operator_rows) operator measurements, $(report.coupled_runs) coupled histories)")
