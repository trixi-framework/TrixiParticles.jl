using TrixiParticles

include("convergence.jl")

dimension = !isempty(ARGS) && (@__FILE__) == abspath(PROGRAM_FILE) ?
            parse(Int, only(ARGS)) : 2
resolutions = CorrectionConvergence.default_resolutions(dimension)
output_directory = joinpath("out", "corrections_periodic",
                            dimension == 2 ? "" : "$(dimension)d")

results = CorrectionConvergence.run_convergence(; dimension, resolutions)
CorrectionConvergence.print_report(results)
csv_file = joinpath(output_directory, "correction_convergence.csv")
report_file = joinpath(output_directory, "correction_convergence.md")
CorrectionConvergence.write_csv(csv_file, results)
open(report_file, "w") do io
    println(io, "# SPH correction consistency: $(dimension)D periodic case\n")
    println(io,
            "Unit domain, periodic in every coordinate, Wendland C6 kernel, h / Δx = 2.")
    println(io, "Resolutions: ", join(resolutions, ", "), ".\n")
    println(io,
            "Errors include all N^$dimension particles. Constant-pressure errors are absolute RMS accelerations.")
    println(io,
            "These are local operator scalings, not full-discretization convergence orders.\n")
    CorrectionConvergence.print_report(results; io)
end
println("\nWrote $csv_file and $report_file")
