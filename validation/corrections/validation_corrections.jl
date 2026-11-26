using TrixiParticles

include("convergence.jl")

resolutions = (12, 24, 48, 96)
output_directory = "out"

results = CorrectionConvergence.run_convergence(; resolutions)
CorrectionConvergence.print_report(results)
csv_file = joinpath(output_directory, "correction_convergence.csv")
report_file = joinpath(output_directory, "correction_convergence.md")
CorrectionConvergence.write_csv(csv_file, results)
open(report_file, "w") do io
    println(io, "# SPH correction consistency study\n")
    println(io, "Regular unit-square patches, Wendland C6 kernel, h / Δx = 2.")
    println(io, "Resolutions: ", join(resolutions, ", "), ".\n")
    println(io, "Boundary and symmetric-interior samples are reported separately.")
    println(io,
            "These are local operator scalings, not full-discretization convergence orders.\n")
    CorrectionConvergence.print_report(results; io)
end
println("\nWrote $csv_file and $report_file")
