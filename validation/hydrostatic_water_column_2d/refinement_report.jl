using TrixiParticles
using CSV
using DataFrames
using JSON

include(joinpath("..", "corrections", "plot_util.jl"))
include("refinement_util.jl")
manifest_file = only(ARGS)
manifest = JSON.parsefile(manifest_file)
allow_incomplete = get(manifest, "allow_incomplete", false)
manifest["status"] == "completed" || allow_incomplete ||
    error("the refinement campaign is incomplete")
directories = Tuple(manifest["campaign_directories"])
summary = vcat([CSV.read(joinpath(directory, "hydrostatic_corrections.csv"), DataFrame)
                for directory in directories]...; cols=:union)
allow_incomplete || nrow(summary) == manifest["expected_runs"] ||
    error("incomplete refinement summary")
expected_cases = Set((job["method"], HydrostaticRefinement.base_density_mode(job),
                      job["correction"], job["resolution"], reversed)
                     for job in manifest["jobs"]
                     for reversed in (job["correction"] == "combined" &&
                          manifest["protocol"]["check_reverse_order"] ?
                          (false, true) : (false,)))
actual_cases = Set((row.method, row.base_density_mode, row.correction, row.resolution,
                    row.reverse_order)
                   for row in eachrow(summary))
((allow_incomplete && issubset(actual_cases, expected_cases)) ||
 actual_cases == expected_cases) ||
    error("refinement method/resolution coverage differs from the plan")
CSV.write(joinpath(dirname(manifest_file), "hydrostatic_refinement_summary.csv"), summary)
orders = [CSV.read(joinpath(directory, "hydrostatic_correction_order.csv"), DataFrame)
          for directory in directories
          if isfile(joinpath(directory, "hydrostatic_correction_order.csv"))]
order = isempty(orders) ? DataFrame() : vcat(orders...; cols=:union)
isempty(order) ||
    CSV.write(joinpath(dirname(manifest_file), "hydrostatic_refinement_order.csv"), order)
checks = NamedTuple[]
for row in eachrow(summary),
    (field, bound) in
    ((:interface_residual, 2e-12), (:eos_residual, 1e-12),
     (:reinitialization_volume_residual, 1e-12),
     (:post_reinitialization_eos_residual, 1e-12),
     (:edac_reinitialization_pressure_change, 1e-12))

    value = row[field]
    ismissing(value) && continue
    push!(checks,
          (; method=row.method, base_density_mode=row.base_density_mode,
           correction=row.correction, resolution=row.resolution,
           reverse_order=row.reverse_order, diagnostic=string(field), value, bound,
           passed=isfinite(value) && value < bound))
end
for row in eachrow(order),
    field in (:density_difference, :pressure_difference, :deflection_difference)
    value = row[field]
    push!(checks,
          (; method=row.method, base_density_mode=row.base_density_mode,
           correction="combined", resolution=row.resolution,
           reverse_order=true, diagnostic="order_$(field)", value, bound=1e-6,
           passed=isfinite(value) && value < 1e-6))
end
CSV.write(joinpath(dirname(manifest_file), "hydrostatic_acceptance.csv"), DataFrame(checks))
report = CorrectionValidationPlots.build_report(;
                                                operator_file=manifest["operator_file"],
                                                hydrostatic_directories=directories,
                                                structure_file=manifest["structure_file"],
                                                diagnostic_directories=get(manifest,
                                                                           "diagnostic_directories",
                                                                           ()),
                                                expanded=true)
CSV.write(joinpath(dirname(manifest_file), "hydrostatic_refinement.csv"), report.refinement)
status_cases = Set((row.method, row.base_density_mode, row.correction, row.resolution,
                    row.reverse_order)
                   for row in eachrow(report.case_status))
status_cases == expected_cases ||
    error("case status ledger does not cover the complete plan")
CSV.write(joinpath(dirname(manifest_file), "hydrostatic_case_status.csv"),
          report.case_status)
status_count(status) = count(==(status), report.case_status.status)
failed_cases, blocked_cases = status_count("failed"), status_count("blocked")
running_cases, pending_cases = status_count("running"), status_count("pending")
failed_checks = count(check -> !check.passed, checks)
all_done = nrow(summary) == manifest["expected_runs"]
numerical_pass = failed_checks > 0 || failed_cases + blocked_cases > 0 ? false :
                 all_done ? true : nothing
audit = (; data_integrity=all_done ? "passed" : "available_data_verified",
         execution_finished=running_cases + pending_cases == 0,
         completed_runs=nrow(summary),
         expected_runs=manifest["expected_runs"],
         failed_cases, blocked_cases, running_cases, pending_cases,
         observed_refinement_rows=nrow(report.refinement),
         diagnostic_runs=report.diagnostic_runs,
         numerical_checks=length(checks),
         failed_numerical_checks=failed_checks,
         numerical_checks_passed=numerical_pass,
         note="Numerical bound exceedances remain reported; data completion does not imply accuracy acceptance")
open(joinpath(dirname(manifest_file), "refinement_audit.json"), "w") do io
    JSON.print(io, audit, 2)
end
CorrectionValidationPlots.write_report(manifest["report_file"], report)
println("Wrote $(manifest["report_file"]) ($(report.coupled_runs) coupled histories)")
