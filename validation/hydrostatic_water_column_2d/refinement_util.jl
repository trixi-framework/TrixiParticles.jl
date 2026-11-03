module HydrostaticRefinement

using TrixiParticles
using CSV
using DataFrames
using JSON
using SHA: sha256

include("correction_methods.jl")
const HCM = HydrostaticCorrectionMethods
normalize(value) = JSON.parse(JSON.json(value))
default_density_mode(method) = string(method) == "wcsph" ? "continuity" : "summation"
base_density_mode(job) = get(job, "density_mode", default_density_mode(job["method"]))
function compatible_density_boundary_policy(settings, job)
    affected = job["method"] == "wcsph" && actual_density_mode(job) == "summation" &&
               HCM.definition(job["correction"]).shepard
    return !affected ||
           get(settings, "density_boundary_policy", "lagged-boundary-density") ==
           "raw-pressure-predictor"
end

function compatible_initialization_policy(settings, job)
    affected = job["method"] == "wcsph" && actual_density_mode(job) == "summation"
    return !affected ||
           get(settings, "wcsph_summation_initialization", "uncalibrated") ==
           "operator-offset-hydrostatic"
end

function compatible_pressure_boundary_policy(settings, job)
    return job["method"] != "wcsph" ||
           get(settings, "wcsph_extrapolated_pressure_gradient", "two-sided") ==
           "fluid-sided"
end

function actual_density_mode(job)
    job["correction"] == "continuity_reinit" ? "continuity" : base_density_mode(job)
end

function plan_jobs(output_root; resolutions=(3, 5, 9, 17), methods=(:wcsph, :edac),
                   density_modes=(:continuity, :summation),
                   correction_cases=HCM.MATRIX_CASES)
    jobs = Dict{String, Any}[]
    for resolution in resolutions, method in methods,
        mode in
        (isnothing(density_modes) ? (Symbol(default_density_mode(method)),) : density_modes),
        case in correction_cases

        mode == :continuity && case == :continuity_reinit && continue
        id = isnothing(density_modes) ? "$(method)_$(case)_$resolution" :
             "$(method)_$(mode)_$(case)_$resolution"
        push!(jobs,
              Dict{String, Any}("id" => id, "method" => string(method),
                                "density_mode" => string(mode),
                                "correction" => string(case),
                                "resolution" => resolution,
                                "directory" => joinpath(output_root, "jobs", id),
                                "status" => "pending"))
    end
    return jobs
end

function write_json(path, data)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON.print(io, data, 2)
    end
    mv(temporary, path; force=true)
    return path
end

function source_signature()
    root = pkgdir(TrixiParticles)
    files = sort([joinpath(directory, file)
                  for (directory, _, filenames) in walkdir(joinpath(root, "src"))
                  for file in filenames if endswith(file, ".jl")])
    append!(files,
            [joinpath(examples_dir(), "fsi", "hydrostatic_water_column_2d.jl"),
                joinpath(examples_dir(), "fsi", "util", "hydrostatic_initialization.jl"),
                joinpath(@__DIR__, "validation_corrections.jl"),
                joinpath(@__DIR__, "correction_util.jl"),
                joinpath(@__DIR__, "correction_methods.jl")])
    return bytes2hex(sha256(join(read.(files, String), "\n")))
end

function case_names(job, protocol)
    name = "$(job["method"])_$(job["correction"])_$(job["resolution"])"
    return job["correction"] == "combined" && protocol["check_reverse_order"] ?
           (name, name * "_reversed") : (name,)
end

function validate_campaign_settings(directory, job, protocol)
    isfile(joinpath(directory, "diagnostic_variant.json")) &&
        error("diagnostic variations cannot be reused as primary campaign results: $directory")
    settings = JSON.parsefile(joinpath(directory, "parameters.json"))
    for key in ("tspan", "sample_dt", "reinitialization_dt", "averaging_interval")
        settings[key] == protocol[key] || error("incompatible $key in $directory")
    end
    stored_mode = get(settings, "$(job["method"])_density_mode",
                      default_density_mode(job["method"]))
    stored_mode == base_density_mode(job) ||
        error("incompatible base density mode in $directory")
    expected = Dict("validation_version" => 2,
                    "fluid_density" => 1000.0, "structure_density" => 2700.0,
                    "gravity" => 9.81, "sound_speed" => 50.0, "E" => 67.5e9, "nu" => 0.3,
                    "plate_size" => [1.0, 0.05], "initial_fluid_size" => [1.0, 2.0],
                    "smoothing_length_factor" => sqrt(2),
                    "edac_adami_pressure_gradient" => "fluid-sided")
    if job["method"] == "wcsph"
        expected["wcsph_pressure_clipping"] = false
        expected["wcsph_viscosity"] = "nothing"
    end
    for (key, value) in expected
        settings[key] == value || error("incompatible $key in $directory")
    end
    return settings
end

function validate_output(directory, job, protocol)
    settings = validate_campaign_settings(directory, job, protocol)
    summary = CSV.read(joinpath(directory, "hydrostatic_corrections.csv"), DataFrame)
    rows = filter(row -> row.method == job["method"] &&
                         row.correction == job["correction"] &&
                         row.resolution == job["resolution"], summary)
    names = case_names(job, protocol)
    nrow(rows) == length(names) || error("incomplete case manifest in $directory")
    expected_definition = normalize(HCM.definition(job["correction"]))
    for name in names
        settings["cases"][name]["definition"] == expected_definition ||
            error("incompatible correction definition in $directory/$name")
        calculator = settings["cases"][name]["density_calculator"]
        expected_calculator = actual_density_mode(job) == "continuity" ?
                              "ContinuityDensity" : "SummationDensity"
        occursin(expected_calculator, calculator) ||
            error("incompatible density calculator in $directory/$name")
        data = JSON.parsefile(joinpath(directory, name,
                                       "validation_run_hydrostatic_corrections.json"))
        errors = only(value
                      for (key, value) in data
                      if startswith(key, "hydrostatic_errors_"))
        first(errors["time"]) == first(protocol["tspan"]) &&
        last(errors["time"]) == last(protocol["tspan"]) ||
            error("incomplete time interval in $directory/$name")
    end
    if length(names) == 2
        order = CSV.read(joinpath(directory, "hydrostatic_correction_order.csv"), DataFrame)
        any(row -> row.method == job["method"] && row.resolution == job["resolution"],
            eachrow(order)) || error("missing order comparison in $directory")
    end
    return rows, settings
end

function reused_failure(job, protocol)
    file = joinpath(job["directory"], "terminal_failure.json")
    isfile(file) || return false
    saved = JSON.parsefile(file)
    saved["protocol"] == protocol || error("resume protocol differs for $(job["id"])")
    compatible_density_boundary_policy(saved, job) ||
        error("cached failure predates the density boundary predictor; use a new output root")
    compatible_initialization_policy(saved, job) ||
        error("cached failure predates hydrostatic reference initialization; use a new output root")
    compatible_pressure_boundary_policy(saved, job) ||
        error("cached failure predates fluid-side WCSPH boundary pressure; use a new output root")
    return true
end

function import_failure!(job, protocol, sources)
    for source in sources
        file = joinpath(source, "parameters.json")
        isfile(file) || continue
        settings = JSON.parsefile(file)
        compatible_density_boundary_policy(settings, job) || continue
        compatible_initialization_policy(settings, job) || continue
        compatible_pressure_boundary_policy(settings, job) || continue
        get(settings, "$(job["method"])_density_mode",
            default_density_mode(job["method"])) ==
        base_density_mode(job) || continue
        ledger = get(settings, "case_status", Dict())
        names = case_names(job, protocol)
        all(name -> haskey(ledger, name), names) || continue
        all(name -> ledger[name]["status"] in ("failed", "blocked"), names) || continue
        validate_campaign_settings(source, job, protocol)
        original = ledger[first(names)]["job_directory"]
        log = joinpath(original, "run.log")
        isfile(log) || continue
        mkpath(job["directory"])
        cp(log, joinpath(job["directory"], "run.log"); force=true)
        write_json(joinpath(job["directory"], "terminal_failure.json"),
                   (; protocol, source_signature=nothing,
                    density_boundary_policy=get(settings, "density_boundary_policy",
                                                "lagged-boundary-density"),
                    wcsph_summation_initialization=get(settings,
                                                       "wcsph_summation_initialization",
                                                       "uncalibrated"),
                    wcsph_extrapolated_pressure_gradient=get(settings,
                                                             "wcsph_extrapolated_pressure_gradient",
                                                             "two-sided"),
                    source_cases=Dict(name => ledger[name] for name in names),
                    provenance="imported terminal outcome with matching campaign settings",
                    imported_from=abspath(source)))
        return true
    end
    return false
end

function completed_job(job, protocol, signature)
    marker = joinpath(job["directory"], "complete.json")
    isfile(marker) || return false
    saved = JSON.parsefile(marker)
    saved["protocol"] == protocol || error("resume protocol differs for $(job["id"])")
    settings = JSON.parsefile(joinpath(job["directory"], "parameters.json"))
    compatible_density_boundary_policy(settings, job) ||
        error("cached density normalization predates the boundary predictor; use a new output root")
    compatible_initialization_policy(settings, job) ||
        error("cached initial state predates hydrostatic reference initialization; use a new output root")
    compatible_pressure_boundary_policy(settings, job) ||
        error("cached pressure flux predates fluid-side WCSPH boundaries; use a new output root")
    if !isnothing(saved["source_signature"])
        saved["source_signature"] == signature ||
            error("simulation source changed since $(job["id"]); use a new output root")
    end
    validate_output(job["directory"], job, protocol)
    return true
end

function import_job!(job, protocol, sources)
    for source in sources
        parameter_file = joinpath(source, "parameters.json")
        isfile(parameter_file) || continue
        settings = JSON.parsefile(parameter_file)
        compatible_density_boundary_policy(settings, job) || continue
        compatible_initialization_policy(settings, job) || continue
        compatible_pressure_boundary_policy(settings, job) || continue
        get(settings, "$(job["method"])_density_mode",
            default_density_mode(job["method"])) ==
        base_density_mode(job) || continue
        all(name -> haskey(get(settings, "cases", Dict()), name),
            case_names(job, protocol)) || continue
        rows, settings = validate_output(source, job, protocol)
        mkpath(job["directory"])
        for name in case_names(job, protocol)
            destination = joinpath(job["directory"], name)
            ispath(destination) && error("unrecorded output already exists at $destination")
            cp(joinpath(source, name), destination)
        end
        settings["cases"] = Dict(name => settings["cases"][name]
                                 for name in case_names(job, protocol))
        settings["correction_cases"] = [job["correction"]]
        settings["methods"] = [job["method"]]
        settings["resolutions"] = [job["resolution"]]
        settings["imported_from"] = abspath(source)
        CSV.write(joinpath(job["directory"], "hydrostatic_corrections.csv"), rows)
        write_json(joinpath(job["directory"], "parameters.json"), settings)
        if length(case_names(job, protocol)) == 2
            order = CSV.read(joinpath(source, "hydrostatic_correction_order.csv"),
                             DataFrame)
            selected = filter(row -> row.method == job["method"] &&
                                     row.resolution == job["resolution"], order)
            CSV.write(joinpath(job["directory"], "hydrostatic_correction_order.csv"),
                      selected)
        end
        write_json(joinpath(job["directory"], "complete.json"),
                   (; protocol, source_signature=nothing,
                    provenance="imported; settings and completion verified; original source hash unavailable"))
        return true
    end
    return false
end

function available_job_output(job, protocol)
    directory = job["directory"]
    if isfile(joinpath(directory, "complete.json"))
        return validate_output(directory, job, protocol)
    end
    # A paired job may have a valid forward trajectory while its reversed run is
    # still running or has failed. Preserve that trajectory without inventing an order check.
    job["correction"] == "combined" || return nothing
    all(isfile(joinpath(directory, file))
        for file in ("parameters.json", "hydrostatic_corrections.csv")) ||
        return nothing
    rows = CSV.read(joinpath(directory, "hydrostatic_corrections.csv"), DataFrame)
    nrow(rows) == 1 && !only(rows.reverse_order) || return nothing
    forward_protocol = copy(protocol)
    forward_protocol["check_reverse_order"] = false
    return validate_output(directory, job, forward_protocol)
end

function job_case_statuses(job, protocol, rows=nothing)
    names = case_names(job, protocol)
    completed = isnothing(rows) ? Set{String}() :
                Set("$(row.method)_$(row.correction)_$(row.resolution)" *
                    (row.reverse_order ? "_reversed" : "") for row in eachrow(rows))
    log_file = joinpath(job["directory"], "run.log")
    log = isfile(log_file) ? read(log_file, String) : ""
    started = [match.captures[1]
               for match in
                   eachmatch(r"(?m)^Running hydrostatic correction case (\S+)", log)]
    last_started = isempty(started) ? first(names) : last(started)
    error_match = match(r"ERROR: (?:LoadError: )?([^\n]*)(?:\n([^\n]*))?", log)
    reason = isnothing(error_match) ? get(job, "error", "") :
             join(filter(!isnothing, collect(error_match.captures)), "\n")
    result = Dict{String, Any}()
    for (index, name) in enumerate(names)
        status = if name in completed
            "completed"
        elseif job["status"] == "failed"
            name == last_started || index == 1 ? "failed" : "blocked"
        elseif job["status"] == "running"
            name == last_started ? "running" : "pending"
        else
            "pending"
        end
        detail = status == "failed" ? reason :
                 status == "blocked" ?
                 "not run because the forward trajectory failed" : ""
        failure_kind = status == "blocked" ? "predecessor_failed" :
                       status == "failed" &&
                       occursin("out of bounds for cell grid", reason) ?
                       "cell_grid_bounds" : status == "failed" ? "execution_error" : ""
        result[name] = (; method=job["method"], correction=job["correction"],
                        base_density_mode=base_density_mode(job),
                        density_mode=actual_density_mode(job),
                        resolution=job["resolution"],
                        reverse_order=endswith(name, "_reversed"),
                        status, failure_kind, reason=detail,
                        target_time=last(protocol["tspan"]),
                        final_time=name in completed ? last(protocol["tspan"]) : missing,
                        job_directory=job["directory"])
    end
    return result
end

function assemble_campaigns!(jobs, protocol, output_root; allow_incomplete=false)
    directories = String[]
    all_statuses = NamedTuple[]
    template = nothing
    summary_template = DataFrame()
    for job in jobs
        output = available_job_output(job, protocol)
        if !isnothing(output)
            template = deepcopy(last(output))
            summary_template = first(output)[1:0, :]
            summary_template.base_density_mode = String[]
            summary_template.density_mode = String[]
            break
        end
    end
    for resolution in sort(unique(job["resolution"] for job in jobs)),
        method in sort(unique(job["method"] for job in jobs)),
        mode in
        sort(unique(base_density_mode(job) for job in jobs if job["method"] == method))

        group = filter(job -> job["resolution"] == resolution && job["method"] == method &&
                              base_density_mode(job) == mode,
                       jobs)
        isempty(group) && continue
        !allow_incomplete && !all(job -> job["status"] in ("completed", "reused"), group) &&
            continue
        directory = joinpath(output_root, "campaigns", "$(method)_$(mode)_$resolution")
        mkpath(directory)
        summaries, orders = DataFrame[], DataFrame[]
        settings = nothing
        cases = Dict{String, Any}()
        statuses = Dict{String, Any}()
        for job in group
            output = allow_incomplete ? available_job_output(job, protocol) :
                     validate_output(job["directory"], job, protocol)
            records = job_case_statuses(job, protocol,
                                        isnothing(output) ? nothing : first(output))
            merge!(statuses, records)
            for (name, record) in records
                push!(all_statuses, (; name, record...))
            end
            isnothing(output) && continue
            rows, parameters = output
            rows.base_density_mode = fill(mode, nrow(rows))
            rows.density_mode = fill(actual_density_mode(job), nrow(rows))
            push!(summaries, rows)
            isnothing(settings) && (settings = deepcopy(parameters))
            for name in case_names(job, protocol)
                haskey(parameters["cases"], name) || continue
                haskey(records, name) && records[name].status == "completed" || continue
                cases[name] = parameters["cases"][name]
                destination = joinpath(directory, name)
                ispath(destination) || cp(joinpath(job["directory"], name), destination)
            end
            if length(case_names(job, protocol)) == 2 && nrow(rows) == 2
                order = CSV.read(joinpath(job["directory"],
                                          "hydrostatic_correction_order.csv"), DataFrame)
                order.base_density_mode = fill(mode, nrow(order))
                push!(orders, order)
            end
        end
        if isempty(summaries)
            isnothing(template) && continue
            settings = deepcopy(template)
            settings["planned_only"] = true
        end
        summary = isempty(summaries) ? copy(summary_template) :
                  vcat(summaries...; cols=:union)
        CSV.write(joinpath(directory, "hydrostatic_corrections.csv"), summary)
        isempty(orders) ||
            CSV.write(joinpath(directory, "hydrostatic_correction_order.csv"),
                      vcat(orders...; cols=:union))
        settings["cases"] = cases
        settings["methods"] = [method]
        settings["resolutions"] = [resolution]
        settings["base_density_mode"] = mode
        settings["$(method)_density_mode"] = mode
        settings["$(method)_density_calculator"] = mode == "continuity" ?
                                                   "ContinuityDensity" : "SummationDensity"
        settings["correction_cases"] = [job["correction"] for job in group]
        settings["campaign_complete"] = all(record -> record.status == "completed",
                                            values(statuses))
        settings["case_status"] = statuses
        settings["planned_resolutions"] = sort(unique(job["resolution"] for job in jobs))
        write_json(joinpath(directory, "parameters.json"), settings)
        push!(directories, directory)
    end
    if !isempty(all_statuses)
        CSV.write(joinpath(output_root, "hydrostatic_case_status.csv"),
                  DataFrame(all_statuses))
        write_json(joinpath(output_root, "hydrostatic_case_status.json"), all_statuses)
    end
    return directories
end

function refinement_status(output_root)
    manifest = JSON.parsefile(joinpath(output_root, "refinement_progress.json"))
    jobs, protocol = manifest["jobs"], manifest["protocol"]
    finished(job) = job["status"] in ("completed", "reused")
    levels = Dict(resolution => (;
                                 validated_runs=sum(length(case_names(job, protocol))
                                                    for job in jobs
                                                    if job["resolution"] == resolution &&
                                                       finished(job);
                                                    init=0),
                                 expected_runs=sum(length(case_names(job, protocol))
                                                   for job in jobs
                                                   if job["resolution"] == resolution;
                                                   init=0),
                                 running_jobs=count(job -> job["resolution"] ==
                                                           resolution &&
                                                           job["status"] == "running",
                                                    jobs),
                                 failed_jobs=count(job -> job["resolution"] == resolution &&
                                                          job["status"] == "failed", jobs))
                  for resolution in manifest["resolutions"])
    return (; status=manifest["status"], expected_runs=manifest["expected_runs"],
            validated_runs=sum(length(case_names(job, protocol))
                               for job in jobs if finished(job); init=0),
            levels,
            configurations=Dict("$method/$mode" => (;
                                                    validated_runs=sum(length(case_names(job,
                                                                                         protocol))
                                                                       for job in jobs
                                                                       if job["method"] ==
                                                                          method &&
                                                                          base_density_mode(job) ==
                                                                          mode &&
                                                                          finished(job);
                                                                       init=0),
                                                    expected_runs=sum(length(case_names(job,
                                                                                        protocol))
                                                                      for job in jobs
                                                                      if job["method"] ==
                                                                         method &&
                                                                         base_density_mode(job) ==
                                                                         mode; init=0))
                                for (method, mode) in
                                    unique([(job["method"], base_density_mode(job))
                                            for job in jobs])),
            report_file=manifest["report_file"],
            report_status=get(manifest, "report_status", "pending"))
end

function publish_refinement(output_root; destination=joinpath(output_root, "report_data"),
                            report_file=nothing, completion_command=String[],
                            diagnostic_directories=())
    manifest = JSON.parsefile(joinpath(output_root, "refinement_progress.json"))
    mkpath(destination)
    directories = assemble_campaigns!(manifest["jobs"], manifest["protocol"], destination;
                                      allow_incomplete=true)
    isempty(directories) && error("no completed trajectories are available for a report")
    snapshot = deepcopy(manifest)
    snapshot["campaign_directories"] = directories
    snapshot["allow_incomplete"] = true
    snapshot["diagnostic_directories"] = abspath.(collect(diagnostic_directories))
    isnothing(report_file) || (snapshot["report_file"] = abspath(report_file))
    snapshot_file = write_json(joinpath(destination, "refinement_progress.json"), snapshot)
    project = dirname(Base.active_project())
    report_worker = joinpath(@__DIR__, "refinement_report.jl")
    run(`$(Base.julia_cmd()) --project=$project --threads=1 $report_worker $snapshot_file`)
    isempty(completion_command) || run(Cmd(completion_command))
    return snapshot
end

function run_refinement(; output_root="out_hydrostatic_refinement",
                        resolutions=(3, 5, 9, 17),
                        methods=(:wcsph, :edac), density_modes=(:continuity, :summation),
                        correction_cases=HCM.MATRIX_CASES,
                        workers=12, reuse_directories=(),
                        tspan=(0.0, 1.0), averaging_interval=(0.5, 1.0), sample_dt=0.005,
                        reinitialization_dt=0.01, check_reverse_order=true,
                        generate_report=true, operator_file=nothing, structure_file=nothing,
                        report_file=joinpath(output_root, "correction_validation.html"),
                        completion_command=String[], execute=true,
                        diagnostic_directories=())
    workers >= 1 || throw(ArgumentError("workers must be positive"))
    all(method -> method in (:wcsph, :edac), methods) ||
        throw(ArgumentError("methods must contain only :wcsph or :edac"))
    allunique(methods) && allunique(correction_cases) ||
        throw(ArgumentError("methods and correction cases must be unique"))
    isnothing(density_modes) ||
        (allunique(density_modes) &&
         all(mode -> mode in (:continuity, :summation), density_modes)) ||
        throw(ArgumentError("density_modes must be unique continuity/summation choices"))
    all(resolution -> resolution >= 3, resolutions) && allunique(resolutions) ||
        throw(ArgumentError("resolutions must be unique and at least three"))
    foreach(HCM.definition, correction_cases)
    output_root = abspath(output_root)
    mkpath(output_root)
    progress_file = joinpath(output_root, "refinement_progress.json")
    if isfile(progress_file)
        previous = JSON.parsefile(progress_file)
        if get(previous, "status", "") == "running"
            for job in previous["jobs"]
                pid = get(job, "pid", 0)
                if job["status"] == "running" && 0 < pid <= typemax(Cint) &&
                   !isfile(joinpath(job["directory"], "complete.json")) &&
                   ccall(:uv_kill, Cint, (Cint, Cint), Cint(pid), 0) != Base.UV_ESRCH
                    error("refinement workers are still running; inspect refinement_status instead of starting a duplicate study")
                end
            end
        end
    end
    protocol = normalize((; tspan, averaging_interval, sample_dt, reinitialization_dt,
                          check_reverse_order))
    signature = source_signature()
    jobs = plan_jobs(output_root; resolutions, methods, density_modes, correction_cases)
    manifest = Dict{String, Any}("protocol" => protocol, "source_signature" => signature,
                                 "resolutions" => collect(resolutions),
                                 "density_modes" => density_modes,
                                 "jobs" => jobs,
                                 "expected_runs" => sum(length(case_names(job, protocol))
                                                        for job in jobs),
                                 "report_file" => abspath(report_file),
                                 "operator_file" => operator_file,
                                 "structure_file" => structure_file,
                                 "diagnostic_directories" => abspath.(collect(diagnostic_directories)),
                                 "status" => "running")
    manifest_file = progress_file
    for job in jobs
        if completed_job(job, protocol, signature) ||
           import_job!(job, protocol, reuse_directories)
            job["status"] = "reused"
        elseif reused_failure(job, protocol) ||
               import_failure!(job, protocol, reuse_directories)
            job["status"] = "failed"
            job["reused_failure"] = true
        end
    end
    write_json(manifest_file, manifest)
    if !execute
        manifest["campaign_directories"] = assemble_campaigns!(jobs, protocol, output_root;
                                                               allow_incomplete=true)
        manifest["status"] = "prepared"
        manifest["allow_incomplete"] = true
        write_json(manifest_file, manifest)
        return manifest
    end
    queue = Channel{Dict{String, Any}}(length(jobs))
    for job in jobs
        job["status"] == "pending" && put!(queue, job)
    end
    close(queue)
    project = dirname(Base.active_project())
    worker_file = joinpath(@__DIR__, "refinement_worker.jl")
    @sync for _ in 1:min(workers, length(jobs))
        @async for job in queue
            mkpath(job["directory"])
            input = write_json(joinpath(job["directory"], "input.json"),
                               (; job, protocol, source_signature=signature))
            job["status"] = "running"
            job["started_unix"] = time()
            write_json(manifest_file, manifest)
            println("Starting $(job["id"])")
            flush(stdout)
            try
                command = `$(Base.julia_cmd()) --project=$project --threads=1 $worker_file $input`
                open(joinpath(job["directory"], "run.log"), "w") do log
                    process = run(pipeline(addenv(command, "OPENBLAS_NUM_THREADS" => "1",
                                                  "GKSwstype" => "100"), stdout=log,
                                           stderr=log);
                                  wait=false)
                    job["pid"] = getpid(process)
                    write_json(manifest_file, manifest)
                    wait(process)
                    success(process) || error("worker exited with code $(process.exitcode)")
                end
                completed_job(job, protocol, signature) ||
                    error("worker did not save a completion record")
                job["status"] = "completed"
            catch exception
                job["status"] = "failed"
                job["error"] = sprint(showerror, exception)
            end
            job["elapsed_seconds"] = time() - job["started_unix"]
            write_json(manifest_file, manifest)
            println("$(job["id"]): $(job["status"])")
            flush(stdout)
        end
    end
    directories = assemble_campaigns!(jobs, protocol, output_root; allow_incomplete=true)
    manifest["campaign_directories"] = directories
    manifest["status"] = all(job -> job["status"] in ("completed", "reused"), jobs) ?
                         "completed" : "failed"
    manifest["allow_incomplete"] = manifest["status"] != "completed"
    write_json(manifest_file, manifest)
    if !isempty(directories) && generate_report
        report_worker = joinpath(@__DIR__, "refinement_report.jl")
        manifest["report_status"] = "running"
        write_json(manifest_file, manifest)
        try
            run(`$(Base.julia_cmd()) --project=$project --threads=1 $report_worker $manifest_file`)
            isempty(completion_command) || run(Cmd(completion_command))
            manifest["report_status"] = "completed"
        catch exception
            manifest["report_status"] = "failed"
            manifest["report_error"] = sprint(showerror, exception)
            rethrow()
        finally
            write_json(manifest_file, manifest)
        end
    end
    return manifest
end

end # module
