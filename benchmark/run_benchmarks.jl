#!/usr/bin/env julia
# Run surface reconstruction benchmarks and optionally compare against a baseline.
#
# Usage:
#   julia --project=benchmark benchmark/run_benchmarks.jl \
#       [--workloads=block_small,block_1e4] [--repeats=3] \
#       [--output=results.json] [--baseline=benchmark/baselines/<file>.json] \
#       [--tolerance=0.10]
#
# Without `--workloads`, all workloads except the opt-in `wave_tank` are run.
# Exit code is 1 when any workload regresses beyond `--tolerance` relative time
# (minimum) or 5% relative allocation growth.
using Pkg

Pkg.develop(path=dirname(dirname(Base.active_project())))
Pkg.instantiate()

include(joinpath(@__DIR__, "workloads.jl"))

using .Workloads
using Printf: @printf

function parse_args(args)
    options = Dict{String, String}(
        "workloads" => join(Workloads.DEFAULT_WORKLOADS, ","),
        "repeats" => "3",
        "output" => "",
        "baseline" => "",
        "tolerance" => "0.10"
    )
    for arg in args
        startswith(arg, "--") && occursin("=", arg) ||
            error("expected `--option=value`, got: $arg")
        key, value = split(arg[3:end], "="; limit=2)
        haskey(options, key) || error("unknown option: $key")
        options[key] = value
    end
    return options
end

function git_sha()
    project_dir = dirname(Base.active_project())
    current = pwd()
    try
        cd(project_dir)
        return chomp(read(`git rev-parse --short HEAD`, String))
    catch
        return "unknown"
    finally
        cd(current)
    end
end

function write_json(path, data::Dict, sha)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"meta\": {")
        println(io, "    \"julia\": \"$(VERSION)\",")
        println(io, "    \"threads\": $(Threads.nthreads()),")
        println(io, "    \"machine\": \"$(gethostname())\",")
        println(io, "    \"git\": \"$sha\"")
        println(io, "  },")
        println(io, "  \"workloads\": {")
        names = sort(collect(keys(data)))
        for (index, name) in enumerate(names)
            result = data[name]
            comma = index < length(names) ? "," : ""
            println(io, "    \"$name\": {")
            println(io, "      \"time_min_seconds\": $(result["time_min_seconds"]),")
            println(io, "      \"time_median_seconds\": $(result["time_median_seconds"]),")
            println(io, "      \"allocated_bytes\": $(result["allocated_bytes"]),")
            println(io, "      \"workspace_bytes\": $(result["workspace_bytes"]),")
            println(io, "      \"particles\": $(result["particles"]),")
            println(io, "      \"grid_dimensions\": $(result["grid_dimensions"]),")
            println(io,
                    "      \"n_correction_evaluations\": $(result["n_correction_evaluations"]),")
            println(io, "      \"vertices\": $(result["vertices"]),")
            println(io, "      \"faces\": $(result["faces"]),")
            println(io, "      \"volume\": $(result["volume"]),")
            println(io, "      \"effective_isovalue\": $(result["effective_isovalue"])")
            println(io, "    }$comma")
        end
        println(io, "  }")
        println(io, "}")
    end
    return path
end

function read_simple_json(path)
    # Minimal reader for files written by `write_json` (flat numeric/string values).
    # Returns the workload results and the thread count of the run.
    data = Dict{String, Any}()
    threads = 0
    current = ""
    for line in eachline(path)
        m = match(r"^    \"threads\": ([0-9]+),?$", line)
        if !isnothing(m)
            threads = parse(Int, m.captures[1])
            continue
        end
        m = match(r"^    \"([^\"]+)\": \{$", line)
        if !isnothing(m)
            current = m.captures[1]
            data[current] = Dict{String, Any}()
            continue
        end
        m = match(r"^      \"([^\"]+)\": ([0-9.e+\-]+),?$", line)
        if !isnothing(m) && !isempty(current)
            data[current][m.captures[1]] = parse(Float64, m.captures[2])
        end
    end
    return data, threads
end

function main()
    options = parse_args(ARGS)
    workload_names = split(options["workloads"], ",")
    repeats = parse(Int, options["repeats"])
    tolerance = parse(Float64, options["tolerance"])
    Workloads.set_repeats!(repeats)

    results = Dict{String, Any}()
    for name in workload_names
        haskey(Workloads.WORKLOADS, name) || error("unknown workload: $name")
        println("running workload $name (repeats=$repeats, threads=$(Threads.nthreads()))")
        results[name] = Workloads.WORKLOADS[name]()
        result = results[name]
        @printf("  min %.3f s | median %.3f s | allocated %.1f MiB | grid %s | evaluations %d\n",
                result["time_min_seconds"], result["time_median_seconds"],
                result["allocated_bytes"] / 2^20,
                join(result["grid_dimensions"], "x"), result["n_correction_evaluations"])
    end

    if !isempty(options["output"])
        mkpath(dirname(options["output"]))
        write_json(options["output"], results, git_sha())
        println("wrote $(options["output"])")
    end

    exit_code = 0
    if !isempty(options["baseline"])
        baseline, baseline_threads = read_simple_json(options["baseline"])
        println()
        if baseline_threads != Threads.nthreads()
            println("WARNING: the baseline was recorded with $baseline_threads threads, " *
                    "this run uses $(Threads.nthreads()); time ratios are not comparable")
        end
        println("workload | time ratio (min) | alloc ratio | verdict")
        for name in workload_names
            if !haskey(baseline, name)
                @printf("%-12s | not in baseline\n", name)
                continue
            end
            base = baseline[name]
            result = results[name]
            time_ratio = result["time_min_seconds"] / base["time_min_seconds"]
            alloc_ratio = result["allocated_bytes"] / base["allocated_bytes"]
            ok = time_ratio <= 1 + tolerance && alloc_ratio <= 1.05
            exit_code = ok ? exit_code : 1
            @printf("%-12s | %8.3f | %8.3f | %s\n", name, time_ratio, alloc_ratio,
                    ok ? "OK" : "REGRESSION")
        end
    end

    return exit_code
end

exit(main())
