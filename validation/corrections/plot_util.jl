module CorrectionValidationPlots

using Base64: base64encode
using CSV
using DataFrames
using JSON
using Plots
using Printf: @sprintf

include(joinpath("..", "hydrostatic_water_column_2d", "correction_methods.jl"))
include(joinpath("..", "hydrostatic_water_column_2d", "refinement_metrics.jl"))

const METHODS = ("none", "shepard", "kernel", "gradient", "blended", "mixed",
                 "shepard_mixed", "combined", "continuity_reinit", "shepard_kernel",
                 "shepard_gradient", "shepard_blended")
# Dark colors retain contrast on white paper; symbols provide an independent encoding.
const COLORS = ("#005AB5", "#007A4D", "#202020", "#B02973", "#8C6500", "#C23B22",
                "#6335A5", "#6335A5", "#006D82", "#404040", "#76275D", "#665000")
const LABELS = ("Uncorrected [N]", "Shepard [S]", "Kernel [K]", "Gradient [G]",
                "Blend λ=0.5 [B]", "Mixed [M]",
                "Shepard + mixed [SM]", "Shepard + mixed [SM]", "Continuity + reinit [CR]",
                "Shepard + kernel [SK]", "Shepard + gradient [SG]", "Shepard + blend [SB]")
const CODES = ("N", "S", "K", "G", "B", "M", "SM", "SM", "CR", "SK", "SG", "SB")
const MARKERS = (:circle, :rect, :diamond, :utriangle, :dtriangle, :xcross, :cross,
                 :cross, :star5, :diamond, :utriangle, :dtriangle)
const DENSITY_COLORS = Dict(("interpolation", "none") => COLORS[1],
                            ("interpolation", "shepard") => COLORS[2],
                            ("summation_density", "none") => COLORS[4],
                            ("summation_density", "shepard") => COLORS[6],
                            ("density_reinitialization", "shepard") => COLORS[7])
const REFERENCE_COLOR = "#666666"
const DIAGNOSTICS = ((:circle, "Force (2e-12)", "#005AB5", 2e-12),
                     (:diamond, "EOS (1e-12)", "#007A4D", 1e-12),
                     (:star5, "Order (1e-6)", "#6335A5", 1e-6),
                     (:utriangle, "Reinit (1e-12)", "#8C6500", 1e-12))
const DENSITY_OPERATORS = ("interpolation", "summation_density", "density_reinitialization")

function escape_html(value)
    replace(string(value), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
            '"' => "&quot;", '\'' => "&#39;")
end
function number(value)
    ismissing(value) || isnothing(value) ? "—" :
    value isa AbstractFloat ? @sprintf("%.4g", value) : escape_html(value)
end
method_index(method) = something(findfirst(==(string(method)), METHODS), 1)
color(method) = COLORS[method_index(method)]
label(method) = LABELS[method_index(method)]
function marker_face(marker, series_color, hollow)
    hollow && !(marker in (:cross, :xcross)) ?
    :white : series_color
end
logvalues(values) = max.(values, 1e-18)
function order_label(order)
    order == -1 ? "O(h⁻¹)" : order == 0 ? "O(1)" :
                             order == 1 ? "O(h)" : "O(h²)"
end

function legend_columns_for(panel)
    clamp(count(series -> series[:primary] && !isempty(series[:label]),
                panel.series_list), 1, 3)
end

function base_plot(; kwargs...)
    return plot(; fontfamily="sans-serif", titlefontsize=12, guidefontsize=11,
                tickfontsize=9, framestyle=:box, gridalpha=0.15,
                background_color=:white, foreground_color="#334155",
                legend=false, left_margin=8Plots.mm, bottom_margin=5Plots.mm,
                linewidth=2.4, kwargs...)
end

function image(figure, description)
    svg = repr(MIME"image/svg+xml"(), figure)
    width, height = figure.attr[:size]
    return "<img width='$width' height='$height' alt='$(escape_html(description))' " *
           "src='data:image/svg+xml;base64,$(base64encode(svg))'>"
end

function download(bytes, filename, description; mime="application/json")
    return "<a download='$(escape_html(filename))' href='data:$mime;base64,$(base64encode(bytes))'>" *
           escape_html(description) * "</a>"
end

function operator_panel(operator)
    operator in DENSITY_OPERATORS && return 1
    operator in ("direct_gradient", "difference_gradient") && return 2
    startswith(operator, "pressure_") && return 3
    startswith(operator, "constant_pressure_") && return 4
    throw(ArgumentError("unknown correction operator: $operator"))
end

"""Show every variant through its min–max envelope; the curve is the maximum error."""
function add_envelope!(panel, data; panel_index, method, region, name, marker,
                       series_color=color(method))
    groups = groupby(sort(data, :resolution), :resolution; sort=true)
    resolution = [first(group.resolution) for group in groups]
    lower = [minimum(group.l2_error) for group in groups]
    upper = [maximum(group.l2_error) for group in groups]
    plot!(panel, resolution, logvalues(upper);
          ribbon=(logvalues(upper) - logvalues(lower), zeros(length(upper))),
          fillalpha=0.025, color=series_color, marker, markersize=5.2,
          markercolor=marker_face(marker, series_color, region == "interior"),
          markerstrokecolor=series_color, markerstrokewidth=1.3,
          linestyle=region == "boundary" ? :solid : :dash,
          label=region == "boundary" ? name : "")
    return (; panel=panel_index, method, region, operators=unique(data.operator),
            resolution, lower, upper, count=nrow(data))
end

function add_guide!(panel, data, method, region, order; factor=0.35)
    selected = filter(row -> row.method == method && row.region == region, data)
    isempty(selected) && return
    groups = groupby(sort(selected, :resolution), :resolution; sort=true)
    x = [first(group.resolution) for group in groups]
    y = factor * maximum(first(groups).l2_error) .* (x ./ first(x)) .^ (-order)
    plot!(panel, x, y; color=REFERENCE_COLOR, linestyle=:dot, linewidth=1.3,
          label="$(order_label(order)) guide", primary=false)
    annotate!(panel, last(x), last(y) * 1.35,
              text(order_label(order), 9, REFERENCE_COLOR, :right))
end

function add_region_key!(panel)
    for (region, style, hollow) in (("Boundary", :solid, false), ("Interior", :dash, true))
        plot!(panel, Float64[], Float64[]; color=REFERENCE_COLOR, linestyle=style,
              marker=:circle, markersize=4, markercolor=hollow ? :white : REFERENCE_COLOR,
              markerstrokecolor=REFERENCE_COLOR, label=region)
    end
end

function annotate_overlap!(panel, curves, description; factor, rtol=0.03)
    length(curves) >= 2 || return
    all(curve -> curve.resolution == first(curves).resolution, curves) || return
    for index in eachindex(first(curves).resolution)
        values = [curve.upper[index] for curve in curves]
        maximum(values) - minimum(values) <= rtol * maximum(values) || return
    end
    y = maximum(curve -> last(curve.upper), curves) * factor
    y > 0 || return
    annotate!(panel, last(first(curves).resolution), y,
              text(description, 9, REFERENCE_COLOR, :right))
end

function empty_panel!(panel, message)
    plot!(panel; xscale=:identity, yscale=:identity, xlims=(0, 1), ylims=(0, 1),
          framestyle=:none, xlabel="", ylabel="", xticks=false, yticks=false)
    annotate!(panel, 0.5, 0.5, text(message, 11))
end

function operator_overview(data)
    titles = ("A  Interpolation & density\nInterpolation/reinit: O(h) boundary; O(h²) interior",
              "B  Direct & difference gradients\nKernel removes h⁻¹ leakage; full correction: O(h) / O(h²)",
              "C  Conservative pressure acceleration\nGradient/mixed: O(h²) interior; boundary floor allowed",
              "D  Constant-pressure response\nInterior target = 0; boundary O(h⁻¹) response allowed")
    panels = [base_plot(; title, xscale=:log2,
                        yscale=:log10,
                        ylabel=index == 4 ? "Absolute RMS acceleration" :
                               "Relative L₂ error")
              for (index, title) in enumerate(titles)]
    envelopes = NamedTuple[]
    categories = [filter(row -> operator_panel(row.operator) == index, data)
                  for index in 1:4]
    for (index, selected) in enumerate(categories)
        if isempty(selected)
            empty_panel!(panels[index], "No measurements supplied for this panel")
            continue
        end
        for method in METHODS, region in ("boundary", "interior")
            subset = filter(row -> row.method == method && row.region == region, selected)
            isempty(subset) && continue
            if index == 1
                for (operator, name, marker) in
                    (("interpolation", "Interpolation", method == "none" ? :circle : :rect),
                     ("summation_density", "Summation density",
                      method == "none" ? :diamond : :utriangle),
                     ("density_reinitialization", "Density reinit", :xcross))
                    values = filter(row -> row.operator == operator, subset)
                    isempty(values) && continue
                    push!(envelopes,
                          add_envelope!(panels[index], values;
                                        panel_index=index, method, region,
                                        name=operator == "density_reinitialization" ?
                                             "Shepard reinitialization" :
                                             "$(method == "none" ? "Raw" : "Shepard") $(lowercase(name))",
                                        marker,
                                        series_color=get(DENSITY_COLORS, (operator, method),
                                                         color(method))))
                end
            else
                push!(envelopes,
                      add_envelope!(panels[index], subset;
                                    panel_index=index, method, region,
                                    name=label(method),
                                    marker=MARKERS[method_index(method)]))
            end
        end
        add_region_key!(panels[index])
        resolutions = sort(unique(selected.resolution))
        plot!(panels[index]; xticks=(resolutions, string.(resolutions)),
              xlims=(first(resolutions) / 1.04, last(resolutions) * 1.08),
              legend=:outerbottom,
              legend_columns=index == 1 ? 2 : legend_columns_for(panels[index]),
              legendfontsize=9)
    end
    add_guide!(panels[1], filter(row -> row.operator == "interpolation", data), "shepard",
               "boundary", 1)
    add_guide!(panels[1], filter(row -> row.operator == "interpolation", data), "shepard",
               "interior", 2)
    for (method, region, order) in (("none", "boundary", -1), ("kernel", "boundary", 0),
         ("mixed", "boundary", 1), ("mixed", "interior", 2))
        add_guide!(panels[2], categories[2], method, region, order;
                   factor=order == 0 ? 0.15 : 0.35)
    end
    add_guide!(panels[3], categories[3], "mixed", "interior", 2)
    add_guide!(panels[3], categories[3], "none", "boundary", 0; factor=0.10)
    add_guide!(panels[4], categories[4], "none", "boundary", -1; factor=0.06)
    annotate_overlap!(panels[1],
                      filter(curve -> curve.panel == 1 && curve.method == "none" &&
                                      curve.region == "boundary", envelopes),
                      "Raw boundary interpolation / density\nnearly coincide"; factor=0.30,
                      rtol=0.05)
    annotate_overlap!(panels[2],
                      filter(curve -> curve.panel == 2 && curve.region == "interior" &&
                                      curve.method in ("gradient", "mixed"), envelopes),
                      "Gradient / Mixed\nnearly coincide (interior)"; factor=6.0)
    annotate_overlap!(panels[3],
                      filter(curve -> curve.panel == 3 && curve.region == "interior" &&
                                      curve.method in ("none", "shepard", "kernel"),
                             envelopes),
                      "Interior N / S / K\nmaxima nearly coincide"; factor=2.6)
    annotate_overlap!(panels[3],
                      filter(curve -> curve.panel == 3 && curve.region == "interior" &&
                                      curve.method in ("gradient", "mixed", "shepard_mixed"),
                             envelopes),
                      "Interior G / M / SM\nmaxima nearly coincide"; factor=4.0)
    if !isempty(categories[4])
        hline!(panels[4], [1e-7]; color=REFERENCE_COLOR, linestyle=:dashdot, linewidth=1.4,
               label="Interior regression bound", primary=false)
        annotate!(panels[4], minimum(categories[4].resolution), 2e-7,
                  text("Interior bound: 10⁻⁷", 9, REFERENCE_COLOR, :left))
        interior = filter(row -> row.region == "interior", categories[4])
        if !isempty(interior)
            worst = maximum(interior.l2_error)
            boundary = filter(row -> row.region == "boundary", categories[4])
            y = isempty(boundary) ? 1e-5 : sqrt(1e-7 * minimum(boundary.l2_error))
            x = sqrt(minimum(interior.resolution) * maximum(interior.resolution))
            message = worst <= 1e-7 ? "All interior responses below bound" :
                      "Interior regression bound exceeded"
            annotate!(panels[4], x, y,
                      text("$message\nWorst: $(number(worst))  ($(number(100worst/1e-7))% of bound)",
                           9, REFERENCE_COLOR))
        end
        lower, upper = ylims(panels[4])
        exponents = 4floor(Int, log10(lower) / 4):4:4ceil(Int, log10(upper) / 4)
        ticks = [10.0^exponent for exponent in exponents if lower <= 10.0^exponent <= upper]
        plot!(panels[4];
              yticks=(ticks,
                      [exponent == 0 ? "1" : "1e$exponent"
                       for exponent in exponents if lower <= 10.0^exponent <= upper]))
    end
    return plot(panels...; layout=(2, 2), size=(1600, 1100)), envelopes
end

function series(data, name)
    keys_ = filter(key -> startswith(key, name * "_"), collect(keys(data)))
    return isempty(keys_) ? nothing : data[only(keys_)]
end

function base_density_mode(method, parameters)
    get(parameters, "base_density_mode",
        get(parameters, "$(method)_density_mode",
            method == "wcsph" ? "continuity" : "summation"))
end
base_density_mode(run) = base_density_mode(run.method, run.parameters)

function load_diagnostics(directories)
    diagnostics = NamedTuple[]
    for directory in unique(abspath.(collect(directories)))
        variant = JSON.parsefile(joinpath(directory, "diagnostic_variant.json"))
        get(variant, "diagnostic", false) ||
            throw(ArgumentError("diagnostic metadata is required for $directory"))
        state_file = joinpath(directory, "diagnostic.json")
        state = isfile(state_file) ? JSON.parsefile(state_file) : Dict{String, Any}()
        name = "$(variant["method"])_$(variant["correction"])_$(variant["resolution"])"
        history_file = joinpath(directory, name,
                                "validation_run_hydrostatic_corrections.json")
        history = isfile(history_file) ? JSON.parsefile(history_file) : Dict{String, Any}()
        log_file = joinpath(directory, "run.log")
        log = isfile(log_file) ? read(log_file, String) : ""
        target = get(variant, "final_time", nothing)
        time = get(state, "integrator_time", nothing)
        outcome = isempty(state) ? "running" :
                  !isnothing(get(state, "exception", nothing)) ?
                  "exception" :
                  !isnothing(target) && !isnothing(time) && time < target ?
                  "stopped_before_target" : "finished"
        push!(diagnostics, (; directory, name, variant, state, history, log, outcome))
    end
    return diagnostics
end

function load_runs(directories)
    runs = NamedTuple[]
    sources = NamedTuple[]
    for directory in unique(abspath.(collect(directories)))
        isdir(directory) ||
            throw(ArgumentError("hydrostatic result directory does not exist: $directory"))
        parameter_file = joinpath(directory, "parameters.json")
        parameters = isfile(parameter_file) ? JSON.parsefile(parameter_file) :
                     Dict{String, Any}()
        variant_file = joinpath(directory, "diagnostic_variant.json")
        isfile(variant_file) &&
            (parameters["diagnostic_variant"] = JSON.parsefile(variant_file))
        get(parameters, "validation_version", 1) == 2 ||
            throw(ArgumentError("outdated hydrostatic reference data in $directory; rerun " *
                                "validation_corrections.jl to generate validation_version=2 data"))
        summary_file = joinpath(directory, "hydrostatic_corrections.csv")
        summary = isfile(summary_file) ? CSV.read(summary_file, DataFrame) : DataFrame()
        source = length(sources) + 1
        count_before = length(runs)
        for (root, _, files) in walkdir(directory)
            filename = "validation_run_hydrostatic_corrections.json"
            filename in files || continue
            matched = match(r"^(wcsph|edac)_(.+)_(\d+)(_reversed)?$",
                            basename(root))
            isnothing(matched) && continue
            method, correction, resolution_, reversed = matched.captures
            Symbol(correction) in HydrostaticCorrectionMethods.DEFAULT_CASES || continue
            resolution = parse(Int, resolution_)
            reverse_order = !isnothing(reversed)
            saved = isempty(summary) ? summary :
                    filter(row -> row.method == method && row.correction == correction &&
                                  row.resolution == resolution &&
                                  row.reverse_order == reverse_order,
                           summary)
            # The summary is the manifest of this campaign. Ignore stale case directories
            # left behind by an earlier run with different cases or a different reference.
            !isempty(summary) && isempty(saved) && continue
            data = JSON.parsefile(joinpath(root, filename))
            errors = series(data, "hydrostatic_errors")
            deflection = series(data, "plate_deflection")
            forces = series(data, "interface_force_balance")
            eos = series(data, "eos_residual")
            reference = get(parameters, "analytical_deflection", nothing)
            reinitialization = isempty(saved) ? missing : only(saved.last_reinitialization)
            saved_value(name) = isempty(saved) || !hasproperty(saved, name) ? missing :
                                only(saved[!, name])
            audit = filter(!ismissing,
                           [saved_value(:reinitialization_volume_residual),
                               saved_value(:post_reinitialization_eos_residual),
                               saved_value(:edac_reinitialization_pressure_change)])
            reinit_residual = isempty(audit) ? missing : maximum(audit)
            post_eos = saved_value(:post_reinitialization_eos_residual)
            eos_residual = isnothing(eos) ? missing : maximum(eos["values"])
            ismissing(post_eos) || (eos_residual = ismissing(eos_residual) ? post_eos :
                            max(eos_residual, post_eos))
            push!(runs,
                  (; source, directory, name=basename(root), method, correction,
                   resolution, reverse_order, data, parameters, reference,
                   final_time=last(errors["time"]),
                   density_error=last(errors["values"])[1],
                   max_density_error=maximum(value -> value[1], errors["values"]),
                   pressure_error=last(errors["values"])[2],
                   max_pressure_error=maximum(value -> value[2], errors["values"]),
                   late_density_error=saved_value(:late_density_error),
                   late_pressure_error=saved_value(:late_pressure_error),
                   spacing=saved_value(:spacing),
                   deflection_error=saved_value(:deflection_error),
                   initial_density_error=saved_value(:initial_density_error),
                   initial_pressure_error=saved_value(:initial_pressure_error),
                   interface_residual=maximum(value -> value[7], forces["values"]),
                   eos_residual, reinit_residual,
                   reinitialization))
        end
        length(runs) > count_before || haskey(parameters, "case_status") ||
            throw(ArgumentError("no completed hydrostatic histories in $directory"))
        push!(sources, (; directory, parameters, count=length(runs) - count_before))
    end
    correction_order = string.(HydrostaticCorrectionMethods.DEFAULT_CASES)
    sort!(runs;
          by=run -> (run.method, base_density_mode(run), run.resolution, run.source,
                     findfirst(==(run.correction), correction_order), run.reverse_order))
    return runs, sources
end

function refinement_samples(runs)
    rows = NamedTuple[]
    protocols = Dict{String, String}()
    # Do not combine different density treatments, durations, windows or material setups.
    keys = ("tspan", "averaging_interval", "sample_dt", "reinitialization_dt",
            "smoothing_kernel", "smoothing_length_factor", "E", "nu", "sound_speed",
            "plate_size", "initial_fluid_size", "edac_adami_pressure_gradient",
            "diagnostic_variant")
    for run in runs
        run.reverse_order && continue
        (ismissing(run.spacing) || ismissing(run.deflection_error)) && continue
        scheme_settings = run.method == "wcsph" ?
                          [get(run.parameters, "wcsph_pressure_clipping", false),
            get(run.parameters, "wcsph_viscosity", "nothing")] : []
        run.method == "wcsph" && push!(scheme_settings,
              get(run.parameters, "wcsph_extrapolated_pressure_gradient", "two-sided"))
        if run.method == "wcsph" && density_mode_code(run) == "Σ"
            push!(scheme_settings,
                  get(run.parameters, "wcsph_summation_initialization", "uncalibrated"))
        end
        if run.method == "wcsph" && density_mode_code(run) == "Σ" &&
           HydrostaticCorrectionMethods.definition(run.correction).shepard
            push!(scheme_settings,
                  get(run.parameters, "density_boundary_policy", "lagged-boundary-density"))
        end
        signature = JSON.json(([get(run.parameters, key, nothing) for key in keys],
                               scheme_settings))
        protocol = get!(protocols, signature) do
            "p$(length(protocols) + 1)"
        end
        push!(rows,
              (; method=run.method, correction=run.correction,
               density_model=density_mode_code(run) == "Σ" ? "summation" : "continuity",
               protocol, resolution=run.resolution, spacing=run.spacing,
               late_density_error=run.late_density_error,
               late_pressure_error=run.late_pressure_error,
               deflection_error=run.deflection_error))
    end
    return DataFrame(rows), Dict(value => key for (key, value) in protocols)
end

function case_status_table(sources, runs)
    rows = NamedTuple[]
    for (source, info) in enumerate(sources)
        ledger = get(info.parameters, "case_status", nothing)
        if isnothing(ledger)
            for run in runs
                run.source == source || continue
                push!(rows,
                      (; source, name=run.name, method=run.method,
                       base_density_mode=base_density_mode(run),
                       density_mode=density_mode_code(run) == "Σ" ? "summation" :
                                    "continuity",
                       correction=run.correction, resolution=run.resolution,
                       reverse_order=run.reverse_order, status="completed",
                       failure_kind="", reason="", target_time=run.final_time,
                       final_time=run.final_time, job_directory=run.directory))
            end
        else
            for (name, record) in ledger
                time = get(record, "final_time", nothing)
                push!(rows,
                      (; source, name, method=record["method"],
                       base_density_mode=get(record, "base_density_mode",
                                             base_density_mode(record["method"],
                                                               info.parameters)),
                       density_mode=get(record, "density_mode",
                                        record["correction"] == "continuity_reinit" ?
                                        "continuity" :
                                        base_density_mode(record["method"],
                                                          info.parameters)),
                       correction=record["correction"],
                       resolution=record["resolution"],
                       reverse_order=record["reverse_order"],
                       status=record["status"],
                       failure_kind=get(record, "failure_kind", ""),
                       reason=get(record, "reason", ""), target_time=record["target_time"],
                       final_time=isnothing(time) ? missing : time,
                       job_directory=get(record, "job_directory", info.directory)))
            end
        end
    end
    result = DataFrame(rows)
    if !isempty(result)
        for column in (:failure_kind, :reason)
            result[!,
                   column] = [ismissing(value) || isempty(value) ? missing : value
                              for value in result[!, column]]
        end
        actual = Set((run.source, run.name) for run in runs)
        declared = Set((row.source, row.name)
                       for row in eachrow(result) if row.status == "completed")
        actual == declared || error("case status ledger disagrees with completed histories")
        sort!(result, [:method, :resolution, :source, :correction, :reverse_order])
    end
    return result
end

function matrix_case_status(runs, group, correction)
    name = "$(group.method)_$(correction)_$(group.resolution)"
    record = get(get(group.parameters, "case_status", Dict()), name, nothing)
    return isnothing(record) ? "unavailable" : record["status"]
end

function use_error_matrix(runs)
    length(runs) > 20 && return true
    return any(run -> any(record -> record["status"] != "completed",
                          values(get(run.parameters, "case_status", Dict()))), runs)
end

function use_refinement_plot(samples)
    isempty(samples) && return false
    samples = filter(row -> row.correction != "continuity_reinit", samples)
    isempty(samples) && return false
    return any(group -> length(unique(group.resolution)) >= 3,
               groupby(samples, [:method, :density_model, :protocol]))
end

function plate_refinement_plot(samples, structure; groups=())
    panel = base_plot(;
                      title="G  Plate-deflection refinement vs Δx [m]\nLine = worst error; band = method range; target = 0",
                      ylabel="Late mean deflection error [%], log", xscale=:log2,
                      yscale=:log10)
    displayed_errors = Float64[]
    shown = Set{Tuple{String, String}}()
    matrix = filter(row -> row.correction != "continuity_reinit", samples)
    for (index, group) in
        enumerate(groupby(matrix, [:method, :density_model, :protocol]; sort=true))
        resolutions = Set(group.resolution)
        length(resolutions) >= 3 || continue
        group = HydrostaticRefinementMetrics.common_method_samples(group)
        isempty(group) && continue
        common_methods = unique(group.correction)
        push!(shown, (first(group.method), first(group.density_model)))
        levels = groupby(sort(group, :spacing), :resolution; sort=false)
        h = [first(level.spacing) for level in levels]
        lower = [100minimum(level.deflection_error) for level in levels]
        upper = [100maximum(level.deflection_error) for level in levels]
        append!(displayed_errors, lower, upper)
        ink = ("#005AB5", "#C23B22", "#007A4D", "#6335A5")[mod1(index, 4)]
        plot!(panel, h, logvalues(upper);
              ribbon=(logvalues(upper) - logvalues(lower), zeros(length(h))),
              fillalpha=0.12, color=ink, marker=isodd(index) ? :circle : :rect,
              markercolor=:white, markerstrokecolor=ink, markersize=5,
              label="$(uppercase(first(group.method))) $(first(group.density_model)) · $(length(common_methods)) common")
    end
    for (method, mode) in unique([(group.method, group.density_mode) for group in groups])
        (method, mode) in shown && continue
        plot!(panel, Float64[], Float64[]; color="#64748b", marker=:x,
              label="$(uppercase(method)) $mode · unavailable")
    end
    if !isempty(structure)
        selected = filter(row -> row.resolution in samples.resolution, structure)
        sort!(selected, :spacing)
        if !isempty(selected)
            append!(displayed_errors, 100abs.(selected.relative_bias))
            plot!(panel, selected.spacing, logvalues(100abs.(selected.relative_bias));
                  color="#202020", linestyle=:dash, marker=:diamond, markercolor=:white,
                  markerstrokecolor="#202020", label="Isolated TLSPH")
        end
    end
    levels = sort(unique([(row.spacing, row.resolution) for row in eachrow(samples)]))
    plot!(panel;
          xticks=([level[1] for level in levels],
                  ["$(number(level[1]))\nn=$(level[2])" for level in levels]),
          legend=:outerbottom, legend_columns=2, legendfontsize=9)
    positive = filter(>(0), displayed_errors)
    if !isempty(positive)
        lower, upper = minimum(positive) / 1.3, maximum(positive) * 1.3
        plot!(panel; ylims=(lower, upper), yticks=percentage_ticks(lower, upper))
    end
    return panel
end

function order_difference(run, runs)
    counterparts = filter(other -> other.source == run.source &&
                                   other.method == run.method &&
                                   other.correction == run.correction &&
                                   other.resolution == run.resolution &&
                                   other.reverse_order != run.reverse_order, runs)
    isempty(counterparts) && return missing
    other = only(counterparts)
    differences = Float64[]
    for (name, component) in
        (("hydrostatic_errors", 3), ("hydrostatic_errors", 4), ("plate_deflection", 0))
        a, b = series(run.data, name), series(other.data, name)
        a["time"] == b["time"] || error("order comparison requires matching sample times")
        x = component == 0 ? a["values"] : getindex.(a["values"], component)
        y = component == 0 ? b["values"] : getindex.(b["values"], component)
        scale = max(maximum(abs, x), maximum(abs, y), eps())
        push!(differences, maximum(abs.(x - y)) / scale)
    end
    return maximum(differences)
end

function history_style(run, resolutions)
    resolution_index = findfirst(==(run.resolution), resolutions)
    class_index = 2 * (resolution_index - 1) + (run.method == "edac" ? 1 : 2)
    styles = (:solid, :dash, :dashdot, :dashdotdot)
    return (;
            linestyle=run.reverse_order ? :dot : styles[mod1(class_index, length(styles))],
            class_index)
end

function run_groups(runs; sources=())
    groups = NamedTuple[]
    start = 1
    key(run) = (run.source, run.method, run.resolution)
    for index in eachindex(runs)
        if index == length(runs) || key(runs[index]) != key(runs[index + 1])
            group = runs[start:index]
            times = unique(getproperty.(group, :final_time))
            duration = length(times) == 1 ? "$(number(only(times))) s" : "varying time"
            run = first(group)
            push!(groups,
                  (; first_index=start, last_index=index,
                   source=run.source, method=run.method, resolution=run.resolution,
                   density_mode=base_density_mode(run), parameters=run.parameters,
                   final_time=maximum(times),
                   label="$(uppercase(run.method)) n$(run.resolution)\ns$(run.source) · $duration"))
            start = index + 1
        end
    end
    for (source, info) in enumerate(sources)
        any(group -> group.source == source, groups) && continue
        ledger = values(get(info.parameters, "case_status", Dict()))
        isempty(ledger) && continue
        method = only(unique(record["method"] for record in ledger))
        resolution = only(unique(record["resolution"] for record in ledger))
        mode = base_density_mode(method, info.parameters)
        time = last(info.parameters["tspan"])
        push!(groups,
              (; first_index=1, last_index=0, source, method, resolution,
               density_mode=mode, parameters=info.parameters, final_time=time,
               label="$(uppercase(method)) n$resolution · no completed trajectories"))
    end
    sort!(groups;
          by=group -> (group.method, group.density_mode, group.resolution, group.source))
    return groups
end

function configuration_spans(groups)
    spans = NamedTuple[]
    start = 1
    for index in eachindex(groups)
        group = groups[index]
        if index == length(groups) ||
           (group.method, group.density_mode) !=
           (groups[index + 1].method, groups[index + 1].density_mode)
            push!(spans,
                  (; first=start, last=index, method=group.method,
                   density_mode=group.density_mode))
            start = index + 1
        end
    end
    return spans
end

function percentage_ticks(lower, upper)
    ticks = [multiplier * 10.0^exponent
             for exponent in floor(Int, log10(lower)):ceil(Int, log10(upper))
             for multiplier in (1, 2, 5) if lower <= multiplier * 10.0^exponent <= upper]
    return (ticks, number.(ticks))
end

function grouped_run_axis!(panel, runs, groups, lower, upper, ticks)
    labels = [CODES[method_index(run.correction)] * (run.reverse_order ? "r" : "")
              for run in runs]
    plot!(panel; yscale=:log10, ylims=(lower, upper), yticks=ticks,
          xticks=(collect(eachindex(runs)), labels), xtickfontsize=9,
          xlims=(0.5, length(runs) + 0.5), xgrid=false)
    for (index, group) in enumerate(groups)
        if iseven(index)
            vspan!(panel, [group.first_index - 0.5, group.last_index + 0.5];
                   color="#94a3b8", alpha=0.045, label="")
        end
        group.last_index == length(runs) ||
            vline!(panel, [group.last_index + 0.5]; color="#b8c2ce", linewidth=0.8,
                   label="")
        annotate!(panel, (group.first_index + group.last_index) / 2, upper / 2.4,
                  text(group.label, 9, "#334155"))
    end
end

function diagnostic_scaling(runs, differences)
    values = [Tuple(ismissing(value) ? missing : value / diagnostic[4]
                    for (diagnostic, value) in zip(DIAGNOSTICS,
                            (run.interface_residual, run.eos_residual, differences[index],
                             run.reinit_residual)))
              for (index, run) in enumerate(runs)]
    positive = Float64[value for row in values
                       for value in row if !ismissing(value) && value > 0]
    minimum_positive = isempty(positive) ? 1e-6 : minimum(positive)
    zero_row = 10.0^(floor(Int, log10(max(1e-300, min(1.0, minimum_positive)))) - 1)
    maximum_positive = isempty(positive) ? 1.0 : max(1.0, maximum(positive))
    exponents = collect((floor(Int, log10(zero_row)) + 1):2:ceil(Int,
                                                                 log10(maximum_positive)))
    push!(exponents, 0)
    sort!(unique!(exponents))
    ticks = vcat(zero_row, 10.0 .^ exponents)
    return (; values, zero_row, upper=10maximum_positive,
            ticks=(ticks, vcat("0", number.(10.0 .^ exponents))))
end

function mean_deflection_bias(run)
    (isnothing(run.reference) || iszero(run.reference)) && return missing
    interval = get(run.parameters, "averaging_interval", [])
    isempty(interval) && return missing
    data = series(run.data, "plate_deflection")
    indices = findall(t -> first(interval) <= t <= last(interval), data["time"])
    isempty(indices) && return missing
    return sum(data["values"][indices]) / (length(indices) * run.reference) - 1
end

function density_mode_code(run)
    settings = get(get(run.parameters, "cases", Dict()), run.name, Dict())
    calculator = get(settings, "density_calculator",
                     run.method == "wcsph" ?
                     get(run.parameters, "wcsph_density_calculator", "ContinuityDensity") :
                     "SummationDensity")
    return occursin("SummationDensity", calculator) ? "Σ" : "C"
end

function configuration_label(runs, group; compact=false)
    compact && return "$(uppercase(group.method))\nn=$(group.resolution)"
    density = group.density_mode == "summation" ? "Summation" : "Continuity"
    return "$(uppercase(group.method))\n$density\nn=$(group.resolution) · $(number(group.final_time)) s"
end

function correction_matrix(runs, groups, field, maximum_field)
    corrections = [string(case)
                   for case in HydrostaticCorrectionMethods.DEFAULT_CASES
                   if any(run -> run.correction == string(case), runs) ||
                      any(group -> any(record -> record["correction"] == string(case),
                                       values(get(group.parameters, "case_status", Dict()))),
                          groups)]
    final = fill(NaN, length(groups), length(corrections))
    peak = copy(final)
    members = Matrix{Vector{Int}}(undef, size(final))
    for (row, group) in enumerate(groups), (column, correction) in enumerate(corrections)
        indices = [index
                   for index in group.first_index:group.last_index
                   if runs[index].correction == correction]
        members[row, column] = indices
        isempty(indices) && continue
        final[row, column] = 100maximum(index -> getproperty(runs[index], field), indices)
        peak[row,
             column] = 100maximum(index -> getproperty(runs[index], maximum_field), indices)
    end
    return (; final, peak, members, corrections)
end

function error_matrix_plot(runs, groups, field, maximum_field, title)
    matrix = correction_matrix(runs, groups, field, maximum_field)
    wide = length(groups) > 8
    finite = filter(isfinite,
                    vec(log10.(logvalues(wide ? vcat(matrix.final, matrix.peak) :
                                         matrix.final))))
    lower = isempty(finite) ? -3.0 : floor(minimum(finite))
    upper = isempty(finite) ? 2.0 : max(lower + 1, maximum(finite))
    labels = HydrostaticCorrectionMethods.axis_label.(matrix.corrections)
    panel = base_plot(; title, background_color_inside="#edf1f5")
    method_rows = length(labels) > 6
    dense = method_rows && length(groups) > 4
    grouped_columns = dense
    x = method_rows ? (1:length(groups)) : (1:length(labels))
    y = method_rows ? (1:length(labels)) : (1:length(groups))
    method_labels = [replace(label, "\nShepard off" => " · no Shepard",
                             "\nShepard on" => " · Shepard", "\n" => " · ")
                     for label in labels]
    configurations = [configuration_label(runs, group) for group in groups]
    if grouped_columns
        configurations = [wide ? string(group.resolution) : "n=$(group.resolution)"
                          for group in groups]
    end
    heatmap!(panel, x, y,
             logvalues(method_rows ? permutedims(matrix.final) : matrix.final);
             color=:viridis, clims=(10.0^lower, 10.0^upper), colorbar=true,
             colorbar_scale=:log10, colorbar_title="Error [%], log",
             colorbar_tickfontsize=8,
             xticks=(x, method_rows ? configurations : labels), xtickfontsize=9,
             yticks=(y, method_rows ? method_labels : configurations),
             ytickfontsize=9, yflip=true, xlims=(first(x) - 0.5, last(x) + 0.5),
             ylims=(grouped_columns ? -0.5 : first(y) - 0.5, last(y) + 0.5),
             xlabel=grouped_columns ? "Particles through plate thickness n" :
                    method_rows ? "Scheme / density model / resolution · time" :
                    "Gradient correction / Shepard density correction",
             grid=false)
    for row in axes(matrix.final, 1), column in axes(matrix.final, 2)
        value = matrix.final[row, column]
        px, py = method_rows ? (row, column) : (column, row)
        status = matrix_case_status(runs, groups[row], matrix.corrections[column])
        if dense
            fraction = isnan(value) ? 0.0 :
                       clamp((log10(max(value, 1e-18)) - lower) / (upper - lower), 0.0, 1.0)
            fill = isnan(value) ? (status == "failed" ? "#fce0e0" : "#edf1f5") :
                   cgrad(:viridis)[fraction]
            cell = Shape([(px - 0.5, py - 0.5), (px + 0.5, py - 0.5),
                             (px + 0.5, py + 0.5), (px - 0.5, py + 0.5)])
            plot!(panel, cell; fillcolor=fill, linecolor=:white, linewidth=1, label="")
            if wide && isfinite(value)
                peak_fraction = clamp((log10(max(matrix.peak[row, column], 1e-18)) - lower) /
                                      (upper - lower), 0.0, 1.0)
                peak_cell = Shape([(px, py - 0.5), (px + 0.5, py - 0.5),
                                      (px + 0.5, py + 0.5), (px, py + 0.5)])
                plot!(panel, peak_cell; fillcolor=cgrad(:viridis)[peak_fraction],
                      linecolor=:transparent, label="")
            end
        end
        if isnan(value)
            marker = status == "failed" ? (wide ? "×" : "FAIL") :
                     status == "running" ? (wide ? "•" : "RUN") :
                     status == "pending" ? (wide ? "…" : "WAIT") : "—"
            annotate!(panel, px, py,
                      text(marker, marker == "—" ? 12 : 8,
                           status == "failed" ? "#991b1b" : "#64748b"))
        else
            wide && continue
            fraction = (log10(max(value, 1e-18)) - lower) / (upper - lower)
            ink = fraction < 0.45 ? :white : :black
            value_label = if dense
                compact(value) = replace(@sprintf("%.2g", value), r"^0\." => ".")
                "$(compact(value))\n$(compact(matrix.peak[row, column]))"
            else
                "$(number(value))\nmax $(number(matrix.peak[row, column]))"
            end
            annotate!(panel, px, py,
                      text(value_label, 9,
                           ink))
        end
    end
    for boundary in (first(x) - 0.5):1:(last(x) + 0.5)
        plot!(panel, [boundary, boundary], [first(y) - 0.5, last(y) + 0.5];
              color=:white, linewidth=1, label="")
    end
    hline!(panel, collect((first(y) - 0.5):1:(last(y) + 0.5)); color=:white, linewidth=1,
           label="")
    if grouped_columns
        header = Shape([(first(x) - 0.5, -0.5), (last(x) + 0.5, -0.5),
                           (last(x) + 0.5, 0.5), (first(x) - 0.5, 0.5)])
        plot!(panel, header; fillcolor=:white, linecolor=:white, label="")
        for span in configuration_spans(groups)
            annotate!(panel, (span.first + span.last) / 2, 0.0,
                      text("$(uppercase(span.method)) · $(span.density_mode)", wide ? 8 : 9,
                           "#334155"))
            span.last == length(groups) ||
                vline!(panel, [span.last + 0.5]; color="#334155", linewidth=1, label="")
        end
    end
    return panel
end

function use_late_errors(runs)
    all(run -> !ismissing(run.late_density_error) &&
               !ismissing(run.late_pressure_error), runs)
end

function use_deflection_envelopes(runs)
    count(run -> !ismissing(mean_deflection_bias(run)), runs) > 8 || return false
    return all(run_groups(runs)) do group
        members = runs[group.first_index:group.last_index]
        all(run -> ismissing(mean_deflection_bias(run)), members) ||
            any(run -> run.correction == "none" && !run.reverse_order &&
                       !ismissing(mean_deflection_bias(run)), members)
    end
end

function deflection_envelope_plot(runs, groups)
    panel = base_plot(;
                      title="G  Plate deflection vs time [s]: method ranges\nLine = uncorrected; band = all methods; dashed = mixed",
                      ylabel="Deflection / analytical deflection")
    colors = ("#005AB5", "#C23B22", "#007A4D", "#6335A5")
    markers = (:circle, :rect, :diamond, :utriangle)
    intervals = Set{Tuple{Float64, Float64}}()
    for (index, group) in enumerate(groups)
        members = [run
                   for run in runs[group.first_index:group.last_index]
                   if !ismissing(mean_deflection_bias(run)) && !run.reverse_order]
        isempty(members) && continue
        raw = only(filter(run -> run.correction == "none", members))
        base = series(raw.data, "plate_deflection")
        all(run -> series(run.data, "plate_deflection")["time"] == base["time"], members) ||
            error("deflection envelopes require matching sample times")
        values = reduce(hcat,
                        [series(run.data, "plate_deflection")["values"] ./ run.reference
                         for run in members])
        lower, upper = vec(minimum(values; dims=2)), vec(maximum(values; dims=2))
        baseline = base["values"] ./ raw.reference
        ink = colors[mod1(index, length(colors))]
        marker = markers[mod1(index, length(markers))]
        plot!(panel, base["time"], baseline;
              ribbon=(baseline - lower, upper - baseline), fillalpha=0.13,
              color=ink, linewidth=1.8, label="")
        stride = max(1, cld(length(baseline), 10))
        sampled = unique(vcat(collect(index:stride:length(baseline)), length(baseline)))
        scatter!(panel, base["time"][sampled], baseline[sampled]; color=ink, marker,
                 markercolor=:white, markerstrokecolor=ink, markersize=4, label="")
        plot!(panel, Float64[], Float64[]; color=ink, marker, markercolor=:white,
              markerstrokecolor=ink, markersize=4,
              label="$(uppercase(raw.method)) n$(raw.resolution) · $(length(members)) methods")
        mixed = filter(run -> run.correction == "mixed", members)
        if !isempty(mixed)
            run = only(mixed)
            data = series(run.data, "plate_deflection")
            plot!(panel, data["time"], data["values"] ./ run.reference;
                  color=ink, linestyle=:dash, linewidth=1.6, label="")
        end
        interval = raw.parameters["averaging_interval"]
        push!(intervals, (first(interval), min(last(interval), raw.final_time)))
    end
    for interval in intervals
        vspan!(panel, collect(interval); color="#64748b", alpha=0.055, label="")
    end
    hline!(panel, [1.0]; color="#202020", linestyle=:dot, linewidth=1.8,
           label="Analytical (=1)")
    plot!(panel; legend=:outerbottom, legend_columns=2, legendfontsize=9,
          xlims=(0.0, maximum(run -> run.final_time, runs)))
    return panel
end

function grouped_diagnostic_plot(runs, groups, data)
    wide = length(groups) > 8
    panel = base_plot(;
                      title="H  Available algebraic checks by configuration\nMarker = maximum; range = completed cases; limit = 1",
                      ylabel="Residual / its regression bound", yscale=:log10,
                      ylims=(data.zero_row / 2, wide ? 1.5data.upper : data.upper / 5),
                      yticks=data.ticks,
                      xlims=(0.5, length(groups) + 0.5), xgrid=false,
                      xticks=(1:length(groups),
                              [wide ? string(group.resolution) :
                               configuration_label(runs, group; compact=length(groups) > 4)
                               for group in groups]),
                      xtickfontsize=8)
    hspan!(panel, [data.zero_row / 2, sqrt(10) * data.zero_row]; color="#94a3b8",
           alpha=0.07, label="")
    for (index, group) in enumerate(groups),
        (kind, (marker, name, ink, limit)) in enumerate(DIAGNOSTICS)
        values = Float64[data.values[row][kind]
                         for row in group.first_index:group.last_index
                         if !ismissing(data.values[row][kind])]
        isempty(values) && continue
        lower, upper = extrema(values)
        x = index + (kind - (length(DIAGNOSTICS) + 1) / 2) * (wide ? 0.22 : 0.12)
        plot!(panel, [x, x],
              [
                  iszero(lower) ? data.zero_row : lower,
                  iszero(upper) ? data.zero_row : upper
              ];
              color=ink, linewidth=1.2, label="")
        scatter!(panel, [x], [iszero(upper) ? data.zero_row : upper]; color=ink, marker,
                 markersize=wide ? 3.5 : 5.5, markercolor=marker == :diamond ? :white : ink,
                 markerstrokecolor=ink, markerstrokewidth=1.3, label="")
    end
    for (marker, name, ink, _) in DIAGNOSTICS
        scatter!(panel, Float64[], Float64[]; color=ink, marker, markersize=5.5,
                 markercolor=marker == :diamond ? :white : ink, markerstrokecolor=ink,
                 markerstrokewidth=1.3, label=name)
    end
    hline!(panel, [1.0]; color=REFERENCE_COLOR, linestyle=:dash, linewidth=1.4,
           label="Regression limit")
    if wide
        for span in configuration_spans(groups)
            annotate!(panel, (span.first + span.last) / 2, 0.6data.upper,
                      text("$(uppercase(span.method))\n$(span.density_mode)", 8, "#334155"))
            span.last == length(groups) ||
                vline!(panel, [span.last + 0.5]; color="#cbd5e1", linewidth=0.8, label="")
        end
        plot!(panel; xlabel="Particles through plate thickness n")
    end
    plot!(panel; legend=:outerbottom, legend_columns=2, legendfontsize=9)
    return panel
end

function coupled_overview(runs; structure=DataFrame(), sources=())
    groups = run_groups(runs; sources)
    if isempty(runs)
        panels = [
            error_matrix_plot(runs, groups, :density_error, :max_density_error,
                              "E  Density outcomes"),
            error_matrix_plot(runs, groups, :pressure_error, :max_pressure_error,
                              "F  Pressure outcomes"),
            base_plot(; title="G  No completed trajectories for refinement"),
            grouped_diagnostic_plot(runs, groups, diagnostic_scaling(runs, []))]
        return plot(panels...; layout=(2, 2), size=(1600, 1100)), Union{Missing, Float64}[]
    end
    normalized_deflection = all(run -> !isnothing(run.reference) && !iszero(run.reference),
                                runs)
    biases = mean_deflection_bias.(runs)
    show_mean_labels = 0 < count(!ismissing, biases) <= 8
    focus_steady = length(runs) > 20 && any(!ismissing, biases)
    history_indices = focus_steady ? findall(!ismissing, biases) : collect(eachindex(runs))
    panels = [base_plot(; title, ylabel)
              for (title, ylabel) in
                  (("E  Density error\nMarker = final; cap = maximum; exact target = 0",
                    "Relative L₂ error [%], log scale"),
                   ("F  Pressure error\nMarker = final; cap = maximum; exact target = 0",
                    "Relative L₂ error [%], log scale"),
                   ("G  Plate deflection vs time [s]\n" *
                    (normalized_deflection ? "Steady reference = 1" : "Black = analytical") *
                    (show_mean_labels ? "; legend: mean signed error" :
                     "; shading: averaging interval"),
                    normalized_deflection ? "Deflection / analytical deflection" :
                    "Plate deflection [μm]"),
                   ("H  Conservation, EOS & declaration order\nRegression limit = 1; exact zeros on bottom row",
                    "Residual / its regression bound"))]
    for (index, final, maximum_) in
        ((1, :density_error, :max_density_error), (2, :pressure_error, :max_pressure_error))
        values = logvalues(100 .*
                           vcat(getproperty.(runs, final), getproperty.(runs, maximum_)))
        lower, body_max = minimum(values) / 2.8, maximum(values)
        grouped_run_axis!(panels[index], runs, groups, lower, 5body_max,
                          percentage_ticks(lower, 2body_max))
    end
    seen = Set{String}()
    differences = [order_difference(run, runs) for run in runs]
    diagnostic_data = diagnostic_scaling(runs, differences)
    grouped_run_axis!(panels[4], runs, groups, diagnostic_data.zero_row / 2,
                      diagnostic_data.upper, diagnostic_data.ticks)
    hspan!(panels[4], [diagnostic_data.zero_row / 2, sqrt(10) * diagnostic_data.zero_row];
           color="#94a3b8", alpha=0.07, label="")
    hline!(panels[4], [sqrt(10) * diagnostic_data.zero_row];
           color="#b8c2ce", linestyle=:dot, linewidth=0.8, label="")
    longest_time = maximum(run -> run.final_time, runs)
    for (index, run) in enumerate(runs)
        marker = MARKERS[method_index(run.correction)]
        for (panel, final, maximum_) in
            ((panels[1], run.density_error, run.max_density_error),
             (panels[2], run.pressure_error, run.max_pressure_error))
            plot!(panel, [index, index], logvalues(100 .* [final, maximum_]);
                  color=color(run.correction), linewidth=1.8, label="")
            plot!(panel, [index - 0.18, index + 0.18], logvalues(fill(100maximum_, 2));
                  color=color(run.correction), linewidth=2, label="")
            scatter!(panel, [index], logvalues([100final]); color=color(run.correction),
                     marker, markersize=5, markerstrokecolor=color(run.correction),
                     markerstrokewidth=1.2,
                     label=!(run.correction in seen) ?
                           label(run.correction) : "")
            if final > 0 && run.final_time == longest_time &&
               run.correction in ("none", "mixed") && !run.reverse_order
                annotate!(panel, index, 100final * 0.68,
                          text(@sprintf("%.3g", 100final), 9, color(run.correction)))
            end
        end
        push!(seen, run.correction)
        for (diagnostic, residual) in zip(DIAGNOSTICS, diagnostic_data.values[index])
            ismissing(residual) && continue
            marker_, _, diagnostic_color, _ = diagnostic
            scatter!(panels[4], [index],
                     [iszero(residual) ? diagnostic_data.zero_row : residual];
                     color=diagnostic_color,
                     marker=marker_, markersize=5.5,
                     markercolor=marker_ == :diamond ? :white : diagnostic_color,
                     markerstrokecolor=diagnostic_color, markerstrokewidth=1.3, label="")
        end
    end
    plot!(panels[1]; legend=:outerbottom, legend_columns=legend_columns_for(panels[1]),
          legendfontsize=9)
    plot!(panels[2]; legend=:outerbottom, legend_columns=legend_columns_for(panels[2]),
          legendfontsize=9)
    for (marker, name, diagnostic_color, _) in DIAGNOSTICS
        scatter!(panels[4], Float64[], Float64[]; color=diagnostic_color, marker,
                 markersize=5.5,
                 markercolor=marker == :diamond ? :white : diagnostic_color,
                 markerstrokecolor=diagnostic_color, markerstrokewidth=1.3,
                 label=name)
    end
    hline!(panels[4], [1.0]; color=REFERENCE_COLOR, linestyle=:dash, linewidth=1.4,
           label="Regression limit")
    plot!(panels[4]; legend=:outerbottom, legend_columns=2,
          legendfontsize=9)
    resolutions = sort(unique(getproperty.(runs, :resolution)))
    legend_keys = Set{Tuple{String, Int}}()
    intervals = Set{Tuple{Float64, Float64}}()
    for index in history_indices
        run = runs[index]
        style = history_style(run, resolutions)
        marker = MARKERS[method_index(run.correction)]
        deflection = series(run.data, "plate_deflection")
        values = normalized_deflection ? deflection["values"] ./ run.reference :
                 1e6 .* deflection["values"]
        short = show_mean_labels && ismissing(biases[index])
        plot!(panels[3], deflection["time"], values;
              color=short ? "#999999" : color(run.correction), linestyle=style.linestyle,
              linewidth=short ? 1.2 : 2.4,
              alpha=short ? 0.45 : 1.0,
              label="")
        stride = max(3, cld(length(deflection["time"]), 7))
        phase = mod(method_index(run.correction) + style.class_index - 2, stride) + 1
        indices = unique(vcat(collect(phase:stride:length(deflection["time"])),
                              length(deflection["time"])))
        if !short
            scatter!(panels[3], deflection["time"][indices], values[indices];
                     color=color(run.correction), marker, markersize=5,
                     markercolor=marker_face(marker, color(run.correction),
                                             run.reverse_order),
                     markerstrokecolor=color(run.correction), markerstrokewidth=1.2,
                     label="")
        end
        key = (run.method, run.resolution)
        push!(legend_keys, key)
        interval = get(run.parameters, "averaging_interval", [])
        if !isempty(interval) && first(interval) < run.final_time
            push!(intervals,
                  (Float64(first(interval)), min(Float64(last(interval)), run.final_time)))
        end
    end
    if show_mean_labels
        for (index, run) in enumerate(runs)
            ismissing(biases[index]) && continue
            style = history_style(run, resolutions)
            method = run.correction == "none" ? "raw" :
                     run.correction == "mixed" ? "mixed" : label(run.correction)
            order = run.reverse_order ? " r" : ""
            bias = @sprintf("%+.2f%%", 100biases[index])
            plot!(panels[3], Float64[], Float64[]; color=color(run.correction),
                  linestyle=style.linestyle,
                  marker=MARKERS[method_index(run.correction)], markersize=5,
                  label="$(uppercase(run.method)) n$(run.resolution) · $method$order ($bias)")
        end
        if !focus_steady && any(ismissing, biases)
            plot!(panels[3], Float64[], Float64[]; color=REFERENCE_COLOR, linewidth=1.2,
                  alpha=0.6, label="Short controls")
        end
    else
        for method in METHODS
            method in seen || continue
            plot!(panels[3], Float64[], Float64[]; color=color(method),
                  marker=MARKERS[method_index(method)], markersize=5,
                  markerstrokecolor=color(method), label=label(method))
        end
        for (method, resolution) in sort(collect(legend_keys))
            style = history_style((; method, resolution, reverse_order=false), resolutions)
            plot!(panels[3], Float64[], Float64[]; color="#202020",
                  linestyle=style.linestyle,
                  label=uppercase(method) * " n=$resolution")
        end
    end
    for interval in sort(collect(intervals))
        vspan!(panels[3], collect(interval); color="#64748b", alpha=0.05, label="")
    end
    physical_references = unique([1e6run.reference
                                  for run in runs if !isnothing(run.reference)])
    references = normalized_deflection ? [1.0] : physical_references
    for reference in references
        hline!(panels[3], [reference]; color="#202020", linestyle=:dot, linewidth=1.8,
               label=normalized_deflection ?
                     (length(physical_references) == 1 ?
                      "Analytical: 1 = $(number(only(physical_references))) μm" :
                      "Analytical (=1 per run)") :
                     "Analytical $(number(reference)) μm")
    end
    if normalized_deflection
        values = vcat([series(runs[index].data, "plate_deflection")["values"] ./
                       runs[index].reference
                       for index in history_indices]...)
        lower, upper = min(0.0, minimum(values)), max(1.0, maximum(values))
        if lower >= -0.5 && upper <= 2.5
            lower, upper = floor(2lower) / 2, ceil(2upper * 1.04) / 2
            plot!(panels[3]; ylims=(lower - 0.04(upper-lower), upper + 0.02(upper-lower)),
                  yticks=collect(lower:0.5:upper))
        end
        for interval in intervals
            annotate!(panels[3], sum(interval)/2, upper * 0.92,
                      text("Mean window\n$(number(first(interval)))–$(number(last(interval))) s",
                           8, REFERENCE_COLOR))
        end
    end
    plot!(panels[3];
          xlims=(minimum(run -> first(series(run.data, "plate_deflection")["time"]), runs),
                 maximum(run -> run.final_time, runs)),
          legend=:outerbottom,
          legend_columns=show_mean_labels ? 2 : legend_columns_for(panels[3]),
          legendfontsize=9)
    if use_error_matrix(runs) || length(groups) > length(run_groups(runs))
        late = use_late_errors(runs)
        statistic = late ? "late mean" : "final"
        panels[1] = error_matrix_plot(runs, groups,
                                      late ? :late_density_error : :density_error,
                                      :max_density_error,
                                      "E  Density errors: density × gradient methods\nCells: $statistic / maximum [%]")
        panels[2] = error_matrix_plot(runs, groups,
                                      late ? :late_pressure_error : :pressure_error,
                                      :max_pressure_error,
                                      "F  Pressure errors: density × gradient methods\nCells: $statistic / maximum [%]")
        panels[4] = grouped_diagnostic_plot(runs, groups, diagnostic_data)
    end
    use_deflection_envelopes(runs) &&
        (panels[3] = deflection_envelope_plot(runs, groups))
    samples, _ = refinement_samples(runs)
    use_refinement_plot(samples) &&
        (panels[3] = plate_refinement_plot(samples, structure; groups))
    return plot(panels...; layout=(2, 2), size=(1600, 1100)), differences
end

const STYLE = """
    *{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#172b42;font:14px/1.45 system-ui,sans-serif}
    main{max-width:1500px;margin:auto;padding:18px 24px}h1{font-size:25px;margin:4px 0}h2{font-size:19px;margin:0 0 10px}
    .intro p{margin:6px 0 18px;color:#52657b}.figure{background:white;border:1px solid #dce4ec;border-radius:10px;padding:16px;margin:16px 0}
    img{display:block;width:100%;height:auto}.caption{font-size:12px;color:#475569;margin:8px 0 0}a{color:#006ca8}
    details{margin:12px 0}summary{cursor:pointer;font-weight:600}.links{display:flex;flex-wrap:wrap;gap:8px 20px}pre{overflow:auto;background:white;padding:12px;font-size:11px}
    @media(max-width:700px){main{padding:8px}.figure{padding:8px}h1{font-size:21px}}
    @page{size:A4 landscape;margin:9mm}
    @media print{body{background:white}main{max-width:none;padding:0}.intro,.data{display:none}.figure{border:0;border-radius:0;padding:0;margin:0;break-after:page}
    .figure:last-of-type{break-after:auto}h2{font-size:14px;margin:0 0 2mm}img{max-height:155mm;object-fit:contain}.caption{font-size:8px;line-height:1.3;margin:1mm 0 0}}
    """

function build_report(; operator_file=nothing, hydrostatic_directories=(),
                      structure_file=nothing, operators=nothing, diagnostic_directories=(),
                      expanded=false, _runs=nothing, _sources=nothing)
    expanded && return build_expanded_report(; operator_file, hydrostatic_directories,
                                 structure_file,
                                 operators, diagnostic_directories)
    data = isnothing(operator_file) ? DataFrame() : CSV.read(operator_file, DataFrame)
    !isnothing(operators) &&
        (data = filter(row -> row.operator in string.(operators), data))
    runs,
    sources = isnothing(_runs) ? load_runs(hydrostatic_directories) : (_runs, _sources)
    statuses = case_status_table(sources, runs)
    legacy_density_policy = any(row -> row.method == "wcsph" &&
                                       row.density_mode == "summation" &&
                                       HydrostaticCorrectionMethods.definition(row.correction).shepard &&
                                       get(sources[row.source].parameters,
                                           "density_boundary_policy",
                                           "lagged-boundary-density") !=
                                       "raw-pressure-predictor",
                                eachrow(statuses))
    planned_runs = isempty(statuses) ? length(runs) : nrow(statuses)
    incomplete_runs = isempty(statuses) ? 0 : count(!=("completed"), statuses.status)
    structure = isnothing(structure_file) ? DataFrame() :
                CSV.read(structure_file, DataFrame)
    diagnostics = load_diagnostics(diagnostic_directories)
    samples, refinement_protocols = refinement_samples(runs)
    refinement = HydrostaticRefinementMetrics.observed_refinement(samples)
    isempty(data) && isempty(runs) && isempty(sources) &&
        throw(ArgumentError("provide operator CSV data or hydrostatic histories"))
    figures = Dict{String, Plots.Plot}()
    envelopes = NamedTuple[]
    differences = Union{Missing, Float64}[]
    body = IOBuffer()
    if !isempty(data)
        figure, envelopes = operator_overview(data)
        figures["operators"] = figure
        print(body,
              "<section class='figure' id='operators'><h2>Figure 1 · Operator consistency vs resolution N (Δx = 1/N)</h2>",
              image(figure,
                    "Four panels: interpolation and density, gradients, pressure acceleration, constant-pressure response"),
              "<p class='caption'><strong>Solid = boundary; dashed = symmetric interior; closed interior symbols are hollow.</strong>
              A uses quantity-specific colors; B–D use correction colors/codes. Curves are maxima and bands span every variant.
              Near-coincident maxima are labeled. Dotted gray slopes are vertically offset local-theory guides, not error bounds or full-solver orders;
              production one-pass density can retain a floor. D's interior target is zero acceleration;
              its 10⁻⁷ regression bound applies only to the interior null-response checks.
              All ", nrow(data), " measurements across ", length(unique(data.operator)),
              " operators are represented.</p></section>")
    end
    if !isempty(sources)
        figure, differences = coupled_overview(runs; structure, sources)
        figures["coupled"] = figure
        matrix_view = use_error_matrix(runs) ||
                      length(run_groups(runs; sources)) > length(run_groups(runs))
        wide_matrix = length(run_groups(runs; sources)) > 8
        statistic = use_late_errors(runs) ? "late-window mean" : "final error"
        windows = unique(get(run.parameters, "averaging_interval", []) for run in runs)
        isempty(windows) && (windows = unique(get(info.parameters, "averaging_interval", [])
                          for info in sources))
        window_caption = use_late_errors(runs) ?
                         " Mean windows: " *
                         join(["$(number(first(window)))–$(number(last(window))) s"
                               for window in windows], ", ") * "." : ""
        error_caption = matrix_view ?
                        "E/F: $statistic [%], then full-history max; logarithmic colors. Repeats use maxima; — = redundant/not run. Compare methods within each scheme/density/resolution." *
                        window_caption :
                        "E/F: symbol = final error, cap = history maximum; numbers label the longest-duration uncorrected/mixed pairs."
        legacy_density_policy &&
            (error_caption *= " Archived WCSPH summation/Shepard results predate the raw-pressure boundary predictor.")
        if wide_matrix
            error_caption = "E/F split cells: left = $statistic, right = full-history maximum [%], using the shared logarithmic color scale. Exact values are embedded." *
                            window_caption
        end
        if incomplete_runs > 0
            error_caption *= wide_matrix ?
                             " × = failed; •/… = running/pending; no error value is imputed." :
                             " FAIL = failed trajectory; RUN/WAIT = unfinished. These cases contribute no error value."
        end
        if matrix_view && !wide_matrix && length(run_groups(runs)) > 4 &&
           length(unique(getproperty.(runs, :correction))) > 6
            error_caption *= " Dense labels use two significant digits; full-precision data are embedded."
        end
        reset_intervals = sort(unique([get(run.parameters, "reinitialization_dt", 0.01)
                                       for run in runs]))
        method_caption = "Methods cross Shepard off/on with gradient None, Kernel (kernel normalization), Gradient (first-moment matrix), Blend (λ=0.5), Mixed (kernel + matrix). Shepard on: summation filter each RHS; continuity resets every " *
                         join(number.(reset_intervals), "/") *
                         " s."
        any(row -> row.correction == "continuity_reinit", eachrow(statuses)) &&
            (method_caption *= " The extra continuity row uses Shepard + Mixed.")
        stress_caption = any(run -> run.method == "wcsph" && density_mode_code(run) == "Σ",
                             runs) ?
                         "WCSPH Σ stress rows start with EOS-amplified summation-density error; their large pressure errors are retained." :
                         ""
        mean_count = count(run -> !ismissing(mean_deflection_bias(run)), runs)
        history_caption = use_refinement_plot(samples) ?
                          "G compares only methods completed at every shown level: line = worst error, band = common-method range, dashed = isolated TLSPH. Legend gives common-method counts. No solver order is imposed; missing levels are not imputed. Histories and observed orders are embedded; negative orders retain deterioration." :
                          use_deflection_envelopes(runs) ?
                          "G: solid = uncorrected, dashed = mixed; bands span all forward correction methods, including continuity controls; legend identifies scheme/resolution. Shading marks the mean window." :
                          matrix_view && mean_count > 0 ?
                          "G shows runs reaching the mean window; shorter histories remain embedded and are covered by E/F/H." :
                          0 < mean_count <= 8 && mean_count < length(runs) ?
                          "G retains all histories, with short controls shown in gray." :
                          "G retains all supplied histories."
        residual_caption = matrix_view ?
                           "H shows maxima and ranges over all correction cases in each configuration." :
                           "H shows each run's checks."
        order_exceedances = unique(["$(uppercase(run.method)) n=$(run.resolution)"
                                    for (index, run) in enumerate(runs)
                                    if !ismissing(differences[index]) &&
                                       differences[index] > 1e-6])
        if !isempty(order_exceedances)
            residual_caption *= " Order-bound exceedance: " *
                                join(order_exceedances, ", ") * "."
        end
        unavailable_order = isempty(statuses) ? DataFrame() :
                            filter(row -> row.reverse_order && row.status != "completed",
                                   statuses)
        if !isempty(unavailable_order)
            residual_caption *= " $(nrow(unavailable_order)) order controls are unfinished/failed, not zero residuals."
        end
        structure_caption = if isempty(structure)
            "Plate deflection also contains structural spatial-discretization error."
        else
            biases = join(["n$(row.resolution): " *
                           @sprintf("%+.2f%%", 100row.relative_bias)
                           for row in eachrow(structure)], "; ")
            "Isolated TLSPH signed biases ($biases) use prescribed load, no fluid or time integration; they expose structural spatial error."
        end
        progress = incomplete_runs == 0 ? "" :
                   " · $(length(runs))/$planned_runs trajectories completed"
        print(body,
              "<section class='figure' id='coupled'><h2>Figure 2 · Coupled hydrostatic benchmark",
              progress, "</h2>",
              image(figure,
                    "Four panels: density error, pressure error, plate deflection, implementation residuals"),
              "<p class='caption'><strong>Axes identify the gradient and density corrections, scheme, base density model, plate resolution and final time.</strong>
                ", error_caption, " ", method_caption, " ", stress_caption,
              "
References use dp/d(depth)=ρg, translated surface, Cole EOS for WCSPH and constant EDAC density.
Plate errors and histories use the analytical plate deflection as reference.
  ", history_caption, " All ", length(runs),
              " completed histories and all case statuses are embedded.
", structure_caption, "
", residual_caption,
              " Bounds: force 2e-12, EOS/reinit 1e-12, order 1e-6; normalized limit = 1. Exact zeros use the bottom row.
Force includes clamped reactions; order compares density, pressure and deflection. Reinit checks evolved volumes and immediate pressure handling.
Raw values and experiment settings are embedded below.</p></section>")
    end
    print(body,
          "<details class='data'><summary>Source data and experiment settings (embedded, offline)</summary><div class='links'>")
    if !isempty(data)
        csv = IOBuffer()
        CSV.write(csv, data)
        print(body,
              download(take!(csv), "correction_convergence.csv",
                       "All operator measurements"; mime="text/csv"))
    end
    if !isempty(runs)
        status_csv = IOBuffer()
        CSV.write(status_csv, statuses)
        print(body,
              download(take!(status_csv), "hydrostatic_case_status.csv",
                       "Every planned case: completion, failures and pending work";
                       mime="text/csv"))
        raw = [Dict("source" => run.directory, "case" => run.name,
                    "parameters" => run.parameters,
                    "history" => run.data) for run in runs]
        print(body,
              download(JSON.json(raw), "hydrostatic_correction_histories.json",
                       "All coupled histories + metadata"))
        definitions = Dict(string(case) => HydrostaticCorrectionMethods.definition(case)
                           for case in HydrostaticCorrectionMethods.DEFAULT_CASES)
        print(body,
              download(JSON.json(definitions), "hydrostatic_method_definitions.json",
                       "Density/gradient method definitions"))
        if !isempty(refinement)
            csv = IOBuffer()
            CSV.write(csv, refinement)
            print(body,
                  download(take!(csv), "hydrostatic_refinement.csv",
                           "Matched inter-level errors and observed orders";
                           mime="text/csv"),
                  download(JSON.json(refinement_protocols),
                           "hydrostatic_refinement_protocols.json",
                           "Refinement protocol keys"))
        end
        for (source, info) in enumerate(sources)
            for filename in
                ("hydrostatic_corrections.csv", "hydrostatic_correction_order.csv",
                 "hydrostatic_correction_comparisons.csv")
                path = joinpath(info.directory, filename)
                isfile(path) && print(body,
                      download(read(path), "source-$(source)_$filename",
                               "Source $source · $filename"; mime="text/csv"))
            end
        end
    end
    if !isempty(structure)
        print(body,
              download(read(structure_file), "hydrostatic_structure.csv",
                       "Isolated-plate refinement control"; mime="text/csv"))
        parameters_file = joinpath(dirname(structure_file), "parameters.json")
        if isfile(parameters_file)
            print(body,
                  download(read(parameters_file), "hydrostatic_structure_parameters.json",
                           "Isolated-plate experiment settings"))
        end
    end
    if !isempty(diagnostics)
        print(body,
              download(JSON.json(diagnostics), "hydrostatic_stability_diagnostics.json",
                       "Tagged stability controls: traces, state snapshots and logs"))
        summaries = [(; name=diagnostic.name, grid=diagnostic.variant["grid"],
                      target_time=get(diagnostic.variant, "final_time", missing),
                      stop_time=get(diagnostic.state, "integrator_time", missing),
                      diagnostic.outcome,
                      previous_max_speed=get(get(diagnostic.state, "previous_state",
                                                 Dict()), "max_speed", missing),
                      previous_min_pressure_Pa=get(get(diagnostic.state, "previous_state",
                                                       Dict()), "min_pressure", missing),
                      previous_escaped_side_particles=get(get(diagnostic.state,
                                                              "previous_state", Dict()),
                                                          "outside_nominal_sides", missing))
                     for diagnostic in diagnostics]
        csv = IOBuffer()
        CSV.write(csv, DataFrame(summaries))
        print(body,
              download(take!(csv), "hydrostatic_stability_diagnostics.csv",
                       "Stability-control summary (excluded from primary statistics)";
                       mime="text/csv"))
    end
    if legacy_density_policy
        diagnosis = joinpath(@__DIR__, "..", "hydrostatic_water_column_2d",
                             "SUMMATION_DIAGNOSIS.md")
        isfile(diagnosis) && print(body,
              download(read(diagnosis), "WCSPH_SUMMATION_DIAGNOSIS.md",
                       "WCSPH summation: startup bias and boundary-staging follow-up";
                       mime="text/markdown"))
    end
    print(body, "</div>")
    for (source, info) in enumerate(sources)
        print(body, "<details><summary>s", source, " · ",
              escape_html(basename(info.directory)), " · ", info.count,
              " histories", isempty(info.parameters) ? " · partial parameter record" : "",
              "</summary><pre>",
              escape_html(info.directory), "\n", escape_html(JSON.json(info.parameters, 2)),
              "</pre></details>")
    end
    print(body, "</details>")
    html = "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>" *
           "<title>SPH correction validation</title><style>" * STYLE *
           "</style></head><body><main>" *
           "<header class='intro'><h1>SPH correction validation</h1><p>Two multipanel figures · eight panels · theory and measurements.
           Print in landscape for a two-page report. All source data are embedded below.</p></header>" *
           String(take!(body)) * "</main></body></html>"
    return (; html, figures, operator_envelopes=envelopes, operator_rows=nrow(data),
            coupled_runs=length(runs), structural_rows=nrow(structure),
            planned_coupled_runs=planned_runs, incomplete_coupled_runs=incomplete_runs,
            case_status=statuses,
            diagnostic_runs=length(diagnostics),
            refinement, refinement_protocols,
            coupled_error_statistic=isempty(runs) || !use_late_errors(runs) ? "final" :
                                    "late mean",
            order_differences=differences)
end

function write_report(filename, report)
    mkpath(dirname(filename))
    write(filename, report.html)
    return filename
end

include("expanded_report.jl")

end # module
