# Included inside CorrectionValidationPlots. Density views share the same raw source
# payload, so splitting the figures does not duplicate or omit experimental records.
function density_view(runs, sources, statuses, mode)
    selected_runs, selected_sources = NamedTuple[], NamedTuple[]
    for (source, info) in enumerate(sources)
        records = filter(row -> row.source == source && row.base_density_mode == mode,
                         statuses)
        isempty(records) && continue
        parameters = deepcopy(info.parameters)
        parameters["base_density_mode"] = mode
        parameters["case_status"] = Dict(row.name => Dict(string(key) => row[key]
                                                          for key in propertynames(row)
                                                          if key != :source)
                                         for row in eachrow(records))
        source_runs = filter(run -> run.source == source && base_density_mode(run) == mode,
                             runs)
        new_source = length(selected_sources) + 1
        append!(selected_runs,
                [(; run..., source=new_source, parameters) for run in source_runs])
        push!(selected_sources,
              (; directory=info.directory, parameters, count=length(source_runs)))
    end
    return selected_runs, selected_sources
end

function initialization_samples(runs, statuses)
    rows = NamedTuple[]
    for status in eachrow(statuses)
        status.reverse_order && continue
        run = findfirst(run -> run.source == status.source && run.name == status.name, runs)
        data = if !isnothing(run)
            Dict("initial_density_error" => runs[run].initial_density_error,
                 "initial_pressure_error" => runs[run].initial_pressure_error)
        else
            file = joinpath(status.job_directory, status.name, "initial_state.json")
            isfile(file) ? JSON.parsefile(file) : Dict()
        end
        for (metric, field) in
            (("density", "initial_density_error"), ("pressure", "initial_pressure_error"))
            value = get(data, field, missing)
            ismissing(value) && continue
            push!(rows,
                  (; method=status.method, density_mode=status.base_density_mode,
                   correction=status.correction, resolution=status.resolution,
                   spacing=0.05/(status.resolution-1), metric, error=value))
        end
    end
    return DataFrame(rows)
end

const CONFIGURATION_COLORS = ("#005AB5", "#C23B22", "#007A4D", "#6335A5")
const CONFIGURATION_MARKERS = (:circle, :rect, :diamond, :utriangle)

function startup_plot(samples, metric, letter)
    panel = base_plot(;
                      title="$letter  Initial $metric consistency vs Δx [m]\nLine = maximum; band = measured correction range; target = 0",
                      ylabel="Initial relative L₂ error [%], log", xscale=:log2,
                      yscale=:log10)
    isempty(samples) && return panel
    data = filter(row -> row.metric == metric, samples)
    for (index, group) in enumerate(groupby(data, [:method, :density_mode]; sort=true))
        levels = groupby(sort(group, :spacing), :resolution; sort=false)
        x = [first(level.spacing) for level in levels]
        display(value) = value <= 1e-8 ? 1e-9 : value
        lower = display.([100minimum(level.error) for level in levels])
        upper = display.([100maximum(level.error) for level in levels])
        ink = CONFIGURATION_COLORS[mod1(index, 4)]
        plot!(panel, x, upper; ribbon=(upper-lower, zeros(length(x))), fillalpha=0.1,
              color=ink, marker=CONFIGURATION_MARKERS[mod1(index, 4)], markercolor=:white,
              markerstrokecolor=ink,
              label="$(uppercase(first(group.method))) $(first(group.density_mode))")
    end
    spacings = sort(unique(data.spacing))
    upper = max(1e-6, 200maximum(data.error))
    ticks = filter(<=(upper), vcat(1e-9, 10.0 .^ collect(-6:2:ceil(Int, log10(upper)))))
    plot!(panel; legend=:outerbottom,
          legend_columns=max(1, min(2, length(groupby(data, [:method, :density_mode])))),
          legendfontsize=8,
          xticks=(spacings, number.(spacings)), ylims=(5e-10, upper),
          yticks=(ticks, vcat("≤1e-8", number.(ticks[2:end]))))
    return panel
end

function pressure_history_plot(runs, groups)
    panel = base_plot(;
                      title="O  Pressure-error transients\nUncorrected = solid; mixed = dashed; finest available n is labeled",
                      ylabel="Relative L₂ pressure error [%], log", yscale=:log10)
    configurations = unique([(group.method, group.density_mode) for group in groups])
    for (index, (method, mode)) in enumerate(configurations)
        candidates = filter(run -> run.method == method && base_density_mode(run) == mode &&
                                   run.correction == "none" && !run.reverse_order, runs)
        ink = CONFIGURATION_COLORS[mod1(index, 4)]
        if isempty(candidates)
            plot!(panel, Float64[], Float64[]; color=ink,
                  label="$(uppercase(method)) $mode · unavailable")
            continue
        end
        raw = first(sort(candidates; by=run -> run.resolution, rev=true))
        errors = series(raw.data, "hydrostatic_errors")
        sampled = findall(>(first(errors["time"])), errors["time"])
        plot!(panel, errors["time"][sampled],
              logvalues(100getindex.(errors["values"][sampled], 2));
              color=ink,
              label="$(uppercase(method)) $mode · n$(raw.resolution)")
        mixed = filter(run -> run.method == method && base_density_mode(run) == mode &&
                              run.resolution == raw.resolution &&
                              run.correction == "mixed" &&
                              !run.reverse_order, runs)
        if !isempty(mixed)
            errors = series(first(mixed).data, "hydrostatic_errors")
            sampled = findall(>(first(errors["time"])), errors["time"])
            plot!(panel, errors["time"][sampled],
                  logvalues(100getindex.(errors["values"][sampled], 2));
                  color=ink, linestyle=:dash, label="")
        end
    end
    plot!(panel; legend=:outerbottom, legend_columns=2, legendfontsize=8)
    return panel
end

function outcome_plot(statuses, groups)
    cases = [string(case)
             for case in HydrostaticCorrectionMethods.DEFAULT_CASES
             if any(==(string(case)), statuses.correction)]
    rows = [HydrostaticCorrectionMethods.axis_label(case) for case in cases]
    reverse = any(statuses.reverse_order)
    reverse && push!(rows, "Reversed order")
    labels = replace.(rows, "\n"=>" · ")
    names = ("completed", "failed", "blocked", "running", "pending", "not_planned")
    colors = ["#15803d", "#b91c1c", "#d97706", "#2563eb", "#cbd5e1", "#f8fafc"]
    values = fill(6.0, length(rows), length(groups))
    for (column, group) in enumerate(groups), row in eachindex(rows)
        reversed = reverse && row == length(rows)
        case = reversed ? "combined" : cases[row]
        found = filter(record -> record.method == group.method &&
                                 record.base_density_mode == group.density_mode &&
                                 record.resolution == group.resolution &&
                                 record.correction == case &&
                                 record.reverse_order == reversed, statuses)
        isempty(found) || (values[row, column] = findfirst(==(first(found.status)), names))
    end
    panel = base_plot(;
                      title="P  Every planned trajectory outcome\nColumns show plate layers n; completion ≠ acceptance")
    heatmap!(panel, 1:length(groups), 1:length(rows), values;
             color=cgrad(colors; categorical=true),
             clims=(0.5, 6.5), colorbar=false,
             xticks=(1:length(groups), string.([group.resolution for group in groups])),
             yticks=(1:length(rows), labels), ytickfontsize=8, xtickfontsize=8,
             yflip=true, ylims=(-0.5, length(rows)+0.5), grid=false)
    # Explicit cell polygons keep category colors aligned with their row centers
    # when the axis includes space for configuration headers above the heatmap.
    for row in axes(values, 1), column in axes(values, 2)
        cell = Shape([(column-0.5, row-0.5), (column+0.5, row-0.5),
                         (column+0.5, row+0.5), (column-0.5, row+0.5)])
        plot!(panel, cell; fillcolor=colors[Int(values[row, column])], linecolor=:white,
              linewidth=0.5, label="")
    end
    header = Shape([
                       (0.5, -0.5),
                       (length(groups)+0.5, -0.5),
                       (length(groups)+0.5, 0.5),
                       (0.5, 0.5)
                   ])
    plot!(panel, header; fillcolor=:white, linecolor=:white, label="")
    for span in configuration_spans(groups)
        annotate!(panel, (span.first+span.last)/2, 0.0,
                  text("$(uppercase(span.method))\n$(span.density_mode)", 8, "#334155"))
    end
    symbols = ("✓", "×", "—", "•", "…", "")
    for row in axes(values, 1), column in axes(values, 2)
        kind = Int(values[row, column])
        isempty(symbols[kind]) && continue
        annotate!(panel, column, row,
                  text(symbols[kind], 7, kind in (1, 2, 4) ? :white : "#334155"))
    end
    for boundary in 0.5:1:(length(groups) + 0.5)
        plot!(panel, [boundary, boundary], [0.5, length(rows)+0.5]; color=:white,
              linewidth=0.5, label="")
    end
    hline!(panel, collect(0.5:1:(length(rows) + 0.5)); color=:white, linewidth=0.5,
           label="")
    for (name, color_) in zip(names, colors)
        scatter!(panel, Float64[], Float64[]; marker=:rect, color=color_,
                 label="$(symbols[findfirst(==(name),names)]) $(replace(name, "_"=>" "))",
                 markersize=4)
    end
    plot!(panel; legend=:outerbottom, legend_columns=3, legendfontsize=8)
    return panel
end

function build_expanded_report(; operator_file=nothing, hydrostatic_directories=(),
                               structure_file=nothing, operators=nothing,
                               diagnostic_directories=())
    base = build_report(; operator_file, hydrostatic_directories, structure_file, operators,
                        diagnostic_directories, expanded=false)
    runs, sources = load_runs(hydrostatic_directories)
    isempty(sources) && return base
    statuses = case_status_table(sources, runs)
    (isnothing(operator_file) || length(unique(statuses.base_density_mode)) < 2) &&
        return base
    figures = Dict{String, Plots.Plot}()
    haskey(base.figures, "operators") && (figures["operators"] = base.figures["operators"])
    sections = String[]
    for (index, mode) in enumerate(("continuity", "summation"))
        selected, selected_sources = density_view(runs, sources, statuses, mode)
        isempty(selected_sources) && continue
        report = build_report(; structure_file, expanded=false, _runs=selected,
                              _sources=selected_sources)
        figure = report.figures["coupled"]
        letters = index == 1 ? ("E", "F", "G", "H") : ("I", "J", "K", "L")
        for (panel, letter) in zip(figure.subplots, letters)
            panel[:title] = replace(panel[:title], r"^[A-Z]"=>letter)
        end
        key = index == 1 ? "coupled" : "summation"
        figures[key] = figure
        caption = match(r"<p class='caption'>.*?</p>"s,
                        only([match.match
                              for match in
                                  eachmatch(r"<section class='figure'.*?</section>"s,
                                            report.html)])).match
        index == 2 &&
            (caption = replace(caption, r"\bE/F\b"=>"I/J", r"\bG\b"=>"K", r"\bH\b"=>"L"))
        progress = report.incomplete_coupled_runs == 0 ? "" :
                   " · $(report.coupled_runs)/$(report.planned_coupled_runs) completed"
        if report.incomplete_coupled_runs > 0 &&
           !any(status -> status in ("running", "pending"), report.case_status.status)
            failed = count(==("failed"), report.case_status.status)
            blocked = count(==("blocked"), report.case_status.status)
            progress = " · $(report.coupled_runs) completed · $failed failed · $blocked blocked"
        end
        push!(sections,
              "<section class='figure' id='$key'><h2>Figure $(index+1) · $(uppercasefirst(mode))-density hydrostatic benchmark$progress</h2>" *
              image(figure, "Four hydrostatic panels for $mode density") * caption *
              "</section>")
    end
    initial = initialization_samples(runs, statuses)
    groups = run_groups(runs; sources)
    comparison = plot(startup_plot(initial, "density", "M"),
                      startup_plot(initial, "pressure", "N"),
                      pressure_history_plot(runs, groups), outcome_plot(statuses, groups);
                      layout=(2, 2), size=(1600, 1100))
    figures["cross_density"] = comparison
    push!(sections,
          "<section class='figure' id='cross-density'><h2>Figure 4 · Cross-density startup, transients and robustness</h2>" *
          image(comparison, "Four cross-density panels") *
          "<p class='caption'>M/N use measured initial residuals, including initialized cases that later failed. Values ≤10⁻⁸% use a roundoff shelf; full values are embedded. O shows post-initial-time uncorrected/mixed histories at the labeled finest available resolution; startup is covered by M/N and unavailable curves are not imputed. P contains every planned forward trajectory and reversed-order control. Missing, failed, blocked and pending work are distinct; completion does not imply accuracy acceptance. All underlying records and protocol settings are embedded below.</p></section>")
    original = only([match.match
                     for match in
                         eachmatch(r"<section class='figure' id='coupled'.*?</section>"s,
                                   base.html)])
    html = replace(base.html, original=>join(sections),
                   "Two multipanel figures · eight panels"=>"Four multipanel figures · sixteen panels",
                   "two-page report"=>"four-page report")
    if !isempty(initial)
        csv = IOBuffer()
        CSV.write(csv, initial)
        html = replace(html,
                       "</div>"=>download(take!(csv), "hydrostatic_initialization.csv",
                                          "Measured initialization residuals";
                                          mime="text/csv") * "</div>"; count=1)
    end
    return (; base..., html, figures, initialization=initial, report_layout="expanded")
end
