module HydrostaticRefinementMetrics

using DataFrames

const ERROR_FIELDS = (:late_density_error, :late_pressure_error, :deflection_error)

"""Keep the same correction methods at every available level of a matched protocol."""
function common_method_samples(samples)
    isempty(samples) && return samples
    resolutions = Set(samples.resolution)
    common = [case
              for case in unique(samples.correction)
              if Set(filter(row -> row.correction == case, samples).resolution) ==
                 resolutions]
    return filter(row -> row.correction in common, samples)
end

"""
    observed_refinement(samples)

Compare adjacent available spacings for identical scheme, correction, density model
and protocol. Repeated measurements use their maximum error. Negative orders retain
deterioration; zero/missing errors have no inferred order. These are matched full-solver
refinement changes, not prescribed spatial truncation orders.
"""
function observed_refinement(samples)
    results = NamedTuple[]
    isempty(samples) && return DataFrame()
    for group in
        groupby(samples, [:method, :correction, :density_model, :protocol]; sort=true)
        levels = groupby(sort(group, :spacing; rev=true), :resolution; sort=false)
        length(levels) >= 2 || continue
        for index in 2:length(levels), field in ERROR_FIELDS
            coarse, fine = levels[index - 1], levels[index]
            a = collect(skipmissing(coarse[!, field]))
            b = collect(skipmissing(fine[!, field]))
            (isempty(a) || isempty(b)) && continue
            coarse_error, fine_error = maximum(a), maximum(b)
            coarse_h, fine_h = first(coarse.spacing), first(fine.spacing)
            coarse_h > fine_h > 0 || error("refinement spacings must decrease")
            ratio = coarse_h / fine_h
            valid = isfinite(coarse_error) && isfinite(fine_error) &&
                    coarse_error > 0 && fine_error > 0
            push!(results,
                  (; method=first(group.method), correction=first(group.correction),
                   density_model=first(group.density_model), protocol=first(group.protocol),
                   metric=string(field), coarse_resolution=first(coarse.resolution),
                   fine_resolution=first(fine.resolution), coarse_spacing=coarse_h,
                   fine_spacing=fine_h, refinement_ratio=ratio, coarse_error, fine_error,
                   error_ratio=coarse_error > 0 ? fine_error / coarse_error : missing,
                   observed_order=valid ? log(coarse_error / fine_error) / log(ratio) :
                                  missing))
        end
    end
    return DataFrame(results)
end

end # module
