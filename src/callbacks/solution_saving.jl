"""
    SolutionSavingCallback(; saveat, custom_quantities...)

Callback to save the solution at specific times specified by `saveat`.
The return value is the tuple `saved_values, callback`.

Additional user-defined quantities can be saved by passing functions
as keyword arguments, which map `(u, t, integrator)` to an `Array` where
the columns represent the particles in the same order as in `u`.
Note that this array must be allocated and cannot be a view to `u`.

# Examples
```julia
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1.0,
                                                       index=(u, t, integrator) -> Pixie.eachparticle(integrator.p))
```
"""
function SolutionSavingCallback(; saveat, custom_quantities...)
    ELTYPE = eltype(saveat)
    extract_quantities = ExtractQuantities(custom_quantities)

    # Create callback
    saved_values = SavedValues(ELTYPE, Dict{Symbol, Array{ELTYPE}})
    callback = SavingCallback(extract_quantities, saved_values, saveat=saveat)

    return saved_values, callback
end


struct ExtractQuantities{CQ}
    custom_quantities::CQ

    function ExtractQuantities(custom_quantities)
        new{typeof(custom_quantities)}(custom_quantities)
    end
end


function (extract_quantities::ExtractQuantities)(u_tmp, t, integrator)
    semi = integrator.p
    @unpack density_calculator, cache = semi
    @unpack custom_quantities = extract_quantities

    # The SavingCallback does not insert tstops, so u had to be interpolated.
    # However, only u has been interpolated, but not semi.cache. To compute the correct
    # cache, we have to call rhs! with the correct u again (u_tmp).
    # We need to pass some cache as du to rhs!. We can use first(get_tmp_cache(integrator))
    # for this.
    # However, u_tmp is either a reference to the actual integrator.u or a reference to the cache
    # first(get_tmp_cache(integrator)).
    # Thus, when we call rhs! with first(get_tmp_cache(integrator)) as du, we might change
    # the contents of u_tmp.
    # For this reason, we copy the current u_tmp before we call rhs!.
    # This allocation is not a problem, since we have to return allocated values anyway
    # (not a view to u_cache).
    u = copy(u_tmp)

    # Call rhs! to compute the correct cache (pressures, etc.)
    u_cache = first(get_tmp_cache(integrator))
    rhs!(u_cache, u, semi, t)

    result = Dict{Symbol, Array{Float64}}(
        # Note that we have to allocate here and can't use views.
        # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
        # However, u has already been allocated above, so we can use views to u.
        :coordinates   => extract_coordinates(u, semi),
        :velocity      => view(u, (ndims(semi)+1):(2*ndims(semi)), :)
    )

    extract_density!(result, u, cache, density_calculator, semi)
    extract_pressure!(result, cache, semi)

    for (key, func) in custom_quantities
        result[key] = func(u, t, integrator)
    end

    return result
end

function extract_coordinates(u, semi::SPHFluidSemidiscretization)
    view(u, 1:ndims(semi), :)
end

function extract_coordinates(u, semi::SPHSolidSemidiscretization)
    update_current_coordinates(u, semi)

    return copy(semi.cache.current_coordinates)
end

function extract_density!(result, u, cache, ::SummationDensity, semi)
    result[:density] = copy(cache.density)
end

function extract_density!(result, u, cache, ::ContinuityDensity, semi)
    result[:density] = view(u, 2*ndims(semi) + 1, :)
end

function extract_density!(result, u, cache, ::SummationDensity, semi::SPHSolidSemidiscretization)
    result[:density] = copy(cache.solid_density)
end

function extract_pressure!(result, cache, ::SPHFluidSemidiscretization)
    result[:pressure] = copy(cache.pressure)
end

function extract_pressure!(result, cache, ::SPHSolidSemidiscretization) end
