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


function (extract_quantities::ExtractQuantities)(u, t, integrator)
    semi = integrator.p
    @unpack density_calculator, cache = semi
    @unpack custom_quantities = extract_quantities

    result = Dict(
        # Note that these have to be allocated and no views can be used here
        :coordinates   => u[1:ndims(semi), :],
        :velocity      => u[(ndims(semi)+1):(2*ndims(semi)), :],
        :density       => extract_density(u, cache, density_calculator),
        :pressure      => copy(cache.pressure)
    )

    for (key, func) in custom_quantities
        result[key] = func(u, t, integrator)
    end

    return result
end

extract_density(u, cache, ::SummationDensity) = copy(cache.density)
extract_density(u, cache, ::ContinuityDensity) = u[end, :]
