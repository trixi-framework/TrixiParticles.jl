struct SolutionSavingCallback{ELTYPE, CB}
    saved_values::SavedValues{ELTYPE, Dict{String, Array{ELTYPE}}}
    callback    ::CB

    function SolutionSavingCallback(; saveat)
        ELTYPE = eltype(saveat)
        saved_values = SavedValues(ELTYPE, Dict{String, Array{ELTYPE}})
        callback = SavingCallback(extract_quantities, saved_values, saveat=saveat)

        new{ELTYPE, typeof(callback)}(saved_values, callback)
    end
end


function extract_quantities(u, t, integrator)
    semi = integrator.p
    @unpack density_calculator, cache = semi

    return Dict(
        # Note that these have to be allocated and no views can be used here
        "coordinates"   => u[1:ndims(semi), :],
        "velocity"      => u[(ndims(semi)+1):(2*ndims(semi)), :],
        "density"       => extract_density(u, cache, density_calculator),
        "pressure"      => copy(cache.pressure)
    )
end

extract_density(u, cache, ::SummationDensity) = copy(cache.density)
extract_density(u, cache, ::ContinuityDensity) = u[end, :]