"""
    SolutionSavingCallback(; saveat, custom_quantities...)

Callback to save the solution at specific times specified by `saveat`.
The return value is the tuple `saved_values, callback`.

Additional user-defined quantities can be saved by passing functions
as keyword arguments, which map `(u, t, container)` to an `Array` where
the columns represent the particles in the same order as in `u`.
Note that this array must be allocated and cannot be a view to `u`.

# Examples
```julia
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1.0,
                                                       index=(u, t, container) -> Pixie.eachparticle(container))
```
"""
function SolutionSavingCallback(; saveat, custom_quantities...)
    ELTYPE = eltype(saveat)
    extract_quantities = ExtractQuantities(custom_quantities)

    # Create callback
    saved_values = SavedValues(ELTYPE, Dict{Symbol, Dict{Symbol, Array{ELTYPE}}})
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
    @unpack particle_containers = semi

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
    u_ode = copy(u_tmp)

    # Call rhs! to compute the correct cache (pressures, etc.)
    u_cache = first(get_tmp_cache(integrator))
    rhs!(u_cache, u_ode, semi, t)

    result = Dict{Symbol, Dict{Symbol, Array{Float64}}}()

    for (container_index, container) in pairs(particle_containers)
        u = wrap_array(u_ode, container_index, semi)
        write_result!(result, u, t, container, extract_quantities)
    end

    return result
end


function write_result!(result, u, t, container, extract_quantities)
    @unpack custom_quantities = extract_quantities

    name, value = extract_quantities(u, container)

    # Extract custom quantities for this container
    for (key, func) in custom_quantities
        value[key] = func(u, t, container)
    end

    # Determine name for this container
    i = 1
    while Symbol("$(name)_$i") in keys(result)
        i += 1
    end

    result[Symbol("$(name)_$i")] = value

    return result
end


function (extract_quantities::ExtractQuantities)(u, container::FluidParticleContainer)
    @unpack density_calculator, cache = container

    result = Dict{Symbol, Array{Float64}}(
        # Note that we have to allocate here and can't use views.
        # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
        :coordinates    => u[1:ndims(container), :],
        :velocity       => u[(ndims(container)+1):(2*ndims(container)), :],
        :pressure       => copy(container.pressure)
    )

    extract_density!(result, u, cache, density_calculator, container)

    return "fluid", result
end

function (extract_quantities::ExtractQuantities)(u, container::SolidParticleContainer)
    n_fixed_particles = nparticles(container) - n_moving_particles(container)
    result = Dict{Symbol, Array{Float64}}(
        # Note that we have to allocate here and can't use views.
        # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
        :coordinates        => copy(container.current_coordinates),
        :velocity           => hcat(u[(ndims(container)+1):(2*ndims(container)), :], zeros(ndims(container), n_fixed_particles)),
        :material_density   => container.material_density
    )

    return "solid", result
end

function (extract_quantities::ExtractQuantities)(u, container::BoundaryParticleContainer)
    result = Dict{Symbol, Array{Float64}}(
        # Note that we have to allocate here and can't use views.
        # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
        :coordinates => copy(container.initial_coordinates)
    )

    return "moving_boundaries", result
end

function extract_density!(result, u, cache, ::SummationDensity, container)
    result[:density] = copy(cache.density)
end

function extract_density!(result, u, cache, ::ContinuityDensity, container)
    result[:density] = u[2*ndims(container) + 1, :]
end
