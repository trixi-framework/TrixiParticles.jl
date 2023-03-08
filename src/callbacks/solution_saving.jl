"""
    SolutionSavingCallback(; saveat, custom_quantities...)

Callback to save the solution at specific times specified by `saveat`.
The return value is the tuple `saved_values, callback`.

Additional user-defined quantities can be saved by passing functions
as keyword arguments, which map `(v, u, t, container)` to an `Array` where
the columns represent the particles in the same order as in `u`.
Note that this array must be allocated and cannot be a view to `u`.

# Examples
```julia
saved_values, saving_callback = SolutionSavingCallback(saveat=0.0:0.02:1.0,
                                                       index=(v, u, t, container) -> Pixie.eachparticle(container))
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

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:SavingAffect{<:ExtractQuantities}})
    @nospecialize cb # reduce precompilation time

    print(io, "SolutionSavingCallback")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:SavingAffect{<:ExtractQuantities}})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        summary_box(io, "SolutionSavingCallback")
    end
end

function (extract_quantities::ExtractQuantities)(u_tmp, t, integrator)
    semi = integrator.p
    @unpack particle_containers = semi

    # The SavingCallback does not insert tstops, so `u` had to be interpolated.
    # However, only `u` (or `u` and `v`) has been interpolated, but not the containers.
    # To upate the containers, we have to call `kick!` with the correct `u` again (`u_tmp`).
    # We need to pass some cache as `dv` to `kick!`.
    # We can use `first(get_tmp_cache(integrator))` for this.
    # However, `u_tmp` is either a reference to the actual `integrator.u` or a reference
    # to the cache `first(get_tmp_cache(integrator))`.
    # Thus, when we call `kick!` with `first(get_tmp_cache(integrator))` as `dv`,
    # we might change the contents of `u_tmp`.
    # For this reason, we copy the current `u_tmp` before we call `kick!`.
    v_ode = copy(u_tmp.x[1])
    u_ode = copy(u_tmp.x[2])

    # Call `kick!` to update the containers (pressures, etc.).
    # We only need the `v`-part of the cache here.
    v_cache = first(get_tmp_cache(integrator)).x[1]
    kick!(v_cache, v_ode, u_ode, semi, t)

    result = Dict{Symbol, Dict{Symbol, Array{Float32}}}()

    for (container_index, container) in pairs(particle_containers)
        v = wrap_v(v_ode, container_index, container, semi)
        u = wrap_u(u_ode, container_index, container, semi)
        write_result!(result, v, u, t, container, extract_quantities)
    end

    return result
end

function write_result!(result, v, u, t, container, extract_quantities)
    @unpack custom_quantities = extract_quantities

    name, value = extract_quantities(v, u, container)

    # Extract custom quantities for this container
    for (key, func) in custom_quantities
        value[key] = func(v, u, t, container)
    end

    # Determine name for this container
    i = 1
    while Symbol("$(name)_$i") in keys(result)
        i += 1
    end

    result[Symbol("$(name)_$i")] = value

    return result
end

function (extract_quantities::ExtractQuantities)(v, u, container::FluidParticleContainer)
    @unpack density_calculator, cache, store_options, surface_tension = container

    NDIMS = ndims(container)
    result = Dict{Symbol, Array{Float32}}(
                                          # Note that we have to allocate here and can't use views.
                                          # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
                                          :coordinates => u[1:NDIMS, :],
                                          :velocity => v[1:NDIMS, :],
                                          :pressure => copy(container.pressure))

    if surface_tension isa SurfaceTensionAkinci
        result[:surface_normal] = copy(cache.surface_normal)
    end

    if store_options isa StoreAll
        result[:a_surface_tension] = copy(cache.a_surface_tension)
        result[:a_viscosity] = copy(cache.a_viscosity)
        result[:a_pressure] = copy(cache.a_pressure)
    end

    extract_density!(result, v, cache, density_calculator, container)

    return "fluid", result
end

function (extract_quantities::ExtractQuantities)(v, u, container::SolidParticleContainer)
    n_fixed_particles = nparticles(container) - n_moving_particles(container)
    result = Dict{Symbol, Array{Float32}}(
                                          # Note that we have to allocate here and can't use views.
                                          # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
                                          :coordinates => copy(container.current_coordinates),
                                          :velocity => hcat(v[1:ndims(container), :],
                                                            zeros(ndims(container),
                                                                  n_fixed_particles)),
                                          :material_density => container.material_density)

    return "solid", result
end

function (extract_quantities::ExtractQuantities)(v, u, container::BoundaryParticleContainer)
    @unpack boundary_model = container

    extract_quantities(v, u, container, boundary_model)
end

function (extract_quantities::ExtractQuantities)(v, u, container::BoundaryParticleContainer,
                                                 boundary_model)
    result = Dict{Symbol, Array{Float64}}(
                                          # Note that we have to allocate here and can't use views.
                                          # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
                                          :coordinates => copy(container.initial_coordinates))

    return "boundary", result
end

function (extract_quantities::ExtractQuantities)(v, u, container::BoundaryParticleContainer,
                                                 boundary_model::BoundaryModelDummyParticles)
    result = Dict{Symbol, Array{Float64}}(
                                          # Note that we have to allocate here and can't use views.
                                          # See https://diffeq.sciml.ai/stable/features/callback_library/#saving_callback.
                                          :coordinates => copy(container.initial_coordinates),
                                          :density => [get_particle_density(particle, v,
                                                                            container)
                                                       for particle in eachparticle(container)],
                                          :pressure => copy(boundary_model.pressure))

    return "boundary", result
end

function extract_density!(result, v, cache, ::SummationDensity, container)
    result[:density] = copy(cache.density)
end

function extract_density!(result, v, cache, ::ContinuityDensity, container)
    result[:density] = v[end, :]
end
