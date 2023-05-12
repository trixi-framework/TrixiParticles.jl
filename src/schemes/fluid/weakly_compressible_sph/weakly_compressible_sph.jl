"""
    WeaklyCompressibleSPH(state_equation)

Weakly compressible SPH introduced by (Monaghan, 1994). This formulation relies on a stiff
equation of state (see  [`StateEquationCole`](@ref)) that generates large pressure changes
for small density variations.

## References:
- Joseph J. Monaghan. "Simulating Free Surface Flows in SPH".
  In: Journal of Computational Physics 110 (1994), pages 399-406.
  [doi: 10.1006/jcph.1994.1034](https://doi.org/10.1006/jcph.1994.1034)
"""
struct WeaklyCompressibleSPH{DC, ELTYPE <: Real, K, SE, V, C}
    density_calculator :: DC
    smoothing_kernel   :: K
    smoothing_length   :: ELTYPE
    state_equation     :: SE
    viscosity          :: V
    pressure           :: Vector{ELTYPE} # [particle]
    cache              :: C

    function WeaklyCompressibleSPH(density_calculator, smoothing_kernel, smoothing_length,
                                   state_equation, viscosity)
        ELTYPE = eltype(smoothing_length)
        pressure = Vector{ELTYPE}(undef, 0)

        create_cache(pressure, density_calculator)

        new{typeof(density_calculator), typeof(smoothing_kernel), typeof(state_equation),
            typeof(viscosity), typeof(cache), ELTYPE}(density_calculator, smoothing_kernel,
                                                      smoothing_length, state_equation,
                                                      viscosity, pressure, cache)
    end
end

function create_cache(initial_density, ::SummationDensity)
    density = similar(initial_density)

    return (; density)
end

create_cache(initial_density, density_calculator) = (;)

# TODO_refactor: Base.show

function initialize!(scheme::WeaklyCompressibleSPH, container, neighborhood_search)
    NDIMS = ndims(container)

    if ndims(scheme.smoothing_kernel) != NDIMS
        throw(ArgumentError("smoothing kernel dimensionality must be $NDIMS for a $(NDIMS)D problem"))
    end

    resize!(scheme.pressure, nparticles(container))
    initialize_cache!(scheme, nparticles(container))
end

function initialize_cache!(scheme::WeaklyCompressibleSPH{SummationDensity}, n_particles)
    resize!(scheme.cache.density, n_particles)
end

function initialize_cache!(scheme::WeaklyCompressibleSPH, n_particles)
    # TODO_refactor
    # throw(ArgumentError("`initial_density` must be a vector of length $(n_particles) when using `ContinuityDensity`"))
end

@inline function v_nvariables(container, ::WeaklyCompressibleSPH{ContinuityDensity})
    # Density is integrated with `ContinuityDensity`
    return ndims(container) + 1
end

@inline function v_nvariables(container, ::WeaklyCompressibleSPH)
    return ndims(container)
end

# include("density_calculators.jl") TODO comment back in
include("state_equations.jl")
include("viscosity.jl")
include("rhs.jl")
