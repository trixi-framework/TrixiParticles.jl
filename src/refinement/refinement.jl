struct ParticleRefinement{ELTYPE, RC}
    spacing_ratio           :: ELTYPE # Ratio between spacings of different refinement bands
    min_spacing             :: ELTYPE # The smallest particle spacing in the simulation
    max_spacing             :: ELTYPE # The largest particle spacing in the simulation
    smoothing_length_factor :: ELTYPE # Ratio between smoothing length and particle spacing
    reference_density       :: ELTYPE # Used to compute the particle spacing from the mass
    criteria                :: RC     # Tuple of refinement criteria
end

function ParticleRefinement(; spacing_ratio, min_spacing, max_spacing, reference_density,
                            smoothing_length_factor=1.2,
                            refinement_criteria=SpatialRefinementCriterion())
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    return ParticleRefinement(spacing_ratio, min_spacing, max_spacing,
                              smoothing_length_factor, reference_density,
                              refinement_criteria)
end

create_cache_refinement(initial_condition, ::Nothing) = (;)

function create_cache_refinement(initial_condition, refinement)
    (; reference_density) = refinement

    # The target spacing persists between refinement calls (see `update_target_spacing!`).
    # Initialize it with the spacing that corresponds to the initial mass.
    target_spacing = (initial_condition.mass ./ reference_density) .^
                     (1 / ndims(initial_condition))
    target_spacing_tmp = similar(target_spacing)

    # Particles whose target spacing was set by a criterion in this refinement call.
    # Use `zeros(Bool, n)` instead of `falses(n)`, since writing to a `BitVector`
    # is not thread-safe.
    is_fixed = zeros(Bool, nparticles(initial_condition))

    return (; target_spacing, target_spacing_tmp, is_fixed)
end

resize_cache_refinement!(cache, ::Nothing, n, spacing) = cache

# The new particles get the target spacing `spacing`
function resize_cache_refinement!(cache, refinement, n, spacing)
    (; target_spacing, target_spacing_tmp, is_fixed) = cache

    resize_and_fill!(target_spacing, n, spacing)
    resize_and_fill!(target_spacing_tmp, n, spacing)
    resize_and_fill!(is_fixed, n, false)

    return cache
end

include("refinement_criteria.jl")
include("spacing.jl")
include("resize.jl")
