struct ParticleRefinement{ELTYPE, RC, SP, SC}
    spacing_ratio           :: ELTYPE # Ratio between spacings of different refinement bands
    min_spacing             :: ELTYPE # The smallest particle spacing in the simulation
    max_spacing             :: ELTYPE # The largest particle spacing in the simulation
    smoothing_length_factor :: ELTYPE # Ratio between smoothing length and particle spacing
    reference_density       :: ELTYPE # Used to compute the particle spacing from the mass
    criteria                :: RC     # Tuple of refinement criteria
    splitting_pattern       :: SP
    max_mass_factor         :: ELTYPE # Particles above `max_mass_factor * m_ref` are split
    min_mass_factor         :: ELTYPE # Merge limit after splitting, see `merge_particles!`
    shifting_technique      :: SC     # Shifting after splitting and merging, see `shift_particles!`
end

function ParticleRefinement(; spacing_ratio, min_spacing, max_spacing, reference_density,
                            smoothing_length_factor=1.2,
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=HexagonalSplitting(),
                            max_mass_factor=1.05, min_mass_factor=0.5,
                            shifting_technique=ShiftingHaftu())
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    return ParticleRefinement(spacing_ratio, min_spacing, max_spacing,
                              smoothing_length_factor, reference_density,
                              refinement_criteria, splitting_pattern, max_mass_factor,
                              min_mass_factor, shifting_technique)
end

create_cache_refinement(initial_condition, ::Nothing) = (;)

function create_cache_refinement(initial_condition, refinement)
    (; reference_density) = refinement
    n_particles = nparticles(initial_condition)
    NDIMS = ndims(initial_condition)
    ELTYPE = eltype(initial_condition)

    # The target spacing persists between refinement calls (see `update_target_spacing!`).
    # Initialize it with the spacing that corresponds to the initial mass.
    target_spacing = (initial_condition.mass ./ reference_density) .^ (1 / NDIMS)
    target_spacing_tmp = similar(target_spacing)

    # Use `zeros(Bool, n)` instead of `falses(n)`, since writing to a `BitVector`
    # is not thread-safe.
    # Particles whose target spacing was set by a criterion in this refinement call
    is_fixed = zeros(Bool, n_particles)

    # Particles that are split in this refinement call
    split_candidate = zeros(Bool, n_particles)

    # The merge partner of each particle in the current merge round, 0 without a partner
    merge_partner = zeros(Int, n_particles)

    # Shifting after splitting and merging (see `shift_particles!`).
    # Vectors are stored as flat vectors with `NDIMS` entries per particle.
    # The shift of the current iteration
    shift = zeros(ELTYPE, NDIMS * n_particles)
    # The sum of the shifts of all iterations
    total_shift = zeros(ELTYPE, NDIMS * n_particles)
    # The shift of the current iteration divided by the smoothing length
    relative_shift = zeros(ELTYPE, n_particles)

    # The corrections of the particle properties after shifting (Eq. 37)
    velocity_correction = zeros(ELTYPE, NDIMS * n_particles)
    pressure_correction = zeros(ELTYPE, n_particles)
    density_correction = zeros(ELTYPE, n_particles)

    return (; target_spacing, target_spacing_tmp, is_fixed, split_candidate, merge_partner,
            shift, total_shift, relative_shift, velocity_correction, pressure_correction,
            density_correction)
end

resize_cache_refinement!(cache, ::Nothing, NDIMS, n, spacing) = cache

# The new particles get the target spacing `spacing`
function resize_cache_refinement!(cache, refinement, NDIMS, n, spacing)
    (; target_spacing, target_spacing_tmp, is_fixed, split_candidate, merge_partner,
     shift, total_shift, relative_shift, velocity_correction, pressure_correction,
     density_correction) = cache
    ELTYPE = eltype(target_spacing)

    resize_and_fill!(target_spacing, n, spacing)
    resize_and_fill!(target_spacing_tmp, n, spacing)
    resize_and_fill!(is_fixed, n, false)
    resize_and_fill!(split_candidate, n, false)
    resize_and_fill!(merge_partner, n, 0)

    resize_and_fill!(shift, NDIMS * n, zero(ELTYPE))
    resize_and_fill!(total_shift, NDIMS * n, zero(ELTYPE))
    resize_and_fill!(relative_shift, n, zero(ELTYPE))
    resize_and_fill!(velocity_correction, NDIMS * n, zero(ELTYPE))
    resize_and_fill!(pressure_correction, n, zero(ELTYPE))
    resize_and_fill!(density_correction, n, zero(ELTYPE))

    return cache
end

# The mass that corresponds to the target spacing of `particle`
@propagate_inbounds function reference_mass(system, particle)
    (; reference_density) = get_refinement(system)

    return reference_density * system.cache.target_spacing[particle]^ndims(system)
end

include("refinement_criteria.jl")
include("spacing.jl")
include("split_pattern.jl")
include("splitting.jl")
include("merging.jl")
include("shifting.jl")
include("resize.jl")
