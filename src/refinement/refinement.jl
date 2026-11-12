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
    n_startup_rounds        :: Int    # Number of refinement rounds in `refinement_startup!`
end

function ParticleRefinement(; spacing_ratio, min_spacing, max_spacing, reference_density,
                            smoothing_length_factor=1.2,
                            refinement_criteria=SpatialRefinementCriterion(),
                            splitting_pattern=HexagonalSplitting(),
                            max_mass_factor=1.05, min_mass_factor=0.5,
                            shifting_technique=ShiftingHaftu(),
                            n_startup_rounds=default_n_startup_rounds(spacing_ratio,
                                                                      min_spacing,
                                                                      max_spacing))
    if !(refinement_criteria isa Tuple)
        refinement_criteria = (refinement_criteria,)
    end

    return ParticleRefinement(spacing_ratio, min_spacing, max_spacing,
                              smoothing_length_factor, reference_density,
                              refinement_criteria, splitting_pattern, max_mass_factor,
                              min_mass_factor, shifting_technique, n_startup_rounds)
end

# The spacing changes by at most `spacing_ratio` per call of Alg. 1, so it takes
# `log(max_spacing / min_spacing) / log(spacing_ratio)` calls to build the full band.
# As in the reference code, use three times this number.
function default_n_startup_rounds(spacing_ratio, min_spacing, max_spacing)
    n_levels = ceil(Int, log(max_spacing / min_spacing) / log(spacing_ratio))

    return 3 * max(n_levels, 1)
end

# The largest smoothing length a particle can have with this refinement.
# Eq. 35 computes the smoothing length from the average mass of the neighbors.
# Particles above `max_mass_factor` times the reference mass are split, and the target
# spacing of the reference mass is at most `max_spacing`.
@inline function max_smoothing_length(system)
    (; max_spacing, max_mass_factor) = get_refinement(system)

    return spacing_to_smoothing_length(system,
                                       max_mass_factor^(1 / ndims(system)) * max_spacing)
end

# The smoothing length that determines the search radius of the neighborhood searches
@inline function search_smoothing_length(system)
    return search_smoothing_length(system, get_refinement(system))
end

@inline search_smoothing_length(system, ::Nothing) = initial_smoothing_length(system)

@inline function search_smoothing_length(system, refinement)
    # The initial particles can be coarser than `max_spacing`.
    # They are split in `refinement_startup!`.
    return max(initial_smoothing_length(system), max_smoothing_length(system))
end

# Warn if a neighborhood search of a fluid with refinement has a smaller search radius
# than the compact support of the largest smoothing length a particle can have.
# Such a search silently misses neighbors of the coarsest particles.
# The search radius is only chosen when the neighborhood searches are created, since
# `update_nhs!` and `resize_systems!` keep it. So it is enough to check it here.
#
# Note that this does not cover all particles. Some particles can still exceed
# `max_smoothing_length` at runtime, and nothing checks this yet:
# - Particles whose split is skipped near a boundary (see `split_particles!`) keep their
#   mass. This only exceeds the maximum if the initial particles are coarser than
#   `max_mass_factor^(1 / ndims) * max_spacing`. The search radius covers this case,
#   since it also uses the initial particle spacing (see `search_smoothing_length`).
# - Particles that enter from an open boundary get the spacing of the open boundary.
# - The smoothing length of a merged particle (Eq. 34) is only replaced by Eq. 35
#   in `shift_particles!`.
function check_refinement_search_radii(semi)
    foreach_system(semi) do system
        isnothing(get_refinement(system)) && return

        # The initial smoothing lengths are passed by the user and are not limited
        # by the refinement, so they are checked as well.
        h_max = max(max_smoothing_length(system), maximum(smoothing_length(system)))
        required_radius = compact_support(system_smoothing_kernel(system), h_max)

        foreach_system(semi) do neighbor
            # Interactions in both directions need the support of the fluid particles
            for (system_, neighbor_) in ((system, neighbor), (neighbor, system))
                nhs = get_neighborhood_search(system_, neighbor_, semi)
                search_radius = PointNeighbors.search_radius(nhs)

                # A search radius of zero is used for neighborhood searches that are never used
                if 0 < search_radius < required_radius
                    @warn "the neighborhood search of the system " *
                          "`$(nameof(typeof(system_)))` with the neighbor system " *
                          "`$(nameof(typeof(neighbor_)))` has the search radius " *
                          "$search_radius, which is smaller than the compact support " *
                          "$required_radius of the largest smoothing length possible with " *
                          "particle refinement. Neighbors of coarse particles will be missed."
                end
            end
        end
    end

    return semi
end

"""
    refinement!(semi, v_ode, u_ode)

Apply one round of particle refinement to all fluid systems with particle refinement.

!!! warning
    This can change the length of `v_ode` and `u_ode` (see `resize_systems!`).
"""
function refinement!(semi, v_ode, u_ode)
    foreach_system(semi) do system
        isnothing(get_refinement(system)) && return

        # Update the self-interaction NHS with the current coordinates
        u = wrap_u(u_ode, system, semi)
        neighborhood_search = get_neighborhood_search(system, semi)
        update_nhs!(neighborhood_search, system, system, u, u, semi)

        # Set the target spacing near boundaries and in criterion regions (Alg. 4)
        apply_refinement_criteria!(system, v_ode, u_ode, semi)

        # Update the target spacing of all other particles (Alg. 1)
        update_target_spacing!(system, v_ode, u_ode, semi)
    end

    # Merge before splitting, every particle below `m_max` is a candidate (Alg. 3).
    # Merging and splitting update all neighborhood searches.
    merge_particles!(semi, v_ode, u_ode, before_splitting=true)

    # Split the particles (Alg. 2)
    split_particles!(semi, v_ode, u_ode)

    # Merge after splitting, only particles below `m_min` are candidates (Alg. 3)
    merge_particles!(semi, v_ode, u_ode)

    # Shift the particles and correct the particle properties.
    # This also updates the smoothing lengths (Eq. 35) and all neighborhood searches.
    shift_particles!(semi, v_ode, u_ode)

    return semi
end

# Build the graded spacing field and the matching particle distribution
# before the time integration starts.
function refinement_startup!(semi, v_ode, u_ode)
    n_rounds = 0
    foreach_system(semi) do system
        refinement = get_refinement(system)
        isnothing(refinement) && return
        n_rounds = max(n_rounds, refinement.n_startup_rounds)
    end

    for _ in 1:n_rounds
        refinement!(semi, v_ode, u_ode)
    end

    return semi
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

# A particle that is activated in a fluid system, e.g. by an inflow, gets the target spacing
# that corresponds to its mass. Otherwise, it would keep the values of the buffer slot.
@inline initialize_refined_particle!(system, particle) = system

@inline function initialize_refined_particle!(system::AbstractFluidSystem, particle)
    return initialize_refined_particle!(system, get_refinement(system), particle)
end

@inline initialize_refined_particle!(system, ::Nothing, particle) = system

@inline function initialize_refined_particle!(system, refinement, particle)
    (; target_spacing, is_fixed) = system.cache
    (; reference_density) = refinement

    mass = hydrodynamic_mass(system, particle)
    target_spacing[particle] = (mass / reference_density)^(1 / ndims(system))
    is_fixed[particle] = false

    return system
end

include("refinement_criteria.jl")
include("spacing.jl")
include("split_pattern.jl")
include("splitting.jl")
include("merging.jl")
include("shifting.jl")
include("resize.jl")
