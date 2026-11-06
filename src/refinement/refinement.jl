struct ParticleRefinement{ELTYPE}
    spacing_ratio           :: ELTYPE # Ratio between spacings of different refinement bands
    min_spacing             :: ELTYPE # The smallest particle spacing in the simulation
    max_spacing             :: ELTYPE # The largest particle spacing in the simulation
    smoothing_length_factor :: ELTYPE # Ratio between smoothing length and particle spacing
    reference_density       :: ELTYPE # Used to compute the particle spacing from the mass
end

function ParticleRefinement(; spacing_ratio, min_spacing, max_spacing, reference_density,
                            smoothing_length_factor=1.2)
    return ParticleRefinement(spacing_ratio, min_spacing, max_spacing,
                              smoothing_length_factor, reference_density)
end
