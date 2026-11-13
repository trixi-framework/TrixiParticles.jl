abstract type AbstractSplittingPattern end

"""
    HexagonalSplitting(; epsilon=0.4, alpha=0.9)

Splitting pattern that splits a particle into seven particles of equal mass:
the particle itself in the center and six new particles on a regular hexagon around it
at a distance of `epsilon * h`, where `h` is the smoothing length of the particle.
All seven particles get the smoothing length `alpha * h`.

The default values are the values of Vacondio et al. (2016) for the 6+1 pattern with equal
masses, as used in the code of Haftu et al. (2022).

!!! note
    This pattern is only implemented in 2D.
"""
struct HexagonalSplitting{ELTYPE} <: AbstractSplittingPattern
    epsilon :: ELTYPE
    alpha   :: ELTYPE

    function HexagonalSplitting(; epsilon=0.4, alpha=0.9)
        return new{typeof(epsilon)}(epsilon, alpha)
    end
end

# The directions of the new particles relative to the particle in the center
@inline function split_directions(pattern::HexagonalSplitting, ::Val{2})
    ELTYPE = typeof(pattern.epsilon)

    # Angles `j * π / 3` for `j = 0, ..., 5` as in the code of Haftu et al.
    return ntuple(j -> SVector(cospi(ELTYPE(j - 1) / 3), sinpi(ELTYPE(j - 1) / 3)), 6)
end

"""
    TriangularSplitting(; epsilon=0.4, alpha=0.92)

Splitting pattern that splits a particle into four particles of equal mass:
the particle itself in the center and three new particles at a distance of `epsilon * h`
at the angles ``\\pi/4``, ``3\\pi/4`` and ``5\\pi/4``, where `h` is the smoothing length
of the particle. All four particles get the smoothing length `alpha * h`.

This is the 3+1 pattern of the code of Haftu et al. (2022) (`copy_props2d_vacondio_4_beta_1`)
with its parameters.

!!! warning
    As in the code of Haftu et al., the new particles are not placed symmetrically around
    the particle in the center, so the center of mass moves by `epsilon * h / 4`.

!!! note
    This pattern is only implemented in 2D.
"""
struct TriangularSplitting{ELTYPE} <: AbstractSplittingPattern
    epsilon :: ELTYPE
    alpha   :: ELTYPE

    function TriangularSplitting(; epsilon=0.4, alpha=0.92)
        return new{typeof(epsilon)}(epsilon, alpha)
    end
end

@inline function split_directions(pattern::TriangularSplitting, ::Val{2})
    ELTYPE = typeof(pattern.epsilon)

    # Angles `π / 2 * (j + 1 / 2)` for `j = 0, 1, 2` as in the code of Haftu et al.
    return ntuple(j -> SVector(cospi(ELTYPE(2 * j - 1) / 4), sinpi(ELTYPE(2 * j - 1) / 4)),
                  3)
end

# TODO: Splitting patterns in 3D
function split_directions(pattern, ::Val{NDIMS}) where {NDIMS}
    throw(ArgumentError("`$(nameof(typeof(pattern)))` is not implemented in $(NDIMS)D"))
end

# The number of particles after splitting, including the particle in the center
@inline function n_split_particles(pattern, ndims_val)
    return length(split_directions(pattern, ndims_val)) + 1
end
