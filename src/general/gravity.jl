@doc raw"""
    NewtonianGravity(; gravitational_constant, softening_length=0, cutoff_radius=Inf)

Model for Newtonian pairwise self-gravity.

For the displacement ``\Delta \boldsymbol{x}_{ab} = \boldsymbol{x}_a -
\boldsymbol{x}_b`` and distance ``r = \lVert \Delta \boldsymbol{x}_{ab} \rVert``, the
acceleration of particle ``a`` due to a neighbor ``b`` of mass ``m_b`` is

```math
\boldsymbol{a}_{ab} = -G m_b
    \frac{\Delta \boldsymbol{x}_{ab}}{(r^2 + \epsilon^2)^{3/2}}.
```

This is Plummer softening with softening length ``\epsilon``. The acceleration is applied
for ``r \leq r_c`` and is zero for ``r > r_c``, where ``r_c`` is the cutoff radius. At
zero distance, the softened model (``\epsilon > 0``) has zero acceleration. The unsoftened
model is singular for distinct coincident particles and raises a `DomainError`.

With a finite cutoff, a continuous potential corresponding to this acceleration is

```math
U_{ab}(r) = -G m_a m_b \left(
    \frac{1}{\sqrt{r^2 + \epsilon^2}} -
    \frac{1}{\sqrt{r_c^2 + \epsilon^2}}
\right)
```

for ``r \leq r_c``, and zero otherwise. For ``r_c = \infty``, the second term vanishes.

# Keywords
- `gravitational_constant`: Non-negative, finite gravitational constant ``G``.
- `softening_length=0`: Non-negative, finite Plummer softening length ``\epsilon``.
- `cutoff_radius=Inf`: Positive cutoff radius ``r_c`` or `Inf`.

All parameters, coordinates, masses, and times must use one consistent unit system. In
particular, ``G`` has dimensions ``L^3 M^{-1} T^{-2}``, while `softening_length` and
`cutoff_radius` have the same length unit as the particle coordinates.
"""
# Encode softening and cutoff as type parameters so the common Newtonian path has no
# runtime branches.
struct NewtonianGravity{ELTYPE <: Real, SOFTENED, CUTOFF}
    gravitational_constant :: ELTYPE
    softening_length       :: ELTYPE
    cutoff_radius          :: ELTYPE

    function NewtonianGravity(; gravitational_constant::Real,
                              softening_length::Real=zero(gravitational_constant),
                              cutoff_radius::Real=oftype(float(gravitational_constant),
                                                         Inf))
        gravitational_constant_, softening_length_,
        cutoff_radius_ = promote(gravitational_constant,
                                 softening_length,
                                 cutoff_radius)

        if !isfinite(gravitational_constant_) ||
           gravitational_constant_ < zero(gravitational_constant_)
            throw(ArgumentError("`gravitational_constant` must be non-negative and finite"))
        end

        if !isfinite(softening_length_) ||
           softening_length_ < zero(softening_length_)
            throw(ArgumentError("`softening_length` must be non-negative and finite"))
        end

        if isnan(cutoff_radius_) || cutoff_radius_ <= zero(cutoff_radius_)
            throw(ArgumentError("`cutoff_radius` must be positive or `Inf`"))
        end

        return new{typeof(gravitational_constant_),
                   !iszero(softening_length_),
                   !isinf(cutoff_radius_)}(gravitational_constant_,
                                           softening_length_,
                                           cutoff_radius_)
    end
end

@inline function gravity_acceleration(gravity::NewtonianGravity, pos_diff, distance,
                                      neighbor_mass)
    if iszero(distance)
        if gravity isa NewtonianGravity{<:Real, false}
            throw(DomainError(distance,
                              "unsoftened Newtonian gravity is singular at zero distance"))
        end

        return zero(pos_diff)
    end

    return gravity_acceleration_factor(gravity, distance, neighbor_mass) * pos_diff
end

@inline function gravity_acceleration_factor(gravity::NewtonianGravity{<:Real, false,
                                                                       false},
                                             distance, neighbor_mass)
    (; gravitational_constant) = gravity

    return -gravitational_constant * neighbor_mass * (1 / distance^3)
end

@inline function gravity_acceleration_factor(gravity::NewtonianGravity{<:Real, true,
                                                                       false},
                                             distance, neighbor_mass)
    (; gravitational_constant, softening_length) = gravity

    distance_square = distance^2 + softening_length^2
    inverse_distance_cube = inv(distance_square * sqrt(distance_square))

    return -gravitational_constant * neighbor_mass * inverse_distance_cube
end

@inline function gravity_acceleration_factor(gravity::NewtonianGravity{<:Real, false,
                                                                       true},
                                             distance, neighbor_mass)
    (; gravitational_constant, cutoff_radius) = gravity

    distance > cutoff_radius && return zero(distance)

    return -gravitational_constant * neighbor_mass * (1 / distance^3)
end

@inline function gravity_acceleration_factor(gravity::NewtonianGravity{<:Real, true,
                                                                       true},
                                             distance, neighbor_mass)
    (; gravitational_constant, softening_length, cutoff_radius) = gravity

    distance > cutoff_radius && return zero(distance)

    distance_square = distance^2 + softening_length^2
    inverse_distance_cube = inv(distance_square * sqrt(distance_square))

    return -gravitational_constant * neighbor_mass * inverse_distance_cube
end
