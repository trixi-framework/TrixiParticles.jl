@doc raw"""
    SourceTermDamping(; damping_coefficient)

A source term to be used when a damping step is required before running a full simulation.
The term ``-c \cdot v_a`` is added to the acceleration ``\frac{\mathrm{d}v_a}{\mathrm{d}t}``
of particle ``a``, where ``c`` is the damping coefficient and ``v_a`` is the velocity of
particle ``a``.

# Keywords
- `damping_coefficient`:    The coefficient ``c`` above. A higher coefficient means more
                            damping. A coefficient of `1e-4` is a good starting point for
                            damping a fluid at rest.

# Examples
```jldoctest; output = false
source_terms = SourceTermDamping(; damping_coefficient=1e-4)

# output
SourceTermDamping{Float64}(0.0001)
```
"""
struct SourceTermDamping{ELTYPE}
    damping_coefficient::ELTYPE

    function SourceTermDamping(; damping_coefficient)
        return new{typeof(damping_coefficient)}(damping_coefficient)
    end
end

@inline function (source_term::SourceTermDamping)(coords, velocity, density, pressure, t)
    (; damping_coefficient) = source_term

    return -damping_coefficient * velocity
end

@doc raw"""
    SpongeLayer(; face_origin, face_normal, length, sound_speed, reference_velocity,
                strength=1)
    SpongeLayer(boundary_zone::BoundaryZone; length, sound_speed,
                reference_velocity=nothing, strength=1)

A source term that damps acoustic waves in a layer in front of an outflow boundary.
Pass it as `source_terms` to the fluid system.

In weakly compressible SPH, a prescribed pressure at an outflow boundary reflects acoustic
waves.
Together with a prescribed inflow velocity, the channel then acts as a resonator whose
acoustic modes are only damped by the numerical dissipation.
These modes can grow until the simulation becomes unstable.
This source term relaxes the velocity towards a reference velocity inside a layer of
thickness `length` in front of the boundary face:
```math
\frac{\mathrm{d}v_a}{\mathrm{d}t} = \dots - \sigma(d_a) \left( v_a - v_{\text{ref}}(x_a, t) \right),
\qquad
\sigma(d) = \sigma_{\max} \left( 1 - \frac{d}{L} \right)^2 \quad \text{for } 0 \leq d \leq L,
```
where ``d_a`` is the distance of particle ``a`` from the boundary face and ``L`` is the
thickness of the layer. Outside of the layer, the source term vanishes.
For acoustic waves much longer than the layer, the layer acts like an outflow impedance
``\rho_0 \int_0^L \sigma(d) \, \mathrm{d}d``,
which is non-reflecting when it equals the characteristic impedance ``\rho_0 c``.
Therefore, we choose
```math
\sigma_{\max} = 3 \, s \, \frac{c}{L},
```
where ``s`` is the `strength`. A `strength` of one is the non-reflecting value
for long waves. With a different strength ``s``, the fraction ``|1 - s| / (1 + s)``
of the amplitude of a long wave is reflected.

Inside the layer, the flow is relaxed towards the reference velocity,
so the layer should be placed downstream of the region of interest.

# Constructors
The second constructor takes the geometry from the [`BoundaryZone`](@ref) of the outflow,
so that the layer is placed in front of its boundary face.
If `reference_velocity` is not passed, the prescribed velocity of the boundary zone is used.

# Keywords
- `face_origin`:        A point on the boundary face.
- `face_normal`:        Normal vector of the boundary face pointing into the fluid domain.
- `length`:             Thickness ``L`` of the layer in front of the boundary face.
- `sound_speed`:        Speed of sound ``c`` of the fluid.
- `reference_velocity`: Velocity towards which the velocity is relaxed. Either a vector
                        or a function mapping the coordinates and time of a particle to
                        its reference velocity, like `reference_velocity` of a
                        [`BoundaryZone`](@ref). This should be the expected mean flow
                        (e.g., the inflow velocity profile) to avoid slowing down the flow.
- `strength=1`:         Strength ``s`` of the damping relative to the non-reflecting value.

# Examples
```jldoctest; output = false
sponge = SpongeLayer(; face_origin=(1.0, 0.0), face_normal=(-1.0, 0.0), length=0.2,
                     sound_speed=20.0, reference_velocity=(1.0, 0.0))

# output
SpongeLayer{2, Float64}(length=0.2, max_relaxation_rate=300.0)
```
"""
struct SpongeLayer{NDIMS, ELTYPE, V}
    face_origin         :: SVector{NDIMS, ELTYPE}
    face_normal         :: SVector{NDIMS, ELTYPE}
    length              :: ELTYPE
    max_relaxation_rate :: ELTYPE
    reference_velocity  :: V
end

function SpongeLayer(; face_origin, face_normal, length, sound_speed, reference_velocity,
                     strength=1)
    NDIMS = Base.length(face_origin)
    ELTYPE = eltype(float(length))

    if Base.length(face_normal) != NDIMS
        throw(ArgumentError("`face_normal` must have the same length as `face_origin`"))
    end

    if length <= 0
        throw(ArgumentError("`length` must be positive"))
    end

    face_origin_ = SVector{NDIMS, ELTYPE}(face_origin)
    face_normal_ = normalize(SVector{NDIMS, ELTYPE}(face_normal))

    # `integral(sigma_max * (1 - d / L)^2, d = 0..L) = sigma_max * L / 3`
    # must be `strength * sound_speed` (see docstring).
    max_relaxation_rate = convert(ELTYPE, 3 * strength * sound_speed / length)

    if reference_velocity isa Function
        reference_velocity_ = reference_velocity
    else
        if Base.length(reference_velocity) != NDIMS
            throw(ArgumentError("`reference_velocity` must be a function or a vector " *
                                "of length $NDIMS"))
        end
        reference_velocity_ = SVector{NDIMS, ELTYPE}(reference_velocity)
    end

    return SpongeLayer(face_origin_, face_normal_, convert(ELTYPE, length),
                       max_relaxation_rate, reference_velocity_)
end

function Base.show(io::IO, sponge::SpongeLayer{NDIMS, ELTYPE}) where {NDIMS, ELTYPE}
    @nospecialize sponge # reduce precompilation time

    print(io, "SpongeLayer{", NDIMS, ", ", ELTYPE, "}(length=", sponge.length,
          ", max_relaxation_rate=", sponge.max_relaxation_rate, ")")
end

@inline function (sponge::SpongeLayer)(coords, velocity, density, pressure, t)
    (; face_origin, face_normal, length, max_relaxation_rate) = sponge

    # Distance from the boundary face into the fluid domain
    distance = dot(coords - face_origin, face_normal)

    # Outside of the layer
    if distance < 0 || distance > length
        return zero(velocity)
    end

    relaxation_rate = max_relaxation_rate * (1 - distance / length)^2
    v_ref = sponge_reference_velocity(sponge.reference_velocity, coords, t)

    return -relaxation_rate * (velocity - v_ref)
end

@inline sponge_reference_velocity(v_ref::SVector, coords, t) = v_ref
@inline sponge_reference_velocity(v_ref, coords, t) = v_ref(coords, t)
