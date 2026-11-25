@doc raw"""
    StructureMotionCalculator(system::TotalLagrangianSPHSystem, semi, position;
                              quantity=identity)

Functor that reconstructs the motion of a [`TotalLagrangianSPHSystem`](@ref) at `position`
in the initial (reference) configuration by interpolating data from neighboring particles.
It can be passed as a custom quantity to [`SolutionSavingCallback`](@ref) and
[`PostprocessCallback`](@ref).

The calculator uses SPH interpolation over all particles within the kernel support of
`position`. This allows tracking points that do not coincide with particle positions and
can reduce sensitivity to individual particle motion compared to tracking a single particle.
Near the surface of the structure, the kernel support is generally truncated. We therefore
use the corrected kernel of [Bonet and Lok (1999)](@cite Bonet1999), which reproduces linear
fields exactly even when the kernel support is truncated.

A field ``A`` is interpolated at the reference point ``\bm{X}`` (`position`) as
```math
A(\bm{X}) = \sum_b V_b A_b \tilde{W}_b(\bm{X}),
```
where ``V_b = m_b / \rho_{0b}`` is the volume of particle ``b`` in the reference configuration
and ``\tilde{W}_b`` is the corrected kernel
```math
\tilde{W}_b(\bm{X}) = W_b(\bm{X}) \alpha(\bm{X}) \left( 1 + \bm{\beta}(\bm{X}) \cdot (\bm{X} - \bm{X}_b) \right),
```
with ``W_b(\bm{X}) = W(\Vert \bm{X} - \bm{X}_b \Vert, h)`` and ``\bm{X}_b`` the initial
position of particle ``b``.
The correction parameters ``\alpha`` and ``\bm{\beta}`` are given by
```math
\bm{\beta}(\bm{X}) = -\bm{L}^{-1} \sum_b V_b (\bm{X} - \bm{X}_b) W_b(\bm{X}), \quad
\bm{L} = \sum_b V_b (\bm{X} - \bm{X}_b) \otimes (\bm{X} - \bm{X}_b) W_b(\bm{X}),
```
and
```math
\alpha(\bm{X}) = \left( \sum_b V_b \left( 1 + \bm{\beta}(\bm{X}) \cdot (\bm{X} - \bm{X}_b) \right) W_b(\bm{X}) \right)^{-1}.
```
This interpolation is exact for affine motions (rigid body motions and homogeneous
deformations). In contrast, a Shepard interpolation (``\bm{\beta} = \bm{0}``)
only reproduces constant fields and is inaccurate close to the surface of the structure.

Calling the functor returns `nothing` for all systems but the one it was constructed with.
For this system, it returns `quantity(motion)`, where `motion` is a named tuple with
the following fields.
- `displacement`: Interpolated displacement.
- `deformation_gradient`: Interpolated deformation gradient.
- `rotation`: In 2D, the rotation angle (in radians) of the proper orthogonal factor
              ``\bm{R}`` of the polar decomposition ``\bm{F} = \bm{R} \bm{U}``.
              `NaN` in 3D, where no single rotation angle is defined.
              Use `deformation_gradient` to compute the rotation in 3D.

Use the keyword argument `quantity` to select a single component of the motion
(see the examples below).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `system`: The [`TotalLagrangianSPHSystem`](@ref) whose motion should be tracked.
- `semi`: The [`Semidiscretization`](@ref) that contains `system`.
- `position`: Point in the initial configuration around which the motion is reconstructed.
              The initial positions of the particles of `system` in the kernel support
              of `position` must not all lie on a line (in 2D) or plane (in 3D).

# Keywords
- `quantity=identity`: Function that is applied to the named tuple described above to
                       extract the value that is returned by the functor.

# Examples
```jldoctest; output = false, setup = :(system = TotalLagrangianSPHSystem(RectangularShape(0.1, (3, 4), (0.0, 0.0), density=1.0); smoothing_kernel=WendlandC2Kernel{2}(), smoothing_length=1.0, young_modulus=1.0, poisson_ratio=1.0); semi = (; systems=(system,), parallelization_backend=SerialBackend())), filter = [r"output directory:.*", r"custom quantities:.*", r"\s+│"]
semi = Semidiscretization(system)

# Note that `Semidiscretization` creates a deep copy of the system,
# which means we have to extract the new system from `semi`.
system_new = semi.systems[1]

# Track the motion of the point of the structure that initially lies at the origin.
tip_position = (0.0, 0.0)

# Custom quantities returning the displacement in x- and y-direction.
deflection_x = StructureMotionCalculator(system_new, semi, tip_position,
                                         quantity=motion -> motion.displacement[1])
deflection_y = StructureMotionCalculator(system_new, semi, tip_position,
                                         quantity=motion -> motion.displacement[2])

saving_callback = SolutionSavingCallback(dt=0.1; deflection_x, deflection_y)

# Now pass the callback to the `solve` function:
# sol = solve(ode, ..., callback=saving_callback)

# output
[ Info: To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`, a deep copy of the system is created inside the `Semidiscretization`. Use `system = semi.systems[i]` to access simulation data.
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ SolutionSavingCallback                                                                           │
│ ══════════════════════                                                                           │
│ dt: ……………………………………………………………………… 0.1                                                              │
│ custom quantities: ……………………………… *ignored with filter regex above*                               │
│ save initial solution: …………………… yes                                                              │
│ save final solution: ………………………… yes                                                              │
│ overwrite solution: …………………………… no                                                               │
│ output directory: ………………………………… *path ignored with filter regex above*                           │
│ prefix: ……………………………………………………………                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```
"""
struct StructureMotionCalculator{NDIMS, ELTYPE, Q}
    system_index :: Int
    position     :: SVector{NDIMS, ELTYPE}
    particles    :: Vector{Int}
    weights      :: Vector{ELTYPE}
    quantity     :: Q
end

function StructureMotionCalculator(system::TotalLagrangianSPHSystem, semi, position;
                                   quantity=identity)
    NDIMS = ndims(system)
    ELTYPE = eltype(system)

    system_index = system_indices(system, semi)

    # Check vector length before converting to `SVector` to avoid extremely long
    # compile times when accidentally passing large vectors.
    if length(position) != ndims(system)
        throw(ArgumentError("length of `position` must match the number of dimensions"))
    end
    position_ = SVector{NDIMS, eltype(position)}(Tuple(position))

    # Transfer to CPU if data is on the GPU. Do nothing if already on CPU.
    initial_coordinates = transfer2cpu(system.initial_coordinates)
    mass = transfer2cpu(system.mass)
    material_density = transfer2cpu(system.material_density)

    # The reference configuration is fixed, so the neighboring particles and the
    # interpolation weights only have to be determined once.
    search_radius = compact_support(system, system)
    particles = Int[]
    weights = ELTYPE[]

    for particle in eachparticle(system)
        pos_diff = extract_svector(initial_coordinates, Val(NDIMS), particle) - position_
        distance = norm(convert.(eltype(system), pos_diff))
        distance >= search_radius && continue

        volume = mass[particle] / material_density[particle]
        weight = volume * kernel(system.smoothing_kernel, distance,
                                 system.smoothing_length)

        push!(particles, particle)
        push!(weights, weight)
    end

    sum_weights = sum(weights, init=zero(ELTYPE))
    if sum_weights <= eps(ELTYPE)
        throw(ArgumentError("`position` is not inside the kernel support of any particle " *
                            "of the system in the initial configuration"))
    end

    # Apply the kernel correction by Bonet and Lok (1999), so that the weights reproduce
    # constant and linear fields.
    pos_diffs = [position_ - extract_svector(initial_coordinates, Val(NDIMS), particle)
                 for particle in particles]
    L = sum(weights[i] * pos_diffs[i] * pos_diffs[i]' for i in eachindex(particles))
    error_degenerate = ArgumentError("the particles in the kernel support of `position` " *
                                     "must not all lie on a line (in 2D) or plane (in 3D)")
    cond(L) > 1 / sqrt(eps(ELTYPE)) && throw(error_degenerate)

    beta = -(L \ sum(weights .* pos_diffs))
    for i in eachindex(weights)
        weights[i] *= 1 + dot(beta, pos_diffs[i])
    end

    inv_alpha = sum(weights)
    inv_alpha <= sqrt(eps(ELTYPE)) * sum_weights && throw(error_degenerate)
    weights ./= inv_alpha

    return StructureMotionCalculator(system_index, position_, particles, weights, quantity)
end

function (calculator::StructureMotionCalculator)(system, dv_ode, du_ode, v_ode, u_ode,
                                                 semi, t)
    if system_indices(system, semi) != calculator.system_index
        return nothing
    end

    return calculator.quantity(structure_motion(calculator, system))
end

# Interpolate the displacement and deformation gradient at the tracked point.
# Note that both callbacks update all systems before evaluating custom quantities, so the
# current coordinates and deformation gradients stored in `system` are up to date.
function structure_motion(calculator::StructureMotionCalculator, system)
    (; particles, weights) = calculator

    displacement = zero(SVector{ndims(system), eltype(system)})
    deformation_grad = zero(SMatrix{ndims(system), ndims(system), eltype(system)})

    for i in eachindex(particles)
        particle = particles[i]
        weight = weights[i]

        pos_diff = current_coords(system, particle) - initial_coords(system, particle)
        displacement += weight * convert.(eltype(system), pos_diff)
        deformation_grad += weight * deformation_gradient(system, particle)
    end

    return (; displacement, deformation_gradient=deformation_grad,
            rotation=polar_decomposition_angle(deformation_grad))
end

# Rotation angle `θ` of the polar decomposition `F = R U`, which splits `F` into
# a rotation `R` around `θ`, and a pure stretch `U`, which is symmetric.
# Standard formula for 2D: `θ = atan2(F[2, 1] - F[1, 2], F[1, 1] + F[2, 2])`.
@inline function polar_decomposition_angle(F::SMatrix{2, 2})
    return atan(F[2, 1] - F[1, 2], F[1, 1] + F[2, 2])
end

# There is no single rotation angle in 3D. Users can compute the rotation from the
# deformation gradient, which is also returned by the calculator.
@inline function polar_decomposition_angle(F::SMatrix{NDIMS, NDIMS, ELTYPE}) where {NDIMS,
                                                                                    ELTYPE}
    return ELTYPE(NaN)
end
