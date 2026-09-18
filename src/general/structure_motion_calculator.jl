@doc raw"""
    StructureMotionCalculator(system::TotalLagrangianSPHSystem, semi, position;
                              quantity=identity)

Functor that reconstructs the motion of a [`TotalLagrangianSPHSystem`](@ref) around a point
`position` in the initial (reference) configuration from the surrounding particles.
It can be passed as a custom quantity to [`SolutionSavingCallback`](@ref) and
[`PostprocessCallback`](@ref).

In contrast to tracking a single particle, the motion is obtained by an SPH interpolation
over all particles in the kernel support of `position`, which makes it less sensitive to
the discretization and allows tracking points that are not particle positions.
Since a Total Lagrangian formulation is used, the reference configuration is fixed, and
the interpolation weights
```math
w_a = \frac{V_a W(\Vert \bm{X}_a - \bm{X} \Vert, h)}
           {\sum_b V_b W(\Vert \bm{X}_b - \bm{X} \Vert, h)}
```
are computed once when the calculator is constructed.
Here, ``\bm{X}`` is `position`, ``\bm{X}_a`` is the initial position of particle ``a``,
and ``V_a = m_a / \rho_a`` is its volume in the reference configuration.
The sum in the denominator is a Shepard correction, which is required because the kernel
support is usually truncated when tracking a point close to the surface of the structure.

Calling the functor returns `nothing` for all systems but the one it was constructed with.
For this system, it returns `quantity(motion)`, where `motion` is a named tuple with
the following fields.
- `displacement`: Interpolated displacement ``\sum_a w_a (\bm{x}_a - \bm{X}_a)``.
- `deformation_gradient`: Interpolated deformation gradient ``\bm{F} = \sum_a w_a \bm{F}_a``.
- `rotation`: In 2D, the rotation angle (in radians) of the proper orthogonal factor
              ``\bm{R}`` of the polar decomposition ``\bm{F} = \bm{R} \bm{U}``.
              `NaN` in 3D, where no single rotation angle is defined.
              Use `deformation_gradient` to compute the rotation in 3D.

Note that both [`SolutionSavingCallback`](@ref) and [`PostprocessCallback`](@ref) expect
a scalar value per system. Use the keyword argument `quantity` to select a single
component of the motion (see the examples below).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Arguments
- `system`: The [`TotalLagrangianSPHSystem`](@ref) whose motion should be tracked.
- `semi`: The [`Semidiscretization`](@ref) that contains `system`.
- `position`: Point in the initial configuration around which the motion is reconstructed.
              Must be inside the kernel support of at least one particle of `system`.

# Keywords
- `quantity=identity`: Function that is applied to the named tuple described above to
                       extract the value that is returned by the functor.

# Examples
```jldoctest; output = false
system = TotalLagrangianSPHSystem(RectangularShape(0.1, (10, 3), (0.0, 0.0), density=1000.0,
                                                   place_on_shell=true);
                                  smoothing_kernel=WendlandC2Kernel{2}(),
                                  smoothing_length=0.2, young_modulus=1.0e6,
                                  poisson_ratio=0.4, clamped_particles=1:3)
semi = Semidiscretization(system)

# Note that `Semidiscretization` creates a deep copy of the system,
# which means we have to extract the new system from `semi`.
system_new = semi.systems[1]

# Track the motion of the center of the right end of the structure
tip_position = (0.9, 0.1)

# Custom quantities returning the displacement in x- and y-direction
deflection_x = StructureMotionCalculator(system_new, semi, tip_position,
                                         quantity=motion -> motion.displacement[1])
deflection_y = StructureMotionCalculator(system_new, semi, tip_position,
                                         quantity=motion -> motion.displacement[2])

saving_callback = SolutionSavingCallback(dt=0.1; deflection_x, deflection_y)

# Now pass the callback to the `solve` function:
# sol = solve(ode, ..., callback=saving_callback)
nothing

# output
[ Info: To create the self-interaction neighborhood search of a `TotalLagrangianSPHSystem`, a deep copy of the system is created inside the `Semidiscretization`. Use `system = semi.systems[i]` to access simulation data.
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
    position_ = SVector{NDIMS, ELTYPE}(position)

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
        distance = norm(pos_diff)
        distance > search_radius && continue

        volume = mass[particle] / material_density[particle]
        weight = volume * kernel(system.smoothing_kernel, distance,
                                 system.smoothing_length)

        # Skip particles exactly on the boundary of the compact support
        iszero(weight) && continue

        push!(particles, particle)
        push!(weights, weight)
    end

    sum_weights = sum(weights, init=zero(ELTYPE))
    if sum_weights <= eps(ELTYPE)
        throw(ArgumentError("`position` is not inside the kernel support of any particle " *
                            "of the system in the initial configuration"))
    end

    # Apply the Shepard correction once, so that the weights sum up to one
    weights ./= sum_weights

    return StructureMotionCalculator(system_index, position_, particles, weights, quantity)
end

function (calculator::StructureMotionCalculator)(system, dv_ode, du_ode, v_ode, u_ode,
                                                 semi, t)
    if system_indices(system, semi) != calculator.system_index
        return nothing
    end

    return calculator.quantity(structure_motion(calculator, system))
end

# Interpolate displacement and deformation gradient at the tracked point.
# Note that both callbacks update all systems before calling custom quantities, so the
# current coordinates and deformation gradients stored in `system` are up to date.
function structure_motion(calculator::StructureMotionCalculator, system)
    (; particles, weights) = calculator

    displacement = zero(SVector{ndims(system), eltype(system)})
    deformation_grad = zero(SMatrix{ndims(system), ndims(system), eltype(system)})

    for i in eachindex(particles)
        particle = particles[i]
        weight = weights[i]

        displacement += weight * (current_coords(system, particle) -
                         initial_coords(system, particle))
        deformation_grad += weight * deformation_gradient(system, particle)
    end

    return (; displacement, deformation_gradient=deformation_grad,
            rotation=polar_decomposition_angle(deformation_grad))
end

# Rotation angle of the proper orthogonal factor `R` of the polar decomposition `F = R U`.
# This follows from `R = F (F^T F)^(-1/2)` being the closest rotation to `F`, which in 2D
# maximizes `tr(R^T F) = cos(θ) (F11 + F22) + sin(θ) (F21 - F12)`.
@inline function polar_decomposition_angle(F::SMatrix{2, 2})
    return atan(F[2, 1] - F[1, 2], F[1, 1] + F[2, 2])
end

# There is no single rotation angle in 3D. Users can compute the rotation from the
# deformation gradient, which is also returned by the calculator.
@inline function polar_decomposition_angle(F::SMatrix{NDIMS, NDIMS, ELTYPE}) where {NDIMS,
                                                                                    ELTYPE}
    return ELTYPE(NaN)
end
