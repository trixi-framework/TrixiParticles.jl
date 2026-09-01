# Sorted in order of computational cost
@doc raw"""
    AkinciFreeSurfaceCorrection(rho0)

Free surface correction according to [Akinci et al. (2013)](@cite Akinci2013).
At a free surface, the mean density is typically lower than the reference density,
resulting in reduced surface tension and viscosity forces.
The free surface correction adjusts the viscosity, pressure, and surface tension forces
near free surfaces to counter this effect.
It's important to note that this correlation is unphysical and serves as an approximation.
The computation time added by this method is about 2--3%.

Mathematically the idea is quite simple. If we have an SPH particle in the middle of a volume
at rest, its density will be identical to the rest density ``\rho_0``. If we now consider an SPH
particle at a free surface at rest, it will have neighbors missing in the direction normal to
the surface, which will result in a lower density. If we calculate the correction factor
```math
k = \rho_0/\rho_\text{mean},
```
this value will be about ~1.5 for particles at the free surface and can then be used to increase
the pressure and viscosity accordingly.

# Arguments
- `rho0`: Rest density.
"""
struct AkinciFreeSurfaceCorrection{ELTYPE}
    rho0::ELTYPE

    function AkinciFreeSurfaceCorrection(rho0)
        ELTYPE = eltype(rho0)
        return new{ELTYPE}(rho0)
    end
end

# `rho_mean` is the mean density of the fluid, which is used to determine correction values near the free surface.
#  Return a tuple `(viscosity_correction, pressure_correction, surface_tension_correction)` representing the correction terms.
@inline function free_surface_correction(correction::AkinciFreeSurfaceCorrection,
                                         particle_system, rho_a, rho_b)
    # Equation 4 in ref
    rho_mean = (rho_a + rho_b) / 2
    k = correction.rho0 / rho_mean

    # Viscosity, pressure, surface_tension
    return k, 1, k
end

@inline function free_surface_correction(correction, particle_system, rho_a, rho_b)
    return 1, 1, 1
end

@doc raw"""
    ShepardKernelCorrection()

Kernel correction, as explained by [Bonet (1999)](@cite Bonet1999), uses Shepard interpolation
to obtain a zeroth-order consistent result (exact reproduction of constants), which was first
proposed by [Li et al. (1996)](@cite Li1996).

The kernel correction coefficient is determined by
```math
c(x) = \sum_{b=1} V_b W_b(x),
```
where ``V_b = m_b / \rho_b`` is the volume of particle ``b``.

This correction is applied with [`SummationDensity`](@ref) to correct the density and leads
to an improvement, especially at free surfaces. With summation density, the current one-pass
implementation uses the density available when each system is processed and therefore reduces
the free-surface error without guaranteeing convergence for multiple interacting systems.
[`DensityReinitializationCallback`](@ref) computes all simultaneously requested corrections
from the independently evolved continuity density before replacing any density.

!!! note
    - It is also referred to as "0th order correction".
    - In 2D, we can expect an increase of about 5--6% in computation time.
"""
struct ShepardKernelCorrection end

@doc raw"""
    KernelCorrection()

Kernel correction, as explained by [Bonet (1999)](@cite Bonet1999), uses Shepard interpolation
to obtain a zeroth-order consistent kernel gradient (an exact zero gradient for constants
when the correction coefficient is valid),
which was first proposed by Li et al.
This can be further extended to obtain a kernel corrected gradient as shown by [Basa et al. (2008)](@cite Basa2008).

The kernel correction coefficient is determined by
```math
c(x) = \sum_{b=1} V_b W_b(x)
```
The gradient of corrected kernel is determined by
```math
\nabla \tilde{W}_{b}(r) =\frac{\nabla W_{b}(r) - W_b(r) \gamma(r)}{\sum_{b=1} V_b W_b(r)} , \quad  \text{where} \quad
\gamma(r) = \frac{\sum_{b=1} V_b \nabla W_b(r)}{\sum_{b=1} V_b W_b(r)}.
```

This correction can be applied with [`SummationDensity`](@ref) and
[`ContinuityDensity`](@ref), which leads to an improvement, especially at free surfaces.

When the kernel correction coefficient is non-finite or not larger than
`sqrt(eps(T))` for the coefficient element type `T`, the correction is disabled
for that particle by setting the coefficient to one and the gradient offset
`γ` (`dw_gamma`) to zero. The corrected gradient then falls back to the
uncorrected kernel gradient and zeroth-order gradient consistency is not
retained for the degenerate particle.

!!! note
    - This only works when the boundary model uses [`SummationDensity`](@ref) (yet).
    - It is also referred to as "0th order correction".
    - In 2D, we can expect an increase of about 10--15% in computation time.
"""
struct KernelCorrection end

@doc raw"""
    MixedKernelGradientCorrection()

Combines [`GradientCorrection`](@ref) and [`KernelCorrection`](@ref),
which results in a first-order consistent kernel gradient reproducing both constant and affine
fields exactly (see [Bonet, 1999](@cite Bonet1999)).

# Notes:
- Stability issues, especially when particles separate into small clusters.
- Doubles the computational effort.
"""
struct MixedKernelGradientCorrection end

@doc raw"""
    CorrectionConfiguration(; density=nothing, gradient=nothing, force=nothing)

Configure density, gradient, and force corrections independently. `density` can be `nothing` or
[`ShepardKernelCorrection`](@ref). `gradient` can be `nothing`, [`KernelCorrection`](@ref),
[`GradientCorrection`](@ref), [`BlendedGradientCorrection`](@ref), or
[`MixedKernelGradientCorrection`](@ref). `force` can be `nothing` or
[`AkinciFreeSurfaceCorrection`](@ref).
"""
struct CorrectionConfiguration{D, G, F}
    density::D
    gradient::G
    force::F

    function CorrectionConfiguration(density::D, gradient::G, force::F) where {D, G, F}
        if !(density === nothing || density isa ShepardKernelCorrection)
            throw(ArgumentError("`density` must be `nothing` or `ShepardKernelCorrection()`"))
        end
        if !(gradient === nothing ||
             gradient isa Union{KernelCorrection, GradientCorrection,
                   BlendedGradientCorrection, MixedKernelGradientCorrection})
            throw(ArgumentError("unsupported gradient correction `$(typeof(gradient))`"))
        end
        return new{D, G, F}(density, gradient, force)
    end
end

function CorrectionConfiguration(; density=nothing, gradient=nothing, force=nothing)
    return CorrectionConfiguration(density, gradient, force)
end

correction_density(::Any) = nothing
correction_density(correction::ShepardKernelCorrection) = correction
correction_density(correction::CorrectionConfiguration) = correction.density

correction_gradient(::Nothing) = nothing
correction_gradient(::ShepardKernelCorrection) = nothing
correction_gradient(::AkinciFreeSurfaceCorrection) = nothing
correction_gradient(correction) = correction
correction_gradient(correction::CorrectionConfiguration) = correction.gradient

correction_force(correction) = correction
correction_force(correction::CorrectionConfiguration) = correction.force

function resolve_correction_configuration(density_correction, gradient_correction,
                                          force_correction)
    if density_correction === nothing && gradient_correction === nothing &&
       force_correction === nothing
        return nothing
    end

    return CorrectionConfiguration(; density=density_correction,
                                    gradient=gradient_correction,
                                    force=force_correction)
end

function kernel_correction_coefficient(system::AbstractFluidSystem, particle)
    return system.cache.kernel_correction_coefficient[particle]
end

function kernel_correction_coefficient(system::AbstractBoundarySystem, particle)
    return system.boundary_model.cache.kernel_correction_coefficient[particle]
end

function compute_correction_values!(system, correction, u, v_ode, u_ode, semi)
    return system
end

function compute_correction_values!(system, ::ShepardKernelCorrection, u, v_ode, u_ode,
                                    semi)
    return compute_shepard_coeff!(system, current_coordinates(u, system), v_ode, u_ode,
                                  semi,
                                  system.cache.kernel_correction_coefficient)
end

function compute_correction_values!(system::AbstractBoundarySystem,
                                    ::ShepardKernelCorrection, u,
                                    v_ode, u_ode, semi)
    return compute_shepard_coeff!(system, current_coordinates(u, system), v_ode, u_ode,
                                  semi,
                                  system.boundary_model.cache.kernel_correction_coefficient)
end

function compute_shepard_coeff!(system, system_coords, v_ode, u_ode, semi,
                                kernel_correction_coefficient)
    set_zero!(kernel_correction_coefficient)

    # Use enabled neighbor systems for the correction value.
    @trixi_timeit timer() "compute correction value" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            if !has_system_interaction(system, neighbor_system, semi)
                # No interaction between these systems.
                return
            end

            neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

            # Loop over all pairs of particles and neighbors within the kernel cutoff
            foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                                   semi) do particle, neighbor, pos_diff, distance
                rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
                m_b = hydrodynamic_mass(neighbor_system, neighbor)
                volume = m_b / rho_b

                kernel_correction_coefficient[particle] += volume *
                                                           kernel(hydrodynamic_smoothing_kernel(system),
                                                                  distance,
                                                                  hydrodynamic_smoothing_length(system,
                                                                                                particle))
            end
        end
    end

    sanitize_kernel_correction_coefficient!(kernel_correction_coefficient, system, semi)

    return kernel_correction_coefficient
end

function sanitize_kernel_correction_coefficient!(coefficient, system, semi)
    @threaded semi for particle in eachindex(coefficient)
        value = coefficient[particle]
        if !isfinite(value) || value <= zero(value)
            coefficient[particle] = one(value)
        end
    end

    return coefficient
end

function dw_gamma(system::AbstractFluidSystem, particle)
    return extract_svector(system.cache.dw_gamma, system, particle)
end

function dw_gamma(system::AbstractBoundarySystem, particle)
    return extract_svector(system.boundary_model.cache.dw_gamma, system, particle)
end

function compute_correction_values!(system::AbstractFluidSystem,
                                    correction::Union{KernelCorrection,
                                                      MixedKernelGradientCorrection}, u,
                                    v_ode, u_ode, semi)
    compute_correction_values!(system, correction, current_coordinates(u, system), v_ode,
                               u_ode, semi,
                               system.cache.kernel_correction_coefficient,
                               system.cache.dw_gamma)
end

function compute_correction_values!(system::AbstractBoundarySystem,
                                    correction::Union{KernelCorrection,
                                                      MixedKernelGradientCorrection}, u,
                                    v_ode, u_ode, semi)
    compute_correction_values!(system, correction, current_coordinates(u, system), v_ode,
                               u_ode, semi,
                               system.boundary_model.cache.kernel_correction_coefficient,
                               system.boundary_model.cache.dw_gamma)
end

function compute_correction_values!(system,
                                    ::Union{KernelCorrection,
                                            MixedKernelGradientCorrection}, system_coords,
                                    v_ode,
                                    u_ode, semi, kernel_correction_coefficient, dw_gamma)
    set_zero!(kernel_correction_coefficient)
    set_zero!(dw_gamma)

    # Use enabled neighbor systems for the correction value.
    @trixi_timeit timer() "compute correction value" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            if !has_system_interaction(system, neighbor_system, semi)
                # No interaction between these systems.
                return
            end

            neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

            # For `distance == 0`, the analytical gradient is zero, but the unsafe gradient
            # and the density diffusion divide by zero.
            # To account for rounding errors, we check if `distance` is almost zero.
            # Since the coordinates are in the order of the smoothing length `h`, `distance^2` is in
            # the order of `h^2`, so we need to check `distance < sqrt(eps(h^2))`.
            # Note that `sqrt(eps(h^2)) != eps(h)`.
            h = hydrodynamic_smoothing_length(system, nothing)
            almostzero = sqrt(eps(h^2))

            # Loop over all pairs of particles and neighbors within the kernel cutoff
            foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                                   semi) do particle, neighbor, pos_diff, distance
                rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
                m_b = hydrodynamic_mass(neighbor_system, neighbor)
                volume = m_b / rho_b

                # Use uncorrected kernel to compute correction coefficients
                smoothing_kernel = hydrodynamic_smoothing_kernel(system)
                smoothing_length_ = hydrodynamic_smoothing_length(system, particle)
                W = kernel(smoothing_kernel, distance, smoothing_length_)

                kernel_correction_coefficient[particle] += volume * W

                # Only consider particles with a distance > 0.
                if distance > almostzero
                    # Now that we know that `distance` is not zero, we can safely call the
                    # unsafe version of the kernel gradient to avoid redundant zero checks.
                    grad_W = kernel_grad_unsafe(smoothing_kernel, pos_diff, distance,
                                                smoothing_length_)
                    tmp = volume * grad_W
                    for i in axes(dw_gamma, 1)
                        dw_gamma[i, particle] += tmp[i]
                    end
                end
            end
        end
    end

    minimum_coefficient = sqrt(eps(eltype(kernel_correction_coefficient)))
    @threaded semi for particle in eachparticle(system)
        coefficient = kernel_correction_coefficient[particle]
        if !isfinite(coefficient) || coefficient <= minimum_coefficient
            kernel_correction_coefficient[particle] = one(coefficient)
            for i in axes(dw_gamma, 1)
                dw_gamma[i, particle] = zero(eltype(dw_gamma))
            end
        else
            for i in axes(dw_gamma, 1)
                dw_gamma[i, particle] /= coefficient
            end
        end
    end

    return kernel_correction_coefficient
end

@doc raw"""
    GradientCorrection()

Compute the corrected gradient of particle interactions based on their relative positions
(see [Bonet, 1999](@cite Bonet1999)).

# Mathematical Details

Given the standard SPH representation, the gradient of a field ``A`` at particle ``a`` is
given by

```math
\nabla A_a = \sum_b m_b \frac{A_b - A_a}{\rho_b} \nabla_{r_a} W(\Vert r_a - r_b \Vert, h),
```
where ``m_b`` is the mass of particle ``b`` and ``\rho_b`` is the density of particle ``b``.

The gradient correction, as commonly proposed, involves multiplying this gradient with a correction matrix $L$:

```math
\tilde{\nabla} A_a = \bm{L}_a \nabla A_a
```

The correction matrix  $\bm{L}_a$ is computed based on the provided particle configuration,
aiming to make the corrected gradient more accurate, especially near domain boundaries.
When its first-moment matrix is full rank and passes the singularity threshold, the
correction gives a first-order consistent gradient by differentiating every affine field
exactly. Rejected matrices fall back to the uncorrected gradient and do not retain this
property.
For smooth fields, the local truncation error is generally ``O(h)`` on asymmetric supports and
``O(h^2)`` on symmetric interior supports.

To satisfy
```math
\sum_b V_b r_{ba} \otimes \tilde{\nabla}W_b(r_a) = \left( \sum_b V_b r_{ba} \otimes \nabla W_b(r_a) \right) \bm{L}_a^T = \bm{I}
```
the correction matrix $\bm{L}_a$ is evaluated explicitly as
```math
\bm{L}_a = \left( \sum_b V_b \nabla W_b(r_{a}) \otimes r_{ba} \right)^{-1}.
```

!!! note
    - Stability issues arise, especially when particles separate into small clusters.
    - Doubles the computational effort.
    - Better stability with smoother smoothing Kernels with larger support, e.g. [`SchoenbergQuinticSplineKernel`](@ref) or [`WendlandC6Kernel`](@ref).
    - Set `dt_max =< 1e-3` for stability.
"""
struct GradientCorrection end

@doc raw"""
    BlendedGradientCorrection()

Calculate a blended gradient to reduce the stability issues of the [`GradientCorrection`](@ref)
as explained by [Bonet (1999)](@cite Bonet1999).

This calculates the following,
```math
\tilde\nabla A_i = (1-\lambda) \nabla A_i + \lambda L_i \nabla A_i
```
with ``0 \leq \lambda \leq 1`` being the blending factor.
For a fixed ``\lambda < 1``, the uncorrected first-moment error remains and no asymptotic order
improvement is guaranteed.

# Arguments
- `blending_factor`: Blending factor between corrected and regular SPH gradient.
"""
struct BlendedGradientCorrection{ELTYPE <: Real}
    blending_factor::ELTYPE

    function BlendedGradientCorrection(blending_factor)
        if !(zero(blending_factor) <= blending_factor <= one(blending_factor))
            throw(ArgumentError("`blending_factor` must be between 0 and 1"))
        end

        return new{eltype(blending_factor)}(blending_factor)
    end
end

# Called only by DensityDiffusion and TLSPH
function compute_gradient_correction_matrix!(corr_matrix, system, coordinates, density_fun,
                                             semi)
    (; mass) = system

    set_zero!(corr_matrix)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    foreach_point_neighbor(system, system, coordinates, coordinates,
                           semi) do particle, neighbor, pos_diff, distance
        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)
        iszero(grad_kernel) && return

        volume = @inbounds mass[neighbor] / density_fun(neighbor)

        # This is the same as using `transpose`, but it's faster due to
        # https://github.com/JuliaLang/LinearAlgebra.jl/issues/1102.
        result = volume * grad_kernel * permutedims(pos_diff)

        for j in 1:ndims(system), i in 1:ndims(system)
            @inbounds corr_matrix[i, j, particle] -= result[i, j]
        end
    end

    correction_matrix_inversion_step!(corr_matrix, system, semi)

    return corr_matrix
end

function compute_gradient_correction_matrix!(corr_matrix::AbstractArray, system,
                                             coordinates, v_ode, u_ode, semi,
                                             correction, smoothing_kernel)
    set_zero!(corr_matrix)

    # Loop over all pairs of particles and neighbors within the kernel cutoff
    @trixi_timeit timer() "compute correction matrix" begin
        foreach_system_wrapped(semi, v_ode,
                               u_ode) do neighbor_system, v_neighbor_system,
                                         u_neighbor_system
            if !has_system_interaction(system, neighbor_system, semi)
                # No interaction between these systems.
                return
            end

            neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)
            almostzero = sqrt(eps(compact_support(system, neighbor_system)^2))

            foreach_point_neighbor(system, neighbor_system, coordinates, neighbor_coords,
                                   semi) do particle, neighbor, pos_diff, distance
                # Skip neighbors with the same position if the kernel gradient is zero.
                # Note that `return` only exits the closure, i.e., skips the current neighbor.
                skip_zero_distance(correction) && distance < almostzero && return

                # Now that we know that `distance` is not zero, we can safely call the unsafe
                # version of the kernel gradient to avoid redundant zero checks.
                smoothing_length_ = hydrodynamic_smoothing_length(system, particle)
                grad_kernel = correction_matrix_kernel_grad_unsafe(correction,
                                                                   smoothing_kernel,
                                                                   pos_diff, distance,
                                                                   smoothing_length_,
                                                                   system,
                                                                   particle)

                volume = hydrodynamic_mass(neighbor_system, neighbor) /
                         current_density(v_neighbor_system, neighbor_system, neighbor)

                # This is the same as using `transpose`, but it's faster due to
                # https://github.com/JuliaLang/LinearAlgebra.jl/issues/1102.
                L = volume * grad_kernel * permutedims(pos_diff)

                # pos_diff is always x_a - x_b hence * -1 to switch the order to x_b - x_a
                @inbounds for j in 1:ndims(system), i in 1:ndims(system)
                    corr_matrix[i, j, particle] -= L[i, j]
                end
            end
        end
    end

    correction_matrix_inversion_step!(corr_matrix, system, semi)

    return corr_matrix
end

@inline function correction_matrix_kernel_grad_unsafe(correction, smoothing_kernel,
                                                      pos_diff,
                                                      distance, smoothing_length_, system,
                                                      particle)
    return kernel_grad_unsafe(smoothing_kernel, pos_diff, distance, smoothing_length_)
end

@inline function correction_matrix_kernel_grad_unsafe(::MixedKernelGradientCorrection,
                                                      smoothing_kernel, pos_diff, distance,
                                                      smoothing_length_, system, particle)
    return corrected_kernel_grad_unsafe(smoothing_kernel, pos_diff, distance,
                                        smoothing_length_,
                                        KernelCorrection(), system, particle)
end

function correction_matrix_inversion_step!(corr_matrix, system, semi)
    @threaded semi for particle in eachparticle(system)
        L = extract_smatrix(corr_matrix, system, particle)

        # The matrix `L` becomes singular when the particle and all neighbors are collinear
        # (in 2D) or lie all in the same plane (in 3D). Nearly singular matrices are also
        # rejected below to avoid amplifying particle disorder.
        # This happens only when two (in 2D) or three (in 3D) particles are isolated,
        # or in cases where there is only one layer of fluid particles on a wall.
        # In these edge cases, we just disable the correction and set the corrected
        # gradient to be the uncorrected one by setting `L` to the identity matrix.
        #
        # Proof: `L` is just a sum of tensor products of relative positions X_ab with
        # themselves. According to
        # https://en.wikipedia.org/wiki/Outer_product#Connection_with_the_matrix_product
        # the sum of tensor products can be rewritten as A A^T, where the columns of A
        # are the relative positions X_ab. The rank of A A^T is equal to the rank of A,
        # so `L` is singular if and only if the position vectors X_ab don't span the
        # full space, i.e., particle a and all neighbors lie on the same line (in 2D)
        # or plane (in 3D).
        minimum_relative_determinant = sqrt(eps(eltype(L)))
        entry_scale = maximum(abs, L)

        if isfinite(entry_scale) && !iszero(entry_scale)
            # Normalize by the Frobenius norm, which is invariant under rotations.
            # Scaling by the largest entry first keeps the norm calculation finite.
            L_entry_scaled = L / entry_scale
            frobenius_scale = norm(L_entry_scaled)
            L_scaled = L_entry_scaled / frobenius_scale
            relative_determinant = abs(det(L_scaled))

            if isfinite(frobenius_scale) && !iszero(frobenius_scale) &&
               isfinite(relative_determinant) &&
               relative_determinant >= minimum_relative_determinant
                # Avoid rescaling roundoff when the direct determinant is representable.
                raw_determinant = det(L)
                if isfinite(raw_determinant) && !iszero(raw_determinant)
                    candidate = inv(L)
                else
                    candidate = inv(L_scaled) / frobenius_scale / entry_scale
                end
                L_inv = all(isfinite, candidate) ? candidate : one(L)
            else
                L_inv = one(L)
            end
        else
            L_inv = one(L)
        end

        # Write inverse back to `corr_matrix`
        for j in 1:ndims(system), i in 1:ndims(system)
            @inbounds corr_matrix[i, j, particle] = L_inv[i, j]
        end
    end

    return corr_matrix
end

create_cache_correction(correction, density, NDIMS, nparticles) = (;)

function create_cache_correction(correction::CorrectionConfiguration, density, NDIMS,
                                 n_particles)
    density_cache = create_cache_correction(correction.density, density, NDIMS, n_particles)
    gradient_cache = create_cache_correction(correction.gradient, density, NDIMS,
                                             n_particles)
    return merge(density_cache, gradient_cache)
end

function create_cache_correction(::ShepardKernelCorrection, density, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density))
end

function create_cache_correction(::KernelCorrection, density, NDIMS, n_particles)
    dw_gamma = Array{eltype(density)}(undef, NDIMS, n_particles)
    return (; kernel_correction_coefficient=similar(density), dw_gamma)
end

function create_cache_correction(::Union{GradientCorrection, BlendedGradientCorrection},
                                 density,
                                 NDIMS, n_particles)
    correction_matrix = Array{eltype(density), 3}(undef, NDIMS, NDIMS, n_particles)
    return (; correction_matrix)
end

function create_cache_correction(::MixedKernelGradientCorrection, density, NDIMS,
                                 n_particles)
    kernel_cache = create_cache_correction(KernelCorrection(), density, NDIMS, n_particles)
    gradient_cache = create_cache_correction(GradientCorrection(), density, NDIMS,
                                             n_particles)

    return (; kernel_cache..., gradient_cache...)
end
