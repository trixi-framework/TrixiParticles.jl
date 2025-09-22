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
                                         particle_system, rho_mean)
    # Equation 4 in ref
    k = correction.rho0 / rho_mean

    # Viscosity, pressure, surface_tension
    return k, 1, k
end

@inline function free_surface_correction(correction, particle_system, rho_mean)
    return 1, 1, 1
end

@doc raw"""
    ShepardKernelCorrection()

Kernel correction, as explained by [Bonet (1999)](@cite Bonet1999), uses Shepard interpolation
to obtain a 0-th order accurate result, which was first proposed by [Li et al. (1996)](@cite Li1996).

The kernel correction coefficient is determined by
```math
c(x) = \sum_{b=1} V_b W_b(x),
```
where ``V_b = m_b / \rho_b`` is the volume of particle ``b``.

This correction is applied with [`SummationDensity`](@ref) to correct the density and leads
to an improvement, especially at free surfaces.

!!! note
    - It is also referred to as "0th order correction".
    - In 2D, we can expect an increase of about 5--6% in computation time.
"""
struct ShepardKernelCorrection end

@doc raw"""
    KernelCorrection()

Kernel correction, as explained by [Bonet (1999)](@cite Bonet1999), uses Shepard interpolation
to obtain a 0-th order accurate result, which was first proposed by Li et al.
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

!!! note
    - This only works when the boundary model uses [`SummationDensity`](@ref) (yet).
    - It is also referred to as "0th order correction".
    - In 2D, we can expect an increase of about 10--15% in computation time.
"""
struct KernelCorrection end

@doc raw"""
    MixedKernelGradientCorrection()

Combines [`GradientCorrection`](@ref) and [`KernelCorrection`](@ref),
which results in a 1st-order-accurate SPH method (see [Bonet, 1999](@cite Bonet1999)).

# Notes:
- Stability issues, especially when particles separate into small clusters.
- Doubles the computational effort.
"""
struct MixedKernelGradientCorrection end

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

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute correction value" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        # Loop over all pairs of particles and neighbors within the kernel cutoff
        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi) do particle, neighbor, pos_diff, distance
            rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            volume = m_b / rho_b

            kernel_correction_coefficient[particle] += volume *
                                                       smoothing_kernel(system, distance,
                                                                        particle)
        end
    end

    return kernel_correction_coefficient
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

    # Use all other systems for the density summation
    @trixi_timeit timer() "compute correction value" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        # Loop over all pairs of particles and neighbors within the kernel cutoff
        foreach_point_neighbor(system, neighbor_system, system_coords, neighbor_coords,
                               semi) do particle, neighbor, pos_diff, distance
            rho_b = current_density(v_neighbor_system, neighbor_system, neighbor)
            m_b = hydrodynamic_mass(neighbor_system, neighbor)
            volume = m_b / rho_b

            # Use uncorrected kernel to compute correction coefficients
            W = kernel(system_smoothing_kernel(system), distance,
                       smoothing_length(system, particle))

            kernel_correction_coefficient[particle] += volume * W

            # Only consider particles with a distance > 0. See `src/general/smoothing_kernels.jl` for more details.
            if distance^2 > eps(initial_smoothing_length(system)^2)
                grad_W = kernel_grad(system_smoothing_kernel(system), pos_diff, distance,
                                     smoothing_length(system, particle))
                tmp = volume * grad_W
                for i in axes(dw_gamma, 1)
                    dw_gamma[i, particle] += tmp[i]
                end
            end
        end
    end

    for particle in eachparticle(system), i in axes(dw_gamma, 1)
        dw_gamma[i, particle] /= kernel_correction_coefficient[particle]
    end
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

# Arguments
- `blending_factor`: Blending factor between corrected and regular SPH gradient.
"""
struct BlendedGradientCorrection{ELTYPE <: Real}
    blending_factor::ELTYPE

    function BlendedGradientCorrection(blending_factor)
        return new{eltype(blending_factor)}(blending_factor)
    end
end

# Called only by DensityDiffusion and TLSPH
function compute_gradient_correction_matrix!(corr_matrix, system, coordinates, density_fun,
                                             semi)
    (; mass) = system

    set_zero!(corr_matrix)

    # Loop over all pairs of particles and neighbors within the kernel cutoff.
    foreach_point_neighbor(system, system, coordinates, coordinates,
                           semi) do particle, neighbor, pos_diff, distance
        volume = mass[neighbor] / density_fun(neighbor)

        grad_kernel = smoothing_kernel_grad(system, pos_diff, distance, particle)

        iszero(grad_kernel) && return

        result = volume * grad_kernel * pos_diff'

        @inbounds for j in 1:ndims(system), i in 1:ndims(system)
            corr_matrix[i, j, particle] -= result[i, j]
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
    @trixi_timeit timer() "compute correction matrix" foreach_system(semi) do neighbor_system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)
        v_neighbor_system = wrap_v(v_ode, neighbor_system, semi)

        neighbor_coords = current_coordinates(u_neighbor_system, neighbor_system)

        foreach_point_neighbor(system, neighbor_system, coordinates, neighbor_coords,
                               semi) do particle, neighbor, pos_diff, distance
            volume = hydrodynamic_mass(neighbor_system, neighbor) /
                     current_density(v_neighbor_system, neighbor_system, neighbor)
            smoothing_length_ = smoothing_length(system, particle)

            function compute_grad_kernel(correction, smoothing_kernel, pos_diff, distance,
                                         smoothing_length_, system, particle)
                return smoothing_kernel_grad(system, pos_diff, distance, particle)
            end

            # Compute gradient of corrected kernel
            function compute_grad_kernel(correction::MixedKernelGradientCorrection,
                                         smoothing_kernel, pos_diff, distance,
                                         smoothing_length_, system, particle)
                return corrected_kernel_grad(smoothing_kernel, pos_diff, distance,
                                             smoothing_length_, KernelCorrection(), system,
                                             particle)
            end

            grad_kernel = compute_grad_kernel(correction, smoothing_kernel, pos_diff,
                                              distance, smoothing_length_, system, particle)

            iszero(grad_kernel) && return

            L = volume * grad_kernel * pos_diff'

            # pos_diff is always x_a - x_b hence * -1 to switch the order to x_b - x_a
            @inbounds for j in 1:ndims(system), i in 1:ndims(system)
                corr_matrix[i, j, particle] -= L[i, j]
            end
        end
    end

    correction_matrix_inversion_step!(corr_matrix, system, semi)

    return corr_matrix
end

function correction_matrix_inversion_step!(corr_matrix, system, semi)
    @threaded semi for particle in eachparticle(system)
        L = extract_smatrix(corr_matrix, system, particle)

        # The matrix `L` only becomes singular when the particle and all neighbors
        # are collinear (in 2D) or lie all in the same plane (in 3D).
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
        if abs(det(L)) < 1.0f-9
            L_inv = I
        else
            L_inv = inv(L)
        end

        # Write inverse back to `corr_matrix`
        for j in 1:ndims(system), i in 1:ndims(system)
            @inbounds corr_matrix[i, j, particle] = L_inv[i, j]
        end
    end

    return corr_matrix
end

create_cache_correction(correction, density, NDIMS, nparticles) = (;)

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
    dw_gamma = Array{eltype(density)}(undef, NDIMS, n_particles)
    correction_matrix = Array{eltype(density), 3}(undef, NDIMS, NDIMS, n_particles)

    return (; kernel_correction_coefficient=similar(density), dw_gamma, correction_matrix)
end
