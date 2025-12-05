abstract type AbstractSmoothingKernel{NDIMS} end

@inline Base.ndims(::AbstractSmoothingKernel{NDIMS}) where {NDIMS} = NDIMS

@inline function kernel_grad(kernel, pos_diff, distance, h)
    # For `distance == 0`, the analytical gradient is zero, but the code divides by zero.
    # To account for rounding errors, we check if `distance` is almost zero.
    # Since the coordinates are in the order of the smoothing length `h`,
    # `distance^2` is in the order of `h^2`, hence the comparison `distance^2 < eps(h^2)`.
    # Note that this is faster than `distance < sqrt(eps(h^2))`.
    # Also note that `sqrt(eps(h^2)) != eps(h)`.
    distance^2 < eps(h^2) && return zero(pos_diff)

    return kernel_deriv(kernel, distance, h) / distance * pos_diff
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h, correction, system,
                                       particle)
    return kernel_grad(kernel, pos_diff, distance, h)
end

@inline function corrected_kernel_grad(kernel_, pos_diff, distance, h, ::KernelCorrection,
                                       system, particle)
    return (kernel_grad(kernel_, pos_diff, distance, h) .-
            kernel(kernel_, distance, h) * dw_gamma(system, particle)) /
           kernel_correction_coefficient(system, particle)
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h,
                                       corr::BlendedGradientCorrection, system,
                                       particle)
    (; blending_factor) = corr

    grad = kernel_grad(kernel, pos_diff, distance, h)
    return (1 - blending_factor) * grad +
           blending_factor * correction_matrix(system, particle) * grad
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h,
                                       ::GradientCorrection, system, particle)
    grad = kernel_grad(kernel, pos_diff, distance, h)
    return correction_matrix(system, particle) * grad
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h,
                                       ::MixedKernelGradientCorrection, system,
                                       particle)
    grad = corrected_kernel_grad(kernel, pos_diff, distance, h, KernelCorrection(),
                                 system, particle)
    return correction_matrix(system, particle) * grad
end

@doc raw"""
    GaussianKernel{NDIMS}()

Gaussian kernel given by
```math
W(r, h) = \frac{\sigma_d}{h^d} e^{-r^2/h^2}
```

where ``d`` is the number of dimensions and

- `` \sigma_2 = \frac{1}{\pi} `` for 2D,
- `` \sigma_3 = \frac{1}{\pi^{3/2}} `` for 3D.

This kernel function has an infinite support, but in practice,
it's often truncated at a certain multiple of ``h``, such as ``3h``.

In this implementation, the kernel is truncated at ``3h``,
so this kernel function has a compact support of ``[0, 3h]``.

The smoothing length is typically in the range ``[1.0\delta, 1.5\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).

Note:
This truncation makes this Kernel not conservative,
which is beneficial in regards to stability but makes it less accurate.
"""
struct GaussianKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

@inline @fastmath function kernel(kernel::GaussianKernel, r::Real, h)
    q = r / h

    # Zero out result if q >= 3
    result = ifelse(q < 3, normalization_factor(kernel, h) * exp(-q^2), zero(q))

    return result
end

@inline @fastmath function kernel_deriv(kernel::GaussianKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    # Zero out result if q >= 3
    result = ifelse(q < 3,
                    -2 * q * normalization_factor(kernel, h) * exp(-q^2) * inner_deriv,
                    zero(q))

    return result
end

@inline compact_support(::GaussianKernel, h) = 3 * h

@inline normalization_factor(::GaussianKernel{2}, h) = 1 / (pi * h^2)
# First convert `pi` to the type of `h` to preserve the type of `h`
@inline normalization_factor(::GaussianKernel{3}, h) = 1 / (oftype(h, pi)^(3 // 2) * h^3)

@doc raw"""
    SchoenbergCubicSplineKernel{NDIMS}()

Cubic spline kernel by [Schoenberg (1946)](@cite Schoenberg1946), given by
```math
    W(r, h) = \frac{1}{h^d} w(r/h)
```
with
```math
w(q) = \sigma \begin{cases}
    \frac{1}{4} (2 - q)^3 - (1 - q)^3   & \text{if } 0 \leq q < 1, \\
    \frac{1}{4} (2 - q)^3               & \text{if } 1 \leq q < 2, \\
    0                                   & \text{if } q \geq 2, \\
\end{cases}
```
where ``d`` is the number of dimensions and ``\sigma`` is a normalization constant given by
$\sigma =[\frac{2}{3}, \frac{10}{7 \pi}, \frac{1}{\pi}]$ in $[1, 2, 3]$ dimensions.

This kernel function has a compact support of ``[0, 2h]``.

For an overview of Schoenberg cubic, quartic and quintic spline kernels including
normalization factors, see [Price (2012)](@cite Price2012).
For an analytic formula for higher order Schoenberg kernels, see [Monaghan (1985)](@cite Monaghan1985).
The largest disadvantage of Schoenberg Spline Kernel is the rather non-smooth first derivative,
which can lead to increased noise compared to other kernel variants.

The smoothing length is typically in the range ``[1.1\delta, 1.3\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct SchoenbergCubicSplineKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

@muladd @inline function kernel(kernel::SchoenbergCubicSplineKernel, r::Real, h)
    q = r / h

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl.
    result = 1 * (2 - q)^3 / 4
    result = result - (q < 1) * (1 - q)^3

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result, zero(result))

    return result
end

@muladd @inline function kernel_deriv(kernel::SchoenbergCubicSplineKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl
    result = -3 * (2 - q)^2 / 4
    result = result + 3 * (q < 1) * (1 - q)^2

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result * inner_deriv,
                    zero(result))

    return result
end

@inline compact_support(::SchoenbergCubicSplineKernel, h) = 2 * h

# Note that `2 // 3 / h` saves one instruction but is significantly slower on GPUs (for now)
@inline normalization_factor(::SchoenbergCubicSplineKernel{1}, h) = 2 / (3 * h)
@inline normalization_factor(::SchoenbergCubicSplineKernel{2}, h) = 10 / (pi * h^2 * 7)
@inline normalization_factor(::SchoenbergCubicSplineKernel{3}, h) = 1 / (pi * h^3)

@doc raw"""
    SchoenbergQuarticSplineKernel{NDIMS}()

Quartic spline kernel by [Schoenberg (1946)](@cite Schoenberg1946), given by
```math
    W(r, h) = \frac{1}{h^d} w(r/h)
```
with
```math
w(q) = \sigma \begin{cases}
    \left(5/2 - q \right)^4 - 5\left(3/2 - q \right)^4
    + 10\left(1/2 - q \right)^4 & \text{if } 0 \leq q < \frac{1}{2}, \\
    \left(5/2 - q \right)^4 - 5\left(3/2 - q \right)^4
    & \text{if } \frac{1}{2} \leq q < \frac{3}{2}, \\
    \left(5/2 - q \right)^4 & \text{if } \frac{3}{2} \leq q < \frac{5}{2}, \\
    0 & \text{if } q \geq \frac{5}{2},
\end{cases}
```
where ``d`` is the number of dimensions and ``\sigma`` is a normalization constant given by
$\sigma =[\frac{1}{24}, \frac{96}{1199 \pi}, \frac{1}{20\pi}]$ in $[1, 2, 3]$ dimensions.

This kernel function has a compact support of ``[0, 2.5h]``.

For an overview of Schoenberg cubic, quartic and quintic spline kernels including
normalization factors, see [Price (2012)](@cite Price2012).
For an analytic formula for higher order Schoenberg kernels, see [Monaghan (1985)](@cite Monaghan1985).

The largest disadvantage of Schoenberg Spline Kernel are the rather non-smooth first derivative,
which can lead to increased noise compared to other kernel variants.

The smoothing length is typically in the range ``[1.1\delta, 1.5\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct SchoenbergQuarticSplineKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

# Note that `floating_point_number^integer_literal` is lowered to `Base.literal_pow`.
# Currently, specializations reducing this to simple multiplications exist only up
# to a power of three, see
# https://github.com/JuliaLang/julia/blob/34934736fa4dcb30697ac1b23d11d5ad394d6a4d/base/intfuncs.jl#L327-L339
# By using the `@fastpow` macro, we are consciously trading off some precision in the result
# for enhanced computational speed. This is especially useful in scenarios where performance
# is a higher priority than exact precision.
@fastpow @muladd @inline function kernel(kernel::SchoenbergQuarticSplineKernel, r::Real, h)
    q = r / h

    # Preserve the type of `q`
    q5_2 = (5 // 2 - q)
    q3_2 = (3 // 2 - q)
    q1_2 = (1 // 2 - q)

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl
    result = q5_2^4
    result = result - 5 * (q < 3 // 2) * q3_2^4
    result = result + 10 * (q < 1 // 2) * q1_2^4

    # Zero out result if q >= 5/2
    result = ifelse(q < 5 // 2, normalization_factor(kernel, h) * result, zero(result))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::SchoenbergQuarticSplineKernel,
                                               r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    # Preserve the type of `q`
    q5_2 = 5 // 2 - q
    q3_2 = 3 // 2 - q
    q1_2 = 1 // 2 - q

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl
    result = -4 * q5_2^3
    result = result + 20 * (q < 3 // 2) * q3_2^3
    result = result - 40 * (q < 1 // 2) * q1_2^3

    # Zero out result if q >= 5/2
    result = ifelse(q < 5 // 2, normalization_factor(kernel, h) * result * inner_deriv,
                    zero(result))

    return result
end

@inline compact_support(::SchoenbergQuarticSplineKernel, h) = 5 * h / 2

@inline normalization_factor(::SchoenbergQuarticSplineKernel{1}, h) = 1 / (24 * h)
@inline normalization_factor(::SchoenbergQuarticSplineKernel{2}, h) = 96 / (pi * h^2 * 1199)
@inline normalization_factor(::SchoenbergQuarticSplineKernel{3}, h) = 1 / (pi * h^3 * 20)

@doc raw"""
    SchoenbergQuinticSplineKernel{NDIMS}()

Quintic spline kernel by [Schoenberg (1946)](@cite Schoenberg1946), given by
```math
    W(r, h) = \frac{1}{h^d} w(r/h)
```
with
```math
w(q) = \sigma \begin{cases}
    (3 - q)^5 - 6(2 - q)^5 + 15(1 - q)^5    & \text{if } 0 \leq q < 1, \\
    (3 - q)^5 - 6(2 - q)^5                  & \text{if } 1 \leq q < 2, \\
    (3 - q)^5                               & \text{if } 2 \leq q < 3, \\
    0                                       & \text{if } q \geq 3,
\end{cases}
```
where ``d`` is the number of dimensions and ``\sigma`` is a normalization constant given by
$\sigma =[\frac{1}{120}, \frac{7}{478 \pi}, \frac{1}{120\pi}]$ in $[1, 2, 3]$ dimensions.

This kernel function has a compact support of ``[0, 3h]``.

For an overview of Schoenberg cubic, quartic and quintic spline kernels including
normalization factors, see [Price (2012)](@cite Price2012).
For an analytic formula for higher order Schoenberg kernels, see [Monaghan (1985)](@cite Monaghan1985).

The largest disadvantage of Schoenberg Spline Kernel are the rather non-smooth first derivative,
which can lead to increased noise compared to other kernel variants.

The smoothing length is typically in the range ``[1.1\delta, 1.5\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct SchoenbergQuinticSplineKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

@fastpow @muladd @inline function kernel(kernel::SchoenbergQuinticSplineKernel, r::Real, h)
    q = r / h
    q3 = (3 - q)
    q2 = (2 - q)
    q1 = (1 - q)

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl
    result = q3^5
    result = result - 6 * (q < 2) * q2^5
    result = result + 15 * (q < 1) * q1^5

    # Zero out result if q >= 3
    result = ifelse(q < 3, normalization_factor(kernel, h) * result, zero(result))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::SchoenbergQuinticSplineKernel,
                                               r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv
    q3 = (3 - q)
    q2 = (2 - q)
    q1 = (1 - q)

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl
    result = -5 * q3^4
    result = result + 30 * (q < 2) * q2^4
    result = result - 75 * (q < 1) * q1^4

    # Zero out result if q >= 3
    result = ifelse(q < 3, normalization_factor(kernel, h) * result * inner_deriv,
                    zero(result))

    return result
end

@inline compact_support(::SchoenbergQuinticSplineKernel, h) = 3 * h

@inline normalization_factor(::SchoenbergQuinticSplineKernel{1}, h) = 1 / (120 * h)
@inline normalization_factor(::SchoenbergQuinticSplineKernel{2}, h) = 7 / (pi * h^2 * 478)
@inline normalization_factor(::SchoenbergQuinticSplineKernel{3}, h) = 1 / (pi * h^3 * 120)

abstract type AbstractWendlandKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

# Compact support for all Wendland kernels
@inline compact_support(::AbstractWendlandKernel, h) = 2 * h

@doc raw"""
    WendlandC2Kernel{NDIMS}()

Wendland C2 kernel [Wendland1995](@cite), a piecewise polynomial function designed
to have compact support and to be twice continuously differentiable everywhere. Given by

```math
 W(r, h) = \frac{1}{h^d} w(r/h)
```

with

```math
w(q) = \sigma \begin{cases}
    (1 - q/2)^4 (2q + 1)  & \text{if } 0 \leq q < 2, \\
    0                     & \text{if } q \geq 2,
\end{cases}
```

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor dependent on the dimension.
The normalization factor `` \sigma `` is `` 40/7\pi `` in two dimensions or `` 21/2\pi `` in three dimensions.

This kernel function has a compact support of `` [0, 2h] ``.

For a detailed discussion on Wendland functions and their applications in SPH, see [Dehnen (2012)](@cite Dehnen2012).
The smoothness of these functions is also the largest disadvantage as they lose details at sharp corners.

The smoothing length is typically in the range ``[1.2\delta, 2\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct WendlandC2Kernel{NDIMS} <: AbstractWendlandKernel{NDIMS} end

@fastpow @inline function kernel(kernel::WendlandC2Kernel, r::Real, h)
    q = r / h

    result = (1 - q / 2)^4 * (2 * q + 1)

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result, zero(q))

    return result
end

@inline function kernel_deriv(kernel::WendlandC2Kernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    result = -5 * (1 - q / 2)^3 * q

    # Zero out result if q >= 2
    result = ifelse(q < 2,
                    normalization_factor(kernel, h) * result * inner_deriv, zero(q))

    return result
end

# Note that `7 // 4` saves one instruction but is significantly slower on GPUs (for now)
@inline normalization_factor(::WendlandC2Kernel{2}, h) = 7 / (pi * h^2 * 4)
@inline normalization_factor(::WendlandC2Kernel{3}, h) = 21 / (pi * h^3 * 16)

@doc raw"""
    WendlandC4Kernel{NDIMS}()

Wendland C4 kernel [Wendland1995](@cite), a piecewise polynomial function designed to have compact support and to
be four times continuously differentiable everywhere. Given by

```math
 W(r, h) = \frac{1}{h^d} w(r/h)
```

with

```math
w(q) = \sigma \begin{cases}
    (1 - q/2)^6 (35q^2 / 12 + 3q + 1)   & \text{if } 0 \leq q < 2, \\
    0                                   & \text{if } q \geq 2,
\end{cases}
```

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor dependent
on the dimension. The normalization factor `` \sigma `` is `` 9 / \pi `` in two dimensions or `` 495 / 32\pi `` in three dimensions.

This kernel function has a compact support of `` [0, 2h] ``.

For a detailed discussion on Wendland functions and their applications in SPH, see [Dehnen (2012)](@cite Dehnen2012).
The smoothness of these functions is also the largest disadvantage as they lose details at sharp corners.

The smoothing length is typically in the range ``[1.5\delta, 2.3\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct WendlandC4Kernel{NDIMS} <: AbstractWendlandKernel{NDIMS} end

@fastpow @inline function kernel(kernel::WendlandC4Kernel, r::Real, h)
    q = r / h

    result = (1 - q / 2)^6 * (35 * q^2 / 12 + 3 * q + 1)

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result, zero(q))

    return result
end

@fastpow @inline function kernel_deriv(kernel::WendlandC4Kernel, r::Real, h)
    q = r / h

    derivative = -7 * q / 3 * (2 + 5 * q) * (1 - q / 2)^5

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * derivative / h,
                    zero(derivative))

    return result
end

@inline normalization_factor(::WendlandC4Kernel{2}, h) = 9 / (pi * h^2 * 4)
@inline normalization_factor(::WendlandC4Kernel{3}, h) = 495 / (pi * h^3 * 256)

@doc raw"""
    WendlandC6Kernel{NDIMS}()

Wendland C6 kernel [Wendland1995](@cite), a piecewise polynomial function designed to have compact support and
to be six times continuously differentiable everywhere. Given by:

```math
W(r, h) = \frac{1}{h^d} w(r/h)
```

with:

```math
w(q) = \sigma \begin{cases}
    (1 - q / 2)^8 (4q^3 + 25q^2 / 4 + 4q + 1)   & \text{if } 0 \leq q < 2, \\
    0                                           & \text{if } q \geq 2,
\end{cases}
```

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor dependent
on the dimension. The normalization factor `` \sigma `` is `` 78 / 7 \pi`` in two dimensions or `` 1365 / 64\pi`` in three dimensions.

This kernel function has a compact support of `` [0, 2h] ``.

For a detailed discussion on Wendland functions and their applications in SPH, [Dehnen (2012)](@cite Dehnen2012).
The smoothness of these functions is also the largest disadvantage as they lose details at sharp corners.

The smoothing length is typically in the range ``[1.7\delta, 2.5\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct WendlandC6Kernel{NDIMS} <: AbstractWendlandKernel{NDIMS} end

@fastpow @inline function kernel(kernel::WendlandC6Kernel, r::Real, h)
    q = r / h

    result = (1 - q / 2)^8 * (4 * q^3 + 25 * q^2 / 4 + 4 * q + 1)

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result, zero(q))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::WendlandC6Kernel, r::Real, h)
    q = r / h

    derivative = -11 * q / 4 * (8 * q^2 + 7 * q + 2) * (1 - q / 2)^7

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * derivative / h,
                    zero(derivative))

    return result
end

@inline normalization_factor(::WendlandC6Kernel{2}, h) = 39 / (pi * h^2 * 14)
@inline normalization_factor(::WendlandC6Kernel{3}, h) = 1365 / (pi * h^3 * 512)

@doc raw"""
    Poly6Kernel{NDIMS}()

Poly6 kernel, a commonly used kernel in SPH literature [Mueller2003](@cite),
especially in computer graphics contexts. It is defined as

```math
W(r, h) = \frac{1}{h^d} w(r/h)
```


with

```math
w(q) = \sigma \begin{cases}
    (1 - q^2)^3    & \text{if } 0 \leq q < 1, \\
    0              & \text{if } q \geq 1,
\end{cases}
```

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor
that depends on the dimension. The normalization factor `` \sigma `` is `` 4 / \pi``
in two dimensions or `` 315 / 64\pi`` in three dimensions.

This kernel function has a compact support of `` [0, h] ``.

Poly6 is well-known for its computational simplicity, though it's worth noting that there are
other kernels that might offer better accuracy for hydrodynamic simulations. Furthermore,
its derivatives are not that smooth, which can lead to stability problems.
It is also susceptible to clumping.

The smoothing length is typically in the range ``[1.5\delta, 2.5\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct Poly6Kernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

@inline function kernel(kernel::Poly6Kernel, r::Real, h)
    q = r / h

    result = (1 - q^2)^3

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) * result, zero(q))

    return result
end

@inline function kernel_deriv(kernel::Poly6Kernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    result = -6 * q * (1 - q^2)^2

    # Zero out result if q >= 1
    result = ifelse(q < 1,
                    result * normalization_factor(kernel, h) * inner_deriv, zero(q))
    return result
end

@inline compact_support(::Poly6Kernel, h) = h

# Note that `315 // 64` saves one instruction but is significantly slower on GPUs (for now)
@inline normalization_factor(::Poly6Kernel{2}, h) = 4 / (pi * h^2)
@inline normalization_factor(::Poly6Kernel{3}, h) = 315 / (pi * h^3 * 64)

@doc raw"""
    SpikyKernel{NDIMS}()

The Spiky kernel is another frequently used kernel in SPH, especially due to its desirable
properties in preserving features near boundaries in fluid simulations [Mueller2003](@cite).
It is defined as:

```math
 W(r, h) = \frac{1}{h^d} w(r/h)
```

with:

```math
w(q) = \sigma \begin{cases}
    (1 - q)^3    & \text{if } 0 \leq q < 1, \\
    0            & \text{if } q \geq 1,
\end{cases}
```

where `` d `` is the number of dimensions and the normalization factor `` \sigma `` is `` 10 / \pi``
in two dimensions or `` 15 / \pi`` in three dimensions.

This kernel function has a compact support of `` [0, h] ``.

The Spiky kernel is particularly known for its sharp gradients, which can help to preserve
sharp features in fluid simulations, especially near solid boundaries.
These sharp gradients at the boundary are also the largest disadvantage as they can lead to instability.

The smoothing length is typically in the range ``[1.5\delta, 3.0\delta]``,
where ``\delta`` is the typical particle spacing.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct SpikyKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

@inline function kernel(kernel::SpikyKernel, r::Real, h)
    q = r / h

    result = (1 - q)^3

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) * result, zero(q))

    return result
end

@inline function kernel_deriv(kernel::SpikyKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    result = -3 * (1 - q)^2

    # Zero out result if q >= 1
    result = ifelse(q < 1, result * normalization_factor(kernel, h) * inner_deriv, zero(q))

    return result
end

@inline compact_support(::SpikyKernel, h) = h

@inline normalization_factor(::SpikyKernel{2}, h) = 10 / (pi * h^2)
@inline normalization_factor(::SpikyKernel{3}, h) = 15 / (pi * h^3)

@doc raw"""
    LaguerreGaussKernel{NDIMS}()

Truncated Laguerre–Gauss kernel (fourth-order smoothing) as defined in
[Wang2024](@cite). Its radial form uses ``q = r/h`` and is truncated at
``q = 2`` (compact support ``2h``):

```math
W(r,h) = \frac{C_d}{h^d}
\left(1 - \frac{q^2}{2} + \frac{q^4}{6}\right) e^{-q^2},
\quad q = \frac{r}{h}, \quad 0 \le q \le 2,
```

where the dimension-dependent normalization constants ``C_d`` are chosen so that
the truncated kernel is normalized,

```math
\int_{\lVert x \rVert \le 2h} W(\lVert x \rVert, h)\,\mathrm{d}^d x = 1.
```

Explicitly, for ``d = 1,2,3`` we obtain

```math
C_1 = \frac{1}{
  2\Bigl(\frac{7\sqrt{\pi}}{16}\,\mathrm{erf}(2)
         - \frac{5}{12} e^{-4}\Bigr)}
    \approx 0.6542878,
```

```math
C_2 = \frac{6}{\pi\bigl(5 - 17 e^{-4}\bigr)}
    \approx 0.4073381,
```

```math
C_3 = \frac{1}{
  4\pi\Bigl(\frac{7\sqrt{\pi}}{32}\,\mathrm{erf}(2)
           - \frac{77}{24} e^{-4}\Bigr)}
    \approx 0.2432461.
```

These values differ from the original infinite-support normalization factors in
[Wang2024](@cite) because the kernel is truncated at ``q = 2`` and then
renormalized.

Recommended practical choice from the paper: use ``h \approx 1.3 \Delta x``
and the same cut-off as Wendland (``2h``) for comparable cost. Negative lobes
enforce the vanishing second moment (fourth-order smoothing) while remaining
stable in Eulerian / total Lagrangian SPH with relaxed particles.

For general information and usage see [Smoothing Kernels](@ref smoothing_kernel).
"""
struct LaguerreGaussKernel{NDIMS} <: AbstractSmoothingKernel{NDIMS} end

@fastpow @inline function kernel(kernel::LaguerreGaussKernel, r::Real, h)
    q = r / h

    # polynomial part: 1 - s^2/2 + s^4/6
    poly = 1 - q^2 / 2 + q^4 / 6

    # zero out for s ≥ 2
    return ifelse(q < 2, normalization_factor(kernel, h) * poly * exp(-q^2), zero(q))
end

@inline function kernel_deriv(kernel::LaguerreGaussKernel, r::Real, h)
    invh = 1 / h
    q = r * invh

    # dg/dq = (q/3)*(-q^4 + 5q^2 - 9) * exp(-q^2)
    poly = ((-q^2 + 5) * q^2 - 9) * (q / 3)

    return ifelse(q < 2, normalization_factor(kernel, h) * exp(-q^2) * poly * invh, zero(q))
end

@inline compact_support(::LaguerreGaussKernel, h) = 2 * h
# Original normalization factors as in Wang2024
# @inline normalization_factor(::LaguerreGaussKernel{1}, h) = (8 / (5 * sqrt(pi))) / h
# @inline normalization_factor(::LaguerreGaussKernel{2}, h) = (3 / (pi)) / (h^2)
# @inline normalization_factor(::LaguerreGaussKernel{3}, h) = (8 / (pi^(3 // 2))) / (h^3)

# Renormalized to the truncated integral over [0,2h]
@inline function normalization_factor(kernel::LaguerreGaussKernel{1}, h)
    # C = 1/(2*(7 * sqrt(pi) / 16) * erf(2) - (5 / 12) * exp(-4))
    C = oftype(h, 0.65428780253539)
    # C' = C/h
    return C / h
end

@inline function normalization_factor(kernel::LaguerreGaussKernel{2}, h)
    # C = 2 * pi * (5 - 17 * exp(-4)) / 12
    C = oftype(h, 2.454963094351984)
    # C' = 1 / (h^2 * C)
    return 1 / (h^2 * C)
end

@inline function normalization_factor(kernel::LaguerreGaussKernel{3}, h)
    # C = (7 * sqrt(pi) / 32) * erf(2) - (77 / 24) * exp(-4)
    C = oftype(h, 0.3271479336905373)

    # 4*pi cannot be pulled into C otherwise the test fails because of differences in rounding
    return 1 / (C * 4 * pi * h^3)
end
