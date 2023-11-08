@doc raw"""
    SmoothingKernel{NDIMS}

An abstract supertype of all smoothing kernels. The type parameter `NDIMS` encodes the number
of dimensions.

We provide the following smoothing kernels:

| Smoothing Kernel                          | Compact Support   |
| :---------------------------------------- | :---------------- |
| [`SchoenbergCubicSplineKernel`](@ref)     | $[0, 2h]$         |
| [`SchoenbergQuarticSplineKernel`](@ref)   | $[0, 2.5h]$       |
| [`SchoenbergQuinticSplineKernel`](@ref)   | $[0, 3h]$         |

!!! note "Usage"
    The kernel can be called as
    ```
    TrixiParticles.kernel(::SmoothingKernel{NDIMS}, r, h)
    ```
    The length of the compact support can be obtained as
    ```
    TrixiParticles.compact_support(::SmoothingKernel{NDIMS}, h)
    ```

    Note that ``r`` has to be a scalar, so in the context of SPH, the kernel
    should be used as
    ```math
    W(\Vert r_a - r_b \Vert, h).
    ```

    The gradient required in SPH,
    ```math
        \nabla_{r_a} W(\Vert r_a - r_b \Vert, h)
    ```
    can be called as
    ```
    TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)
    ```
    where `pos_diff` is $r_a - r_b$ and `distance` is $\Vert r_a - r_b \Vert$.
"""
abstract type SmoothingKernel{NDIMS} end

@inline Base.ndims(::SmoothingKernel{NDIMS}) where {NDIMS} = NDIMS

@inline function kernel_grad(kernel, pos_diff, distance, h)
    return kernel_deriv(kernel, distance, h) * pos_diff / distance
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h, correction, system,
                                       particle)
    return kernel_grad(kernel, pos_diff, distance, h)
end

@inline function corrected_kernel_grad(kernel, pos_diff, distance, h,
                                       ::KernelGradientCorrection, system, particle)
    return (kernel_grad(kernel, pos_diff, distance, h) .- dw_gamma(system, particle)) /
           kernel_correction_coefficient(system, particle)
end

@doc raw"""
    GaussianKernel{NDIMS}()

Gaussian kernel given by
```math
W(r, h) = \frac{\sigma_d}{h^d} e^{-r^2/h^2}
```

where ``d`` is the number of dimensions and

- `` \sigma_2 = \frac{1}{\pi} `` for 2D
- `` \sigma_3 = \frac{1}{\pi^{3/2}} `` for 3D

This kernel function has an infinite support, but in practice,
it's often truncated at a certain multiple of ``h``, such as ``3h``.

In this implementation, the kernel is truncated at `3h`,
so this kernel function has a compact support of ``[0, 3h]``.

Note:
This truncation makes this Kernel not conservative,
which is beneficial in regards to stability but makes it less accurate.
"""
struct GaussianKernel{NDIMS} <: SmoothingKernel{NDIMS} end

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
@inline normalization_factor(::GaussianKernel{3}, h) = 1 / (pi^(3 / 2) * h^3)

@doc raw"""
    SchoenbergCubicSplineKernel{NDIMS}()

Cubic spline kernel by Schoenberg (Schoenberg, 1946), given by
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
normalization factors, see (Price, 2012).
For an analytic formula for higher order Schoenberg kernels, see (Monaghan, 1985).
The largest disadvantage of Schoenberg Spline Kernel is the rather non-smooth first derivative,
which can lead to increased noise compared to other kernel variants.

For general information and usage see [`SmoothingKernel`](@ref).

## References:
- Daniel J. Price. "Smoothed particle hydrodynamics and magnetohydrodynamics".
  In: Journal of Computational Physics 231.3 (2012), pages 759-794.
  [doi: 10.1016/j.jcp.2010.12.011](https://doi.org/10.1016/j.jcp.2010.12.011)
- Joseph J. Monaghan. "Particle methods for hydrodynamics".
  In: Computer Physics Reports 3.2 (1985), pages 71–124.
  [doi: 10.1016/0167-7977(85)90010-3](https://doi.org/10.1016/0167-7977(85)90010-3)
- Isaac J. Schoenberg. "Contributions to the problem of approximation of equidistant data by analytic functions.
  Part B. On the problem of osculatory interpolation. A second class of analytic approximation formulae."
  In: Quarterly of Applied Mathematics 4.2 (1946), pages 112–141.
  [doi: 10.1090/QAM/16705](https://doi.org/10.1090/QAM/16705)
"""
struct SchoenbergCubicSplineKernel{NDIMS} <: SmoothingKernel{NDIMS} end

function kernel(kernel::SchoenbergCubicSplineKernel, r::Real, h)
    q = r / h

    if q >= 2
        return 0.0
    end

    result = 1 / 4 * (2 - q)^3

    if q < 1
        result -= (1 - q)^3
    end

    return normalization_factor(kernel, h) * result
end

function kernel_deriv(kernel::SchoenbergCubicSplineKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    if q >= 2
        return 0.0
    end

    result = -3 / 4 * (2 - q)^2

    if q < 1
        result += 3 * (1 - q)^2
    end

    return normalization_factor(kernel, h) * result * inner_deriv
end

@inline compact_support(::SchoenbergCubicSplineKernel, h) = 2 * h

@inline normalization_factor(::SchoenbergCubicSplineKernel{1}, h) = 2 / 3h
@inline normalization_factor(::SchoenbergCubicSplineKernel{2}, h) = 10 / (7 * pi * h^2)
@inline normalization_factor(::SchoenbergCubicSplineKernel{3}, h) = 1 / (pi * h^3)

@doc raw"""
    SchoenbergQuarticSplineKernel{NDIMS}()

Quartic spline kernel by Schoenberg (Schoenberg, 1946), given by
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
normalization factors, see (Price, 2012).
For an analytic formula for higher order Schoenberg kernels, see (Monaghan, 1985).

The largest disadvantage of Schoenberg Spline Kernel are the rather non-smooth first derivative,
which can lead to increased noise compared to other kernel variants.
For general information and usage see [`SmoothingKernel`](@ref).

## References:
- Daniel J. Price. "Smoothed particle hydrodynamics and magnetohydrodynamics".
  In: Journal of Computational Physics 231.3 (2012), pages 759-794.
  [doi: 10.1016/j.jcp.2010.12.011](https://doi.org/10.1016/j.jcp.2010.12.011)
- Joseph J. Monaghan. "Particle methods for hydrodynamics".
  In: Computer Physics Reports 3.2 (1985), pages 71–124.
  [doi: 10.1016/0167-7977(85)90010-3](https://doi.org/10.1016/0167-7977(85)90010-3)
- Isaac J. Schoenberg. "Contributions to the problem of approximation of equidistant data by analytic functions.
  Part B. On the problem of osculatory interpolation. A second class of analytic approximation formulae."
  In: Quarterly of Applied Mathematics 4.2 (1946), pages 112–141.
  [doi: 10.1090/QAM/16705](https://doi.org/10.1090/QAM/16705)
"""
struct SchoenbergQuarticSplineKernel{NDIMS} <: SmoothingKernel{NDIMS} end

# Note that `floating_point_number^integer_literal` is lowered to `Base.literal_pow`.
# Currently, specializations reducing this to simple multiplications exist only up
# to a power of three, see
# https://github.com/JuliaLang/julia/blob/34934736fa4dcb30697ac1b23d11d5ad394d6a4d/base/intfuncs.jl#L327-L339
# Higher powers just call a generic power implementation. Here, we accept to lose
# some precision but gain performance by using plain multiplications instead via
# `@fastpow`.
@muladd @fastpow @inline function kernel(kernel::SchoenbergQuarticSplineKernel, r::Real, h)
    q = r / h

    q5_2_pow4 = (5 / 2 - q)^4
    q3_2_pow4 = (3 / 2 - q)^4
    q1_2_pow4 = (1 / 2 - q)^4

    # We do not use `+=` or `-=` since these are not recognized by MuladdMacro.jl.
    result = q5_2_pow4
    result = result - 5 * (q < 3 / 2) * q3_2_pow4
    result = result + 10 * (q < 1 / 2) * q1_2_pow4

    # Zero out result if q >= 5/2
    result = ifelse(q < 5 / 2, normalization_factor(kernel, h) * result, zero(result))

    return result
end

function kernel_deriv(kernel::SchoenbergQuarticSplineKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    if q >= 5 / 2
        return 0.0
    end

    result = -4 * (5 / 2 - q)^3

    if q < 3 / 2
        result += 20 * (3 / 2 - q)^3

        if q < 1 / 2
            result -= 40 * (1 / 2 - q)^3
        end
    end

    return normalization_factor(kernel, h) * result * inner_deriv
end

@inline compact_support(::SchoenbergQuarticSplineKernel, h) = 2.5 * h

@inline normalization_factor(::SchoenbergQuarticSplineKernel{1}, h) = 1 / 24h
@inline normalization_factor(::SchoenbergQuarticSplineKernel{2}, h) = 96 / (1199 * pi * h^2)
@inline normalization_factor(::SchoenbergQuarticSplineKernel{3}, h) = 1 / (20 * pi * h^3)

@doc raw"""
    SchoenbergQuinticSplineKernel{NDIMS}()

Quintic spline kernel by Schoenberg (Schoenberg, 1946), given by
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
normalization factors, see (Price, 2012).
For an analytic formula for higher order Schoenberg kernels, see (Monaghan, 1985).

The largest disadvantage of Schoenberg Spline Kernel are the rather non-smooth first derivative,
which can lead to increased noise compared to other kernel variants.
For general information and usage see [`SmoothingKernel`](@ref).

## References:
- Daniel J. Price. "Smoothed particle hydrodynamics and magnetohydrodynamics".
  In: Journal of Computational Physics 231.3 (2012), pages 759-794.
  [doi: 10.1016/j.jcp.2010.12.011](https://doi.org/10.1016/j.jcp.2010.12.011)
- Joseph J. Monaghan. "Particle methods for hydrodynamics".
  In: Computer Physics Reports 3.2 (1985), pages 71–124.
  [doi: 10.1016/0167-7977(85)90010-3](https://doi.org/10.1016/0167-7977(85)90010-3)
- Isaac J. Schoenberg. "Contributions to the problem of approximation of equidistant data by analytic functions.
  Part B. On the problem of osculatory interpolation. A second class of analytic approximation formulae."
  In: Quarterly of Applied Mathematics 4.2 (1946), pages 112–141.
  [doi: 10.1090/QAM/16705](https://doi.org/10.1090/QAM/16705)
"""
struct SchoenbergQuinticSplineKernel{NDIMS} <: SmoothingKernel{NDIMS} end

function kernel(kernel::SchoenbergQuinticSplineKernel, r::Real, h)
    q = r / h

    if q >= 3
        return 0.0
    end

    result = (3 - q)^5

    if q < 2
        result -= 6 * (2 - q)^5

        if q < 1
            result += 15 * (1 - q)^5
        end
    end

    return normalization_factor(kernel, h) * result
end

function kernel_deriv(kernel::SchoenbergQuinticSplineKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    if q >= 3
        return 0.0
    end

    result = -5 * (3 - q)^4

    if q < 2
        result += 30 * (2 - q)^4

        if q < 1
            result -= 75 * (1 - q)^4
        end
    end

    return normalization_factor(kernel, h) * result * inner_deriv
end

@inline compact_support(::SchoenbergQuinticSplineKernel, h) = 3 * h

@inline normalization_factor(::SchoenbergQuinticSplineKernel{1}, h) = 1 / 120h
@inline normalization_factor(::SchoenbergQuinticSplineKernel{2}, h) = 7 / (478 * pi * h^2)
@inline normalization_factor(::SchoenbergQuinticSplineKernel{3}, h) = 1 / (120 * pi * h^3)

abstract type WendlandKernel{NDIMS} <: SmoothingKernel{NDIMS} end

# Compact support for all Wendland kernels
@inline compact_support(::WendlandKernel, h) = h

@doc raw"""
    WendlandC2Kernel{NDIMS}()

Wendland C2 kernel (Wendland, 1995), a piecewise polynomial function designed to have compact support and to
be twice continuously differentiable everywhere. Given by

\[ W(r, h) = \frac{1}{h^d} w(r/h) \]

with

\[
w(q) = \sigma \begin{cases}
    (1 - q)^4 (4q + 1)    & \text{if } 0 \leq q < 1, \\
    0                     & \text{if } q \geq 1,
\end{cases}
\]

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor dependent on the dimension.
The normalization factor `` \sigma `` is `` 40/7\pi `` in two dimensions or `` 21/2\pi `` in three dimensions.

This kernel function has a compact support of `` [0, h] ``.

For a detailed discussion on Wendland functions and their applications in SPH, see (Dehnen & Aly, 2012).
The smoothness of these functions is also the largest disadvantage as they lose details at sharp corners.

## References:
- Walter Dehnen & Hassan Aly.
"Improving convergence in smoothed particle hydrodynamics simulations without pairing instability".
In: Monthly Notices of the Royal Astronomical Society 425.2 (2012), pages 1068-1082.
[doi: 10.1111/j.1365-2966.2012.21439.x](https://doi.org/10.1111/j.1365-2966.2012.21439.x)

- Holger Wendland.
 "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree."
 In: Advances in computational Mathematics 4 (1995): 389-396.
 [doi: 10.1007/BF02123482](https://doi.org/10.1007/BF02123482)

"""
struct WendlandC2Kernel{NDIMS} <: WendlandKernel{NDIMS} end

@fastpow @inline function kernel(kernel::WendlandC2Kernel, r::Real, h)
    q = r / h

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) * (1 - q)^4 * (4q + 1), zero(q))

    return result
end

@fastpow @inline function kernel_deriv(kernel::WendlandC2Kernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    q1_3 = (1 - q)^3
    q1_4 = (1 - q)^4

    # Zero out result if q >= 1
    result = ifelse(q < 1,
                    normalization_factor(kernel, h) * (-4 * q1_3 * (4q + 1) + q1_4 * 4) *
                    inner_deriv, zero(q))

    return result
end

@inline normalization_factor(::WendlandC2Kernel{2}, h) = 7 / (pi * h^2)
@inline normalization_factor(::WendlandC2Kernel{3}, h) = 21 / (2pi * h^3)

@doc raw"""
    WendlandC4Kernel{NDIMS}()

Wendland C4 kernel, a piecewise polynomial function designed to have compact support and to
be four times continuously differentiable everywhere. Given by

\[ W(r, h) = \frac{1}{h^d} w(r/h) \]

with

\[
w(q) = \sigma \begin{cases}
    (1 - q)^6 * (35q^2 / 3 + 6q + 1)   & \text{if } 0 \leq q < 1, \\
    0                                  & \text{if } q \geq 1,
\end{cases}
\]

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor dependent
on the dimension. The exact value of `` \sigma `` needs to be determined based on the dimension to ensure proper normalization.

This kernel function has a compact support of `` [0, h] ``.

For a detailed discussion on Wendland functions and their applications in SPH, see (Dehnen & Aly, 2012).
The smoothness of these functions is also the largest disadvantage as they loose details at sharp corners.

## References:
- Walter Dehnen & Hassan Aly.
"Improving convergence in smoothed particle hydrodynamics simulations without pairing instability".
In: Monthly Notices of the Royal Astronomical Society 425.2 (2012), pages 1068-1082.
[doi: 10.1111/j.1365-2966.2012.21439.x](https://doi.org/10.1111/j.1365-2966.2012.21439.x)

- Holger Wendland.
 "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree."
 In: Advances in computational Mathematics 4 (1995): 389-396.
 [doi: 10.1007/BF02123482](https://doi.org/10.1007/BF02123482)
"""
struct WendlandC4Kernel{NDIMS} <: WendlandKernel{NDIMS} end

@fastpow @inline function kernel(kernel::WendlandC4Kernel, r::Real, h)
    q = r / h

    # Zero out result if q >= 1
    result = ifelse(q < 1,
                    normalization_factor(kernel, h) * (1 - q)^6 * (35q^2 / 3 + 6q + 1),
                    zero(q))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::WendlandC4Kernel, r::Real, h)
    q = r / h
    term1 = (1 - q)^6 * (6 + 70 / 3 * q)
    term2 = 6 * (1 - q)^5 * (1 + 6q + 35 / 3 * q^2)
    derivative = term1 - term2

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) * derivative / h,
                    zero(derivative))

    return result
end

@inline normalization_factor(::WendlandC4Kernel{2}, h) = 9 / (pi * h^2)
@inline normalization_factor(::WendlandC4Kernel{3}, h) = 495 / (32pi * h^3)

@doc raw"""
    WendlandC6Kernel{NDIMS}()

Wendland C6 kernel, a piecewise polynomial function designed to have compact support and to be six times continuously differentiable everywhere. Given by:

\[ W(r, h) = \frac{1}{h^d} w(r/h) \]

with:

\[
w(q) = \sigma \begin{cases}
    (1 - q)^8 (32q^3 + 25q^2 + 8q + 1)    & \text{if } 0 \leq q < 1, \\
    0                                     & \text{if } q \geq 1,
\end{cases}
\]

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor dependent
on the dimension. The exact value of `` \sigma `` needs to be determined based on the dimension to ensure proper normalization.

This kernel function has a compact support of `` [0, h] ``.

For a detailed discussion on Wendland functions and their applications in SPH, see (Dehnen & Aly, 2012).
The smoothness of these functions is also the largest disadvantage as they loose details at sharp corners.

## References:
- Walter Dehnen & Hassan Aly.
"Improving convergence in smoothed particle hydrodynamics simulations without pairing instability".
In: Monthly Notices of the Royal Astronomical Society 425.2 (2012), pages 1068-1082.
[doi: 10.1111/j.1365-2966.2012.21439.x](https://doi.org/10.1111/j.1365-2966.2012.21439.x)

- Holger Wendland.
 "Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree."
 In: Advances in computational Mathematics 4 (1995): 389-396.
 [doi: 10.1007/BF02123482](https://doi.org/10.1007/BF02123482)
"""
struct WendlandC6Kernel{NDIMS} <: WendlandKernel{NDIMS} end

@fastpow @inline function kernel(kernel::WendlandC6Kernel, r::Real, h)
    q = r / h

    # Zero out result if q >= 1
    result = ifelse(q < 1,
                    normalization_factor(kernel, h) * (1 - q)^8 * (32q^3 + 25q^2 + 8q + 1),
                    zero(q))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::WendlandC6Kernel, r::Real, h)
    q = r / h
    term1 = -8 * (1 - q)^7 * (32q^3 + 25q^2 + 8q + 1)
    term2 = (1 - q)^8 * (96q^2 + 50q + 8)
    derivative = term1 + term2

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) * derivative / h, zero(derivative))

    return result
end

@inline normalization_factor(::WendlandC6Kernel{2}, h) = 78 / (7pi * h^2)
@inline normalization_factor(::WendlandC6Kernel{3}, h) = 1365 / (64pi * h^3)

@doc raw"""
    Poly6Kernel{NDIMS}()

Poly6 kernel, a commonly used kernel in SPH literature, especially in computer graphics contexts. It is defined as

\[ W(r, h) = \frac{1}{h^d} w(r/h) \]

with

\[
w(q) = \sigma \begin{cases}
    (1 - q^2)^3    & \text{if } 0 \leq q < 1, \\
    0              & \text{if } q \geq 1,
\end{cases}
\]

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor that depends
on the dimension. The exact value of `` \sigma `` needs to be determined based on the dimension to ensure proper normalization.

This kernel function has a compact support of `` [0, h] ``.

Poly6 is well-known for its computational simplicity, though it's worth noting that there are
other kernels that might offer better accuracy for hydrodynamic simulations. Furthermore,
its derivatives are not that smooth, which can lead to stability problems.
It is also susceptible to clumping.

## References:
- Matthias Müller, David Charypar, and Markus Gross. "Particle-based fluid simulation for interactive applications".
  In: Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association. 2003, pages 154-159.
  [doi: 10.5555/846276.846298](https://doi.org/10.5555/846276.846298)
"""
struct Poly6Kernel{NDIMS} <: SmoothingKernel{NDIMS} end

@inline function kernel(kernel::Poly6Kernel, r::Real, h)
    q = r / h

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) *(1 - q^2)^3, zero(q))

    return result
end

@inline function kernel_deriv(kernel::Poly6Kernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    # Zero out result if q >= 1
    result = ifelse(q < 1, -6 * q * (1 - q^2)^2 * normalization_factor(kernel, h) * inner_deriv,
    zero(q))
    return result
end

@inline compact_support(::Poly6Kernel, h) = h

@inline normalization_factor(::Poly6Kernel{2}, h) = 4 / (pi * h^2)
@inline normalization_factor(::Poly6Kernel{3}, h) = 315 / (64pi * h^3)

@doc raw"""
    SpikyKernel{NDIMS}()

The Spiky kernel is another frequently used kernel in SPH, especially due to its desirable
properties in preserving features near boundaries in fluid simulations. It is defined as:

\[ W(r, h) = \frac{1}{h^d} w(r/h) \]

with:

\[
w(q) = \sigma \begin{cases}
    (1 - q)^3    & \text{if } 0 \leq q < 1, \\
    0            & \text{if } q \geq 1,
\end{cases}
\]

where `` d `` is the number of dimensions and `` \sigma `` is a normalization factor, which
depends on the dimension and ensures the kernel integrates to 1 over its support.

This kernel function has a compact support of `` [0, h] ``.

The Spiky kernel is particularly known for its sharp gradients, which can help to preserve
sharp features in fluid simulations, especially near solid boundaries.
These sharp gradients at the boundary are also the largest disadvantage as they can lead to instability.

## References:
- Matthias Müller, David Charypar, and Markus Gross. "Particle-based fluid simulation for interactive applications".
  In: Proceedings of the 2003 ACM SIGGRAPH/Eurographics symposium on Computer animation. Eurographics Association. 2003, pages 154-159.
  [doi: 10.5555/846276.846298](https://doi.org/10.5555/846276.846298)
"""
struct SpikyKernel{NDIMS} <: SmoothingKernel{NDIMS} end

@inline function kernel(kernel::SpikyKernel, r::Real, h)
    q = r / h

    # Zero out result if q >= 1
    result = ifelse(q < 1, normalization_factor(kernel, h) * (1 - q)^3, zero(q))

    return result
end

@inline function kernel_deriv(kernel::SpikyKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    # Zero out result if q >= 1
    result = ifelse(q < 1, -3 * (1 - q)^2 * normalization_factor(kernel, h) * inner_deriv,
                    zero(q))

    return result
end

@inline compact_support(::SpikyKernel, h) = h

@inline normalization_factor(::SpikyKernel{2}, h) = 10 / (pi * h^2)
@inline normalization_factor(::SpikyKernel{3}, h) = 15 / (pi * h^3)
