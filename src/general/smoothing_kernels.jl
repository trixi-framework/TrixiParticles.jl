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
where ``d`` is the number of dimensions and ``\sigma = 17/(7\pi)``
in two dimensions or ``\sigma = 1/\pi`` in three dimensions is a normalization factor.

This kernel function has a compact support of ``[0, 2h]``.

For an overview of Schoenberg cubic, quartic and quintic spline kernels
including normalization factors, see (Price, 2012).
For an analytic formula for higher order kernels, see (Monaghan, 1985).

!!! note "Usage"
    The kernel can be called as `TrixiParticles.kernel(::SchoenbergCubicSplineKernel, r, h)`.
    The length of the compact support can be obtained as
    `TrixiParticles.compact_support(::SchoenbergCubicSplineKernel, h)`.

    Note that ``r`` has to be a scalar, so in the context of SPH, the kernel
    should be used as ``W(\Vert r_a - r_b \Vert, h)``.
    The gradient required in SPH,
    ```math
    \frac{\partial}{\partial r_a} W(\Vert r_a - r_b \Vert, h)
    ```
    can be called as
    `TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)`,
    where `pos_diff` is $r_a - r_b$ and `distance` is $\Vert r_a - r_b \Vert$.

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

@muladd @inline function kernel(kernel::SchoenbergCubicSplineKernel, r::Real, h)
    q = r / h

    result = 1 / 4 * (2 - q)^3 - (q < 1) * (1 - q)^3

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result, zero(result))

    return result
end

@muladd @inline function kernel_deriv(kernel::SchoenbergCubicSplineKernel, r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv

    result = -3 / 4 * (2 - q)^2 + 3 * (q < 1) * (1 - q)^2

    # Zero out result if q >= 2
    result = ifelse(q < 2, normalization_factor(kernel, h) * result * inner_deriv,
                    zero(result))

    return result
end

@inline compact_support(::SchoenbergCubicSplineKernel, h) = 2 * h

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
where ``d`` is the number of dimensions and ``\sigma = 96/(1199\pi)``
in two dimensions or ``\sigma = 1/(20\pi)`` in three dimensions is a normalization factor.

This kernel function has a compact support of ``[0, 2.5h]``.

For an overview of Schoenberg cubic, quartic and quintic spline kernels
including normalization factors, see (Price, 2012).
For an analytic formula for higher order kernels, see (Monaghan, 1985).

!!! note "Usage"
    The kernel can be called as `TrixiParticles.kernel(::SchoenbergQuarticSplineKernel, r, h)`.
    The length of the compact support can be obtained as
    `TrixiParticles.compact_support(::SchoenbergQuarticSplineKernel, h)`.

    Note that ``r`` has to be a scalar, so in the context of SPH, the kernel
    should be used as ``W(\Vert r_a - r_b \Vert, h)``.
    The gradient required in SPH,
    ```math
    \frac{\partial}{\partial r_a} W(\Vert r_a - r_b \Vert, h)
    ```
    can be called as
    `TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)`,
    where `pos_diff` is $r_a - r_b$ and `distance` is $\Vert r_a - r_b \Vert$.

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

# Note that currently specializations reducing this to simple multiplications exist only up
# to a power of three, see
# https://github.com/JuliaLang/julia/blob/34934736fa4dcb30697ac1b23d11d5ad394d6a4d/base/intfuncs.jl#L327-L339
# Here, we accept to lose some precision but gain performance by using plain
# multiplications instead via `@fastpow`.
@fastpow @muladd @inline function kernel(kernel::SchoenbergQuarticSplineKernel, r::Real, h)
    q = r / h
    q5_4 = (5 / 2 - q)^4
    q3_4 = (3 / 2 - q)^4
    q1_4 = (1 / 2 - q)^4

    result = q5_4
    result = result - 5 * (q < 3 / 2) * q3_4
    result = result + 10 * (q < 1 / 2) * q1_4

    # Zero out result if q >= 5/2
    result = ifelse(q < 5 / 2, normalization_factor(kernel, h) * result, zero(result))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::SchoenbergQuarticSplineKernel,
                                               r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv
    q5_2 = 5 / 2 - q
    q3_2 = 3 / 2 - q
    q1_2 = 1 / 2 - q

    result = -4 * q5_2^3
    result = result + 20 * (q < 3 / 2) * q3_2^3
    result = result - 40 * (q < 1 / 2) * q1_2^3

    # Zero out result if q >= 5/2
    result = ifelse(q < 5 / 2, normalization_factor(kernel, h) * result * inner_deriv,
                    zero(result))

    return result
end

@inline compact_support(::SchoenbergQuarticSplineKernel, h) = 2.5 * h

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
where ``d`` is the number of dimensions and ``\sigma = 7/(478\pi)``
in two dimensions or ``\sigma = 1/(120\pi)`` in three dimensions is a normalization factor.

This kernel function has a compact support of ``[0, 3h]``.

For an overview of Schoenberg cubic, quartic and quintic spline kernels
including normalization factors, see (Price, 2012).
For an analytic formula for higher order kernels, see (Monaghan, 1985).

!!! note "Usage"
    The kernel can be called as `TrixiParticles.kernel(::SchoenbergQuinticSplineKernel, r, h)`.
    The length of the compact support can be obtained as
    `TrixiParticles.compact_support(::SchoenbergQuinticSplineKernel, h)`.

    Note that ``r`` has to be a scalar, so in the context of SPH, the kernel
    should be used as ``W(\Vert r_a - r_b \Vert, h)``.
    The gradient required in SPH,
    ```math
    \frac{\partial}{\partial r_a} W(\Vert r_a - r_b \Vert, h)
    ```
    can be called as
    `TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)`,
    where `pos_diff` is $r_a - r_b$ and `distance` is $\Vert r_a - r_b \Vert$.

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

@fastpow @muladd @inline function kernel(kernel::SchoenbergQuinticSplineKernel, r::Real, h)
    q = r / h
    q3_5 = (3 - q)^5
    q2_5 = (2 - q)^5
    q1_5 = (1 - q)^5

    result = q3_5

    # (q < 2) evaluates to 1 if q is less than 2 and 0 otherwise.
    result = result - 6 * (q < 2) * q2_5
    result = result + 15 * (q < 1) * q1_5

    # Zero out result if q >= 3
    result = ifelse(q < 3, normalization_factor(kernel, h) * result, zero(result))

    return result
end

@fastpow @muladd @inline function kernel_deriv(kernel::SchoenbergQuinticSplineKernel,
                                               r::Real, h)
    inner_deriv = 1 / h
    q = r * inner_deriv
    q3_4 = (3 - q)^4
    q2_4 = (2 - q)^4
    q1_4 = (1 - q)^4

    result = -5 * q3_4

    # (q < 2) evaluates to 1 if q is less than 2 and 0 otherwise.
    result = result + 30 * (q < 2) * q2_4
    result = result - 75 * (q < 1) * q1_4
    # Zero out result if q >= 3
    result = ifelse(q < 3, normalization_factor(kernel, h) * result * inner_deriv,
                    zero(result))

    return result
end

@inline compact_support(::SchoenbergQuinticSplineKernel, h) = 3 * h

@inline normalization_factor(::SchoenbergQuinticSplineKernel{2}, h) = 7 / (478 * pi * h^2)
@inline normalization_factor(::SchoenbergQuinticSplineKernel{3}, h) = 1 / (120 * pi * h^3)
