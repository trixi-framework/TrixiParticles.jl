# [Smoothing Kernels](@id smoothing_kernel)
Currently the following smoothing kernels are available:

| Smoothing Kernel                          | Compact Support   | Typ. Smoothing Length | Recommended Application | Stability |
| :---------------------------------------- | :---------------- | :-------------------- | :---------------------- | :-------- |
| [`SchoenbergCubicSplineKernel`](@ref)     | $[0, 2h]$         | $1.1$ to $1.3$        | General + sharp waves   | ++        |
| [`SchoenbergQuarticSplineKernel`](@ref)   | $[0, 2.5h]$       | $1.1$ to $1.5$        | General                 | +++       |
| [`SchoenbergQuinticSplineKernel`](@ref)   | $[0, 3h]$         | $1.1$ to $1.5$        | General                 | ++++      |
| [`GaussianKernel`](@ref)                  | $[0, 3h]$         | $1.0$ to $1.5$        | Literature              | +++++     |
| [`WendlandC2Kernel`](@ref)                | $[0, 1h]$         | $2.5$ to $4.0$        | General (recommended)   | ++++      |
| [`WendlandC4Kernel`](@ref)                | $[0, 1h]$         | $3.0$ to $4.5$        | General                 | +++++     |
| [`WendlandC6Kernel`](@ref)                | $[0, 1h]$         | $3.5$ to $5.0$        | General                 | +++++     |
| [`Poly6Kernel`](@ref)                     | $[0, 1h]$         | $1.5$ to $2.5$        | Literature              | +         |
| [`SpikyKernel`](@ref)                     | $[0, 1h]$         | $1.5$ to $3.0$        | Sharp corners + waves   | +         |

We recommend to use the [`WendlandC2Kernel`](@ref) for most applications.
If less smoothing is needed, try [`SchoenbergCubicSplineKernel`](@ref), for more smoothing try [`WendlandC6Kernel`](@ref).

!!! note "Usage"
    The kernel can be called as
    ```
    TrixiParticles.kernel(::SmoothingKernel{NDIMS}, r, h)
    ```
    The length of the compact support can be obtained as
    ```
    TrixiParticles.compact_support(::SmoothingKernel{NDIMS}, h)
    ```

    where `SmoothingKernel{NDIMS}` is an abstract supertype of all smoothing kernels.
    The type parameter `NDIMS` encodes the number of dimensions.

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

```@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("general", "smoothing_kernels.jl")]
```
