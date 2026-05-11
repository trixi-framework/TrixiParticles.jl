# TODO this might be integrated into CUDA.jl at some point, see
# https://github.com/JuliaGPU/CUDA.jl/pull/3077
module TrixiParticlesCUDAExt

using CUDA: CUDA
using TrixiParticles: TrixiParticles

# Use faster version of `div_fast` for `Float64` on CUDA.
# By default, `div_fast` translates to `Base.FastMath.div_fast`, but there is
# no fast division for `Float64` on CUDA, so we need to redefine it here to use the
# improved fast reciprocal defined below.
CUDA.@device_override TrixiParticles.div_fast(x, y::Float64) = x * fast_inv_cuda(y)

# Improved fast reciprocal for `Float64` by @Mikolaj-A-Kowalski, which is significantly
# more accurate than just calling "llvm.nvvm.rcp.approx.ftz.d" without the cubic iteration,
# while still being much faster than a full division.
# This is copied from Oceananigans.jl, see https://github.com/CliMA/Oceananigans.jl/pull/5140.
@inline function fast_inv_cuda(a::Float64)
    # Get the approximate reciprocal
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp-approx-ftz-f64
    # This instruction chops off last 32bits of mantissa and computes inverse
    # while treating all subnormal numbers as 0.0
    # If reciprocal would be subnormal, underflows to 0.0
    # 32 least significant bits of the result are filled with 0s
    inv_a = ccall("llvm.nvvm.rcp.approx.ftz.d", llvmcall, Float64, (Float64,), a)

    # Approximate the missing 32bits of mantissa with a single cubic iteration
    e = fma(inv_a, -a, 1.0)
    e = fma(e, e, e)
    inv_a = fma(e, inv_a, inv_a)
    return inv_a
end

end # module
