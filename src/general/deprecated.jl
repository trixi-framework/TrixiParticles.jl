# Deprecated kernel functions, kept for backwards compatibility.
#
# Previously, the kernel functions were only safe inside the compact support and for
# nonzero distances, and their `_unsafe` versions skipped these checks.
# Neighbor loops had to skip pairs at (almost) zero distance with `skip_zero_distance`
# before calling the `_unsafe` versions.
# Now, `kernel`, `kernel_deriv`, `kernel_grad`, `smoothing_kernel` and
# `smoothing_kernel_grad` are safe for all distances without branching, so these safe
# functions can be used in performance-critical, SIMD-vectorized neighbor loops.
# The functions in this file are therefore no longer needed (see NEWS.md for how to migrate).
# They are not used anywhere in TrixiParticles.jl.
#
# TODO remove this file in the next major release.
#
# The smoothing kernels of TrixiParticles.jl, which define the safe functions `kernel` and
# `kernel_deriv_div_r` directly.
const BuiltinSmoothingKernel = Union{GaussianKernel, SchoenbergCubicSplineKernel,
                                     SchoenbergQuarticSplineKernel,
                                     SchoenbergQuinticSplineKernel, AbstractWendlandKernel,
                                     Poly6Kernel, SpikyKernel, LaguerreGaussKernel}

# Fallbacks for custom kernels that only define the deprecated functions `kernel_unsafe`
# and `kernel_deriv_div_r_unsafe`, as described in the custom kernel tutorial of previous
# versions. These functions are only valid inside the compact support, and
# `kernel_deriv_div_r_unsafe` may divide by `r`, so we keep the checks of the previous
# implementation. The kernels of TrixiParticles.jl define more specific methods.
@inline function kernel(kernel, r::Real, h)
    # Zero out result if outside of compact support
    compact_support_ = compact_support(kernel, h)
    return ifelse(r < compact_support_, kernel_unsafe(kernel, r, h), zero(r))
end

@inline function kernel_deriv_div_r(kernel, r::Real, h)
    # Zero out result if outside of compact support or if `r` is almost zero
    # to avoid division by zero in the unsafe version (see `almostzero`).
    if r < compact_support(kernel, h) && r >= almostzero(h)
        return kernel_deriv_div_r_unsafe(kernel, r, h)
    end

    return zero(r)
end

# The `_unsafe` versions below only exist for the kernels of TrixiParticles.jl.
# Custom kernels that define them are handled by the fallbacks above.
@inline function kernel_unsafe(kernel_::BuiltinSmoothingKernel, r::Real, h)
    return kernel(kernel_, r, h)
end

@inline function kernel_deriv_div_r_unsafe(kernel_::BuiltinSmoothingKernel, r::Real, h)
    return kernel_deriv_div_r(kernel_, r, h)
end

@inline function kernel_grad_unsafe(kernel, pos_diff, distance, h)
    return kernel_grad(kernel, pos_diff, distance, h)
end

@inline function corrected_kernel_grad_unsafe(kernel, pos_diff, distance, h, correction,
                                              system, particle)
    return corrected_kernel_grad(kernel, pos_diff, distance, h, correction,
                                 system, particle)
end

@inline function smoothing_kernel_unsafe(system, distance, particle)
    return smoothing_kernel(system, distance, particle)
end

@inline function smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
    return smoothing_kernel_grad(system, pos_diff, distance, particle)
end

# Deprecated, the kernel gradient is now exactly zero for zero distance, and pairs of
# particles at zero distance no longer need to be skipped.
# This returns `true` if the kernel gradient is zero at zero distance, which is not
# the case for some corrected kernels.
@inline function skip_zero_distance(correction)
    return true
end

@inline function skip_zero_distance(::Union{KernelCorrection,
                                            MixedKernelGradientCorrection})
    # For these corrections, the kernel gradient between a particle and itself is not zero.
    return false
end

@inline function skip_zero_distance(system::AbstractSystem)
    return skip_zero_distance(system_correction(system))
end
