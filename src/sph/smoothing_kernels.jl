struct CubicSplineKernel{NDIMS} end

function kernel(kernel::CubicSplineKernel, r, h)
    q = r / h

    if q >= 2
        return 0.0
    end

    result = 1/4 * (2 - q)^3

    if q < 1
        result -= (1 - q)^3
    end

    return normalization_factor(kernel, h) * result
end

function kernel_deriv(kernel::CubicSplineKernel, r, h)
    inner_deriv = 1/h
    q = r * inner_deriv

    if q >= 2
        return 0.0
    end

    result = -3/4 * (2 - q)^2

    if q < 1
        result += 3 * (1 - q)^2
    end

    return normalization_factor(kernel, h) * result * inner_deriv
end

normalization_factor(::CubicSplineKernel{2}, h) = 10 / (7 * pi * h^2)
normalization_factor(::CubicSplineKernel{3}, h) = 1 / (pi * h^3)
