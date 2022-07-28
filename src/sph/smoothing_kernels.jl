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

compact_support(::CubicSplineKernel, h) = 2 * h

normalization_factor(::CubicSplineKernel{2}, h) = 10 / (7 * pi * h^2)
normalization_factor(::CubicSplineKernel{3}, h) = 1 / (pi * h^3)


struct QuarticSplineKernel{NDIMS} end

function kernel(kernel::QuarticSplineKernel, r, h)
    q = r / h

    if q >= 5/2
        return 0.0
    end

    result = (5/2 - q)^4

    if q < 3/2
        result -= 5 * (3/2 - q)^4

        if q < 1/2
            result += 10 * (1/2 - q)^4
        end
    end

    return normalization_factor(kernel, h) * result
end

function kernel_deriv(kernel::QuarticSplineKernel, r, h)
    inner_deriv = 1/h
    q = r * inner_deriv

    if q >= 5/2
        return 0.0
    end

    result = -4 * (5/2 - q)^3

    if q < 3/2
        result += 20 * (3/2 - q)^3

        if q < 1/2
            result -= 40 * (1/2 - q)^3
        end
    end

    return normalization_factor(kernel, h) * result * inner_deriv
end

compact_support(::QuarticSplineKernel, h) = 2.5 * h

normalization_factor(::QuarticSplineKernel{2}, h) = 96 / (1199 * pi * h^2)
normalization_factor(::QuarticSplineKernel{3}, h) = 1 / (20 * pi * h^3)
