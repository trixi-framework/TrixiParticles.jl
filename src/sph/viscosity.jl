struct NoViscosity end

function (::NoViscosity)(c, v_diff, pos_diff, distance, density_diff, h)
    return 0.0
end


struct ArtificialViscosityMonaghan{ELTYPE}
    alpha   ::ELTYPE
    beta    ::ELTYPE
    epsilon ::ELTYPE

    function ArtificialViscosityMonaghan(alpha, beta, epsilon=0.01)
        new{typeof(alpha)}(alpha, beta, epsilon)
    end
end

function (viscosity::ArtificialViscosityMonaghan)(c, v_diff, pos_diff, distance, density_diff, h)
    @unpack alpha, beta, epsilon = viscosity

    vr = sum(v_diff .* pos_diff)

    if vr < 0
        mu = h * vr / (distance^2 + epsilon * h^2)
        return (-alpha * c * mu + beta * mu^2) / density_diff
    end

    return 0.0
end
