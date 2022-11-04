@inline function reset_du!(du)
    du .= zero(eltype(du))

    return du
end


@inline function calc_gravity!(du, particle, semi)
    @unpack gravity = semi

    for i in 1:ndims(semi)
        du[i+ndims(semi), particle] += gravity[i]
    end

    return du
end


include("smoothing_kernels.jl")
include("fluid/fluid.jl")
include("solid/solid.jl")
