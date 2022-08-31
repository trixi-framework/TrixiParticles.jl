@doc raw"""
    PPEExplicitLiu(eta)

test
```math
a+b
```
where ``a`` ...
"""
struct PPEExplicitLiu{ELTYPE}
    eta     :: ELTYPE
    function PPEExplicitLiu(eta)
        new{typeof(eta)}(eta)
    end
end

function (pressure_poisson_equation::PPEExplicitLiu)(density)


    return density
end

