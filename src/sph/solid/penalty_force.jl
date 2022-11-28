@doc raw"""
    PenaltyForceGanzenmueller()

TODO
"""
struct PenaltyForceGanzenmueller{ELTYPE}
    alpha   :: ELTYPE
    function PenaltyForceGanzenmueller(; alpha=1.0)
        new{typeof(alpha)}(alpha)
    end
end
