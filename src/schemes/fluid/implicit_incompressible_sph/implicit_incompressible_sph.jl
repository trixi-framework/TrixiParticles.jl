struct LegacyIISPHResidual end
struct ProjectedIISPHResidual end
struct OmitFrozenWallForce end
struct IncludeFrozenWallForce end

struct AllIISPHRows end
struct IISPHComponentRows
    labels::Vector{Int}
    label::Int
end

include("pressure_boundaries.jl")
include("system.jl")
include("wall_pressure_staging.jl")
