abstract type IISPHPressureMode end
struct StandardIISPHPressure <: IISPHPressureMode end
struct FrozenWallIISPHPressure <: IISPHPressureMode end

include("pressure_boundaries.jl")
include("system.jl")
