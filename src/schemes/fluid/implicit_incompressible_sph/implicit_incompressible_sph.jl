# Walls whose pressure is extrapolated from the fluid pressure in every iteration of the
# IISPH pressure solve, see `wall_pressure_extrapolation.jl`
const IISPHExtrapolatingWall = WallBoundarySystem{<:BoundaryModelDummyParticles{<:Union{AdamiPressureExtrapolation,
                                                                                        BernoulliPressureExtrapolation}}}

# Variants of the IISPH pressure solve, see `update_implicit_sph!`.
# Without pressure-extrapolating walls, the solve starts from half the previous pressure
# and uses the density error of Ihmsen et al. (2013).
struct StandardIISPHSolve end
# With Adami or Bernoulli walls, the wall pressure is extrapolated from the pressure iterate
# in every iteration. The solve starts from zero pressure and uses the projected residual.
struct WallExtrapolationIISPHSolve end

include("pressure_boundaries.jl")
include("system.jl")
include("wall_pressure_extrapolation.jl")
