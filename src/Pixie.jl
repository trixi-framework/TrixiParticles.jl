module Pixie

using LinearAlgebra: norm
using Morton: cartesian2morton
using Polyester: @batch
using Printf: @printf
using SciMLBase: DiscreteCallback, ODEProblem, u_modified!
using StaticArrays: SVector
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using UnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("util.jl")
include("callbacks/alive.jl")
include("sph/boundary_conditions.jl") # TODO load before sph.jl
include("sph/viscosity.jl") # TODO load before sph.jl
include("sph/neighborhood_search.jl")
include("sph/sph.jl")
include("sph/smoothing_kernels.jl")
include("sph/state_equations.jl")
include("visualization/write2vtk.jl")

export SPHSemidiscretization, semidiscretize, AliveCallback
export ContinuityDensity, SummationDensity
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel, SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan
export BoundaryConditionMonaghanKajtar
export SpatialHashingSearch

end # module
