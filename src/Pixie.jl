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
include("sph/viscosity.jl") # TODO load before sph.jl
include("sph/boundary_conditions.jl") # TODO load before sph.jl
include("sph/neighborhood_search.jl")
include("sph/sph.jl")
include("sph/smoothing_kernels.jl")
include("sph/state_equations.jl")
include("sph/pressure_poisson_equation.jl")
include("visualization/write2vtk.jl")

export WCSPHSemidiscretization, EISPHSemidiscretization, semidiscretize, AliveCallback
export ContinuityDensity, SummationDensity
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel, SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan, ViscosityClearyMonaghan
export BoundaryConditionMonaghanKajtar, BoundaryConditionCrespo
export SpatialHashingSearch
export PPEExplicitLiu

end # module
