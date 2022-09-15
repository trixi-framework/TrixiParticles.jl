module Pixie

using DiffEqCallbacks: SavedValues, SavingCallback
using LinearAlgebra: norm
using Morton: cartesian2morton
using Polyester: @batch
using Printf: @printf
using SciMLBase: CallbackSet, DiscreteCallback, ODEProblem, u_modified!
using StaticArrays: SVector
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using UnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("util.jl")
include("sph/boundary_conditions.jl") # TODO load before sph.jl
include("sph/viscosity.jl") # TODO load before sph.jl
include("sph/neighborhood_search.jl")
include("sph/sph.jl")
include("sph/smoothing_kernels.jl")
include("sph/state_equations.jl")
include("callbacks/alive.jl")
include("callbacks/solution_saving.jl")
include("visualization/write2vtk.jl")

export SPHSemidiscretization, semidiscretize, AliveCallback, SolutionSavingCallback
export ContinuityDensity, SummationDensity
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel, SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan
export BoundaryConditionMonaghanKajtar, BoundaryConditionFrozenMirrored
export SpatialHashingSearch
export examples_dir, pixie_include
export pixie2vtk

end # module
