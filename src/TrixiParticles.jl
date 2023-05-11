module TrixiParticles

using Reexport: @reexport

using Dates
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using LinearAlgebra: norm, dot, I, tr
using Morton: cartesian2morton
using Polyester: @batch
using Printf: @printf
using SciMLBase: CallbackSet, DiscreteCallback, DynamicalODEProblem, u_modified!,
                 get_tmp_cache
@reexport using StaticArrays: SVector
using StaticArrays: @SMatrix, SMatrix
using StrideArrays: PtrArray, StaticInt
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
@reexport using SimpleUnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

# util needs to be first because of macro @trixi_timeit
include("util.jl")
include("callbacks/callbacks.jl")
include("setups/setups.jl")
include("containers/container.jl")
include("general/general.jl")
include("schemes/schemes.jl")
include("visualization/write2vtk.jl")

export Semidiscretization, semidiscretize, restart_with!
export FluidParticleContainer, SolidParticleContainer, BoundaryParticleContainer
export InfoCallback, SolutionSavingCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation
export SpatialHashingSearch
export examples_dir, trixi_include
export trixi2vtk
export MergeShapes, RectangularTank, RectangularShape, CircularShape
export DrawCircle, FillCircle, reset_wall!

end # module
