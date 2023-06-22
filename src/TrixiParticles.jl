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
include("general/general.jl")
include("setups/setups.jl")
include("schemes/schemes.jl")
# Note that `semidiscretization.jl` depends on the system types and has to be
# included separately.
include("general/semidiscretization.jl")
include("visualization/write2vtk.jl")

export Semidiscretization, semidiscretize, restart_with!
export InitialCondition
export WeaklyCompressibleSPHSystem, TotalLagrangianSPHSystem, BoundarySPHSystem
export InfoCallback, SolutionSavingCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan, ViscosityAdami
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation
export SpatialHashingSearch
export examples_dir, trixi_include
export trixi2vtk
export RectangularTank, RectangularShape, CircularShape
export DrawCircle, FillCircle, reset_wall!
export ShepardKernelCorrection, AkinciFreeSurfaceCorrection
export nparticles

end # module
