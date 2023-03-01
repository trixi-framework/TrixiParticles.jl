module Pixie

using Reexport: @reexport

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
@reexport using UnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

# util needs to be first because of macro @pixie_timeit
include("util.jl")
include("sph/sph.jl")
include("setups/setups.jl")
include("containers/container.jl")
include("semidiscretization/semidiscretization.jl")
include("interactions/interactions.jl")
include("callbacks/callbacks.jl")
include("visualization/write2vtk.jl")

export Semidiscretization, semidiscretize, restart_with!
export FluidParticleContainer, SolidParticleContainer, BoundaryParticleContainer
export AliveCallback, SolutionSavingCallback, SummaryCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation
export SpatialHashingSearch
export examples_dir, pixie_include
export pixie2vtk
export RectangularTank, RectangularShape, CircularShape
export DrawCircle, FillCircle, reset_wall!

end # module
