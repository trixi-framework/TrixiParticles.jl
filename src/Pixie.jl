module Pixie

using DiffEqCallbacks: SavedValues, SavingCallback, SavingAffect
using LinearAlgebra: norm, dot, I, tr
using Morton: cartesian2morton
using Polyester: @batch
using Printf: @printf
using SciMLBase: CallbackSet, DiscreteCallback, ODEProblem, u_modified!, get_tmp_cache
using StaticArrays: SVector, @SMatrix, SMatrix
using StrideArrays: PtrArray, StaticInt
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using UnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("util.jl")
include("sph/sph.jl")
include("containers/container.jl")
include("semidiscretization/semidiscretization.jl")
include("interactions/interactions.jl")
include("callbacks/callbacks.jl")
include("visualization/write2vtk.jl")
include("setups/setups.jl")

export Semidiscretization, semidiscretize
export FluidParticleContainer, SolidParticleContainer, BoundaryParticleContainer
export AliveCallback, SolutionSavingCallback, SummaryCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel, SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation
export SpatialHashingSearch
export examples_dir, pixie_include
export pixie2vtk
export RectangularTank, RectangularShape
export draw_circle, fill_circle, reset_right_wall!

end # module
