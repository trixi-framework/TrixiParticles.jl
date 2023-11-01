module TrixiParticles

using Reexport: @reexport

using Dates
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using FastPow: @fastpow
using LinearAlgebra: norm, dot, I, tr
using Morton: cartesian2morton
using MuladdMacro: @muladd
using Polyester: Polyester, @batch
using Printf: @printf, @sprintf
using SciMLBase: CallbackSet, DiscreteCallback, DynamicalODEProblem, u_modified!,
                 get_tmp_cache
@reexport using StaticArrays: SVector
using StaticArrays: @SMatrix, SMatrix, setindex
using StrideArrays: PtrArray, StaticInt
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, paraview_collection, vtk_save
using ForwardDiff

# util needs to be first because of macro @trixi_timeit
include("util.jl")
include("callbacks/callbacks.jl")
include("general/general.jl")
include("neighborhood_search/neighborhood_search.jl")
include("setups/setups.jl")
include("schemes/schemes.jl")

# Note that `semidiscretization.jl` depends on the system types and has to be
# included separately.
include("general/semidiscretization.jl")
include("visualization/write2vtk.jl")

export Semidiscretization, semidiscretize, restart_with!
export InitialCondition
export WeaklyCompressibleSPHSystem, EntropicallyDampedSPHSystem, TotalLagrangianSPHSystem,
       BoundarySPHSystem
export InfoCallback, SolutionSavingCallback, DensityReinitializationCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan, ViscosityAdami
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation,
       PressureMirroring, PressureZeroing
export BoundaryMovement
export GridNeighborhoodSearch
export examples_dir, trixi_include
export trixi2vtk
export RectangularTank, RectangularShape, SphereShape
export VoxelSphere, RoundSphere, reset_wall!
export ShepardKernelCorrection, KernelGradientCorrection, AkinciFreeSurfaceCorrection, GradientCorrection
export nparticles

end # module
