module TrixiParticles

using Reexport: @reexport

using Dates
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using FastPow: @fastpow
using ForwardDiff: ForwardDiff
using LinearAlgebra: norm, dot, I, tr, inv, pinv, det
using Morton: cartesian2morton
using MuladdMacro: @muladd
using Polyester: Polyester, @batch
using Printf: @printf, @sprintf
using RecipesBase: RecipesBase, @series
using SciMLBase: CallbackSet, DiscreteCallback, DynamicalODEProblem, u_modified!,
                 get_tmp_cache, set_proposed_dt!, ODESolution, ODEProblem
@reexport using StaticArrays: SVector
using StaticArrays: @SMatrix, SMatrix, setindex
using StrideArrays: PtrArray, StaticInt
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using TrixiBase: trixi_include
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, paraview_collection, vtk_save

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
include("visualization/recipes_plots.jl")

export Semidiscretization, semidiscretize, initialize_ode, restart_with!
export InitialCondition
export WeaklyCompressibleSPHSystem, EntropicallyDampedSPHSystem, TotalLagrangianSPHSystem,
       BoundarySPHSystem
export InfoCallback, SolutionSavingCallback, DensityReinitializationCallback,
       StepsizeCallback, default_callback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel, GaussianKernel, WendlandC2Kernel, WendlandC4Kernel,
       WendlandC6Kernel, SpikyKernel, Poly6Kernel
export StateEquationCole
export ArtificialViscosityMonaghan, ViscosityAdami
export DensityDiffusion, DensityDiffusionMolteniColagrossi, DensityDiffusionFerrari,
       DensityDiffusionAntuono
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation,
       PressureMirroring, PressureZeroing
export BoundaryMovement
export GridNeighborhoodSearch, TrivialNeighborhoodSearch
export examples_dir, trixi_include
export trixi2vtk
export RectangularTank, RectangularShape, SphereShape
export VoxelSphere, RoundSphere, reset_wall!
export SourceTermDamping
export ShepardKernelCorrection, KernelCorrection, AkinciFreeSurfaceCorrection,
       GradientCorrection, BlendedGradientCorrection, MixedKernelGradientCorrection
export nparticles
export interpolate_line, interpolate_point, interpolate_plane_3d, interpolate_plane_2d

end # module
