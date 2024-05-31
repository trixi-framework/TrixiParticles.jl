module TrixiParticles

using Reexport: @reexport

using Adapt: Adapt
using CSV: CSV
using Dates
using DataFrames: DataFrame
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect, PresetTimeCallback
using FastPow: @fastpow
using ForwardDiff: ForwardDiff
using JSON: JSON
using LinearAlgebra: norm, dot, I, tr, inv, pinv, det
using MuladdMacro: @muladd
using Polyester: Polyester, @batch
using Printf: @printf, @sprintf
using RecipesBase: RecipesBase, @series
using SciMLBase: CallbackSet, DiscreteCallback, DynamicalODEProblem, u_modified!,
                 get_tmp_cache, set_proposed_dt!, ODESolution, ODEProblem
@reexport using StaticArrays: SVector
using StaticArrays: @SMatrix, SMatrix, setindex
using StrideArrays: PtrArray, StaticInt
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using TrixiBase: trixi_include
@reexport using PointNeighbors: TrivialNeighborhoodSearch, GridNeighborhoodSearch
using PointNeighbors: PointNeighbors, for_particle_neighbor
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, paraview_collection, vtk_save

# util needs to be first because of macro @trixi_timeit
include("util.jl")
include("callbacks/callbacks.jl")
include("general/general.jl")
include("setups/setups.jl")
include("schemes/schemes.jl")

# Note that `semidiscretization.jl` depends on the system types and has to be
# included separately. `gpu.jl` in turn depends on the semidiscretization type.
include("general/semidiscretization.jl")
include("general/gpu.jl")
include("visualization/write2vtk.jl")
include("visualization/recipes_plots.jl")

export Semidiscretization, semidiscretize, restart_with!
export InitialCondition
export WeaklyCompressibleSPHSystem, EntropicallyDampedSPHSystem, TotalLagrangianSPHSystem,
       RigidSPHSystem, BoundarySPHSystem, DEMSystem, BoundaryDEMSystem
export InfoCallback, SolutionSavingCallback, DensityReinitializationCallback,
       PostprocessCallback, StepsizeCallback, UpdateCallback
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
export examples_dir, validation_dir, trixi_include
export trixi2vtk
export RectangularTank, RectangularShape, SphereShape
export VoxelSphere, RoundSphere, reset_wall!, extrude_geometry
export SourceTermDamping
export ShepardKernelCorrection, KernelCorrection, AkinciFreeSurfaceCorrection,
       GradientCorrection, BlendedGradientCorrection, MixedKernelGradientCorrection
export nparticles
export kinetic_energy, total_mass, max_pressure, min_pressure, avg_pressure,
       max_density, min_density, avg_density
export interpolate_line, interpolate_point, interpolate_plane_3d, interpolate_plane_2d,
       interpolate_plane_2d_vtk
export SurfaceTensionAkinci, CohesionForceAkinci

end # module
