module TrixiParticles

using Reexport: @reexport

using Adapt: Adapt
using Base: @propagate_inbounds
using CSV: CSV
using Dates
using DataFrames: DataFrames, DataFrame
using DelimitedFiles: DelimitedFiles
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect, PresetTimeCallback
using FastPow: @fastpow
using FileIO: FileIO
using ForwardDiff: ForwardDiff
using GPUArraysCore: AbstractGPUArray
using JSON: JSON
using KernelAbstractions: KernelAbstractions, @kernel, @index
using LinearAlgebra: norm, normalize, cross, dot, I, tr, inv, pinv, det
using MuladdMacro: @muladd
using Polyester: Polyester, @batch
using Printf: @printf, @sprintf
using ReadVTK: ReadVTK
using RecipesBase: RecipesBase, @series
using Random: seed!
using SciMLBase: SciMLBase, CallbackSet, DiscreteCallback, DynamicalODEProblem, u_modified!,
                 get_tmp_cache, set_proposed_dt!, ODESolution, ODEProblem, terminate!,
                 add_tstop!, get_du
@reexport using StaticArrays: SVector
using StaticArrays: @SMatrix, SMatrix, setindex
using Statistics: Statistics
using StrideArraysCore: PtrArray, StaticInt
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!, @notimeit
using TrixiBase: @trixi_timeit, timer, timeit_debug_enabled,
                 disable_debug_timings, enable_debug_timings, TrixiBase
@reexport using TrixiBase: trixi_include, trixi_include_changeprecision
@reexport using PointNeighbors: TrivialNeighborhoodSearch, GridNeighborhoodSearch,
                                PrecomputedNeighborhoodSearch, PeriodicBox,
                                ParallelUpdate, SemiParallelUpdate, SerialUpdate,
                                SerialIncrementalUpdate,
                                FullGridCellList, DictionaryCellList,
                                SerialBackend, PolyesterBackend, ThreadsStaticBackend,
                                ThreadsDynamicBackend, default_backend
using PointNeighbors: PointNeighbors, foreach_point_neighbor, copy_neighborhood_search,
                      @threaded
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, paraview_collection, vtk_save

# `util.jl` needs to be first because of the macros `@trixi_timeit` and `@threaded`
include("util.jl")
include("general/abstract_system.jl")
include("general/general.jl")
include("setups/setups.jl")
include("schemes/schemes.jl")
# `callbacks.jl` requires the system types to be defined
include("callbacks/callbacks.jl")

# Note that `semidiscretization.jl` depends on the system types and has to be
# included separately. `gpu.jl` in turn depends on the semidiscretization type.
include("general/semidiscretization.jl")
include("general/gpu.jl")
include("preprocessing/preprocessing.jl")
include("io/io.jl")
include("visualization/recipes_plots.jl")

export Semidiscretization, semidiscretize, restart_with!
export InitialCondition
export WeaklyCompressibleSPHSystem, EntropicallyDampedSPHSystem, TotalLagrangianSPHSystem,
       WallBoundarySystem, DEMSystem, BoundaryDEMSystem, OpenBoundarySystem,
       ImplicitIncompressibleSPHSystem
export BoundaryZone, InFlow, OutFlow, BidirectionalFlow
export InfoCallback, SolutionSavingCallback, DensityReinitializationCallback,
       PostprocessCallback, StepsizeCallback, UpdateCallback, SteadyStateReachedCallback,
       SplitIntegrationCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller, TransportVelocityAdami, ParticleShiftingTechnique,
       ParticleShiftingTechniqueSun2017, ConsistentShiftingSun2019,
       ContinuityEquationTermSun2019, MomentumEquationTermSun2019
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel,
       SchoenbergQuinticSplineKernel, GaussianKernel, WendlandC2Kernel, WendlandC4Kernel,
       WendlandC6Kernel, SpikyKernel, Poly6Kernel, LaguerreGaussKernel
export StateEquationCole, StateEquationIdealGas
export ArtificialViscosityMonaghan, ViscosityAdami, ViscosityMorris, ViscosityAdamiSGS,
       ViscosityMorrisSGS
export DensityDiffusionMolteniColagrossi, DensityDiffusionFerrari, DensityDiffusionAntuono
export tensile_instability_control
export BoundaryModelMonaghanKajtar, BoundaryModelDummyParticles, AdamiPressureExtrapolation,
       PressureMirroring, PressureZeroing, BoundaryModelCharacteristicsLastiwka,
       BoundaryModelMirroringTafuni, BoundaryModelDynamicalPressureZhang,
       BernoulliPressureExtrapolation, PressureBoundaries
export FirstOrderMirroring, ZerothOrderMirroring, SimpleMirroring
export HertzContactModel, LinearContactModel
export PrescribedMotion, OscillatingMotion2D
export RCRWindkesselModel
export examples_dir, validation_dir
export trixi2vtk, vtk2trixi
export RectangularTank, RectangularShape, SphereShape, ComplexShape
export ParticlePackingSystem, SignedDistanceField
export WindingNumberHormann, WindingNumberJacobson
export VoxelSphere, RoundSphere, reset_wall!, extrude_geometry, load_geometry,
       sample_boundary, planar_geometry_to_face
export SourceTermDamping
export ShepardKernelCorrection, KernelCorrection, AkinciFreeSurfaceCorrection,
       GradientCorrection, BlendedGradientCorrection, MixedKernelGradientCorrection
export nparticles
export available_data, kinetic_energy, total_mass, max_pressure, min_pressure, avg_pressure,
       max_density, min_density, avg_density
export interpolate_line, interpolate_points, interpolate_plane_3d, interpolate_plane_2d,
       interpolate_plane_2d_vtk
export SurfaceTensionAkinci, CohesionForceAkinci, SurfaceTensionMorris,
       SurfaceTensionMomentumMorris
export ColorfieldSurfaceNormal
export SymplecticPositionVerlet

end # module
