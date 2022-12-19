module Pixie

using DiffEqCallbacks: SavedValues, SavingCallback
using LinearAlgebra: norm, dot, I, tr
using Morton: cartesian2morton
using Polyester: @batch
using Printf: @printf
using SciMLBase: CallbackSet, DiscreteCallback, ODEProblem, u_modified!, get_tmp_cache
using StaticArrays: SVector, @SMatrix, SMatrix
using ThreadingUtilities
using TimerOutputs: TimerOutput, TimerOutputs, print_timer, reset_timer!
using UnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("util.jl")
include("sph/neighborhood_search.jl") # semidiscretization/semidiscretization.jl depends on this
include("sph/fluid/density_calculators.jl") # containers/containers.jl depends on this
include("sph/solid/penalty_force.jl") # containers/containers.jl depends on this
include("containers/container.jl")
include("semidiscretization/semidiscretization.jl")
include("sph/sph.jl")
include("callbacks/alive.jl")
include("callbacks/solution_saving.jl")
include("visualization/write2vtk.jl")
include("setups/rectangular_tank.jl")
include("setups/moving_wall.jl")

export Semidiscretization, semidiscretize
export FluidParticleContainer, SolidParticleContainer, BoundaryParticleContainer
export AliveCallback, SolutionSavingCallback
export ContinuityDensity, SummationDensity
export PenaltyForceGanzenmueller
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel, SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan
export BoundaryModelMonaghanKajtar, BoundaryModelFrozen
export SpatialHashingSearch
export examples_dir, pixie_include
export pixie2vtk
export RectangularTank, reset_right_wall!, RectangularWall

end # module
