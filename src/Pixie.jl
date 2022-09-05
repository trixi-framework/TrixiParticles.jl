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
include("sph/pressure_poisson_equation.jl")
include("sph/semidiscretization.jl") 
include("sph/viscosity.jl") # TODO load before sph.jl
include("sph/boundary_conditions.jl") # TODO load before sph.jl
include("sph/neighborhood_search.jl")
include("sph/sph.jl")
include("sph/smoothing_kernels.jl")
include("sph/state_equations.jl")
include("callbacks/alive.jl")
include("callbacks/solution_saving.jl")
include("callbacks/step_size.jl")
include("visualization/write2vtk.jl")

export WCSPHSemidiscretization, EISPHSemidiscretization, semidiscretize, AliveCallback, SolutionSavingCallback, StepSizeCallback
export ContinuityDensity, SummationDensity
export SchoenbergCubicSplineKernel, SchoenbergQuarticSplineKernel, SchoenbergQuinticSplineKernel
export StateEquationIdealGas, StateEquationCole
export ArtificialViscosityMonaghan, ViscosityClearyMonaghan
export BoundaryConditionMonaghanKajtar, BoundaryConditionCrespo, BoundaryConditionFixedParticleLiu
export SpatialHashingSearch
export PPEExplicitLiu

end # module
