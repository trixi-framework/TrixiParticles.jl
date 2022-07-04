module Pixie

using LinearAlgebra: norm
using SciMLBase: DiscreteCallback, ODEProblem
using StaticArrays: SVector
using UnPack: @unpack
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("sph/sph.jl")
include("visualization/write2vtk.jl")

end # module
