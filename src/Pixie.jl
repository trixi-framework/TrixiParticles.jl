module Pixie

using LinearAlgebra: norm
using SciMLBase: ODEProblem
using StaticArrays: SVector
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes

include("sph/sph.jl")
include("visualization/write2vtk.jl")

end # module
