# Example File to read a vtk file and convert it to InitialCondition
using TrixiParticles
using OrdinaryDiffEq

ic = vtk2trixi("out/fluid_1_1.vtu")

trixi2vtk(ic; filename="trixi2vtk_test", output_directory="out_vtk",
          custom_quantity=nothing)
