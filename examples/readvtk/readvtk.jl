# Example File to read a vtk file and convert it to InitialCondition
using TrixiParticles

filename = "read_vtk_example"
file = joinpath("examples", "preprocessing", "data", filename * ".vtu")

ic = vtk2trixi(file)

trixi2vtk(ic; filename="readvtk", output_directory="out",
          custom_quantity=nothing)
