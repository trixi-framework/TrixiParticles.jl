# Example File to read a vtk file and convert it to InitialCondition
using TrixiParticles

rectangle = RectangularShape(0.1, (10, 10), (0, 0), density=1.0)
trixi2vtk(rectangle; filename="rectangle", output_directory="out",
          custom_quantity=nothing)

filename = "rectangle"
file = joinpath("out", filename * ".vtu")

# Read the vtk file and convert it to an 'InitialCondition'
ic = vtk2trixi(file)

trixi2vtk(ic; filename="readvtk", output_directory="out",
          custom_quantity=nothing)
