# Example File to read a vtk file and convert it to InitialCondition
using TrixiParticles

coordinates = [0.0 0.0 0.0; 1.0 0.0 0.0]
initial_condition = InitialCondition(; coordinates, density=1.0, particle_spacing=0.1)
trixi2vtk(initial_condition; filename="initial_condition", output_directory="out",
          custom_quantity=nothing)

filename = "initial_condition"
file = joinpath("out", filename * ".vtu")

#rectangle = RectangularShape(0.1, (10, 10), (0, 0), density=1.0)
#trixi2vtk(rectangle; filename="rectangle", output_directory="out",
#          custom_quantity=nothing)

#filename = "rectangle"
#file = joinpath("out", filename * ".vtu")

# The example file is from the simulation “dam_break_2d.jl” at time step 70 (1.4s) with added 'custom_quantities'
#filename = "fluid_1_6"
#file = joinpath("out", filename * ".vtu")

# Read the vtk file and convert it to an 'InitialCondition'
ic = vtk2trixi(file)

trixi2vtk(ic; filename="readvtk", output_directory="out",
          custom_quantity=nothing)
