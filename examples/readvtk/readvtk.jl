# Example File to read a vtk file and convert it to InitialCondition
using TrixiParticles

# the example file is from the simulation “dam_break_2d.jl” at time step 70 (1.4s) with an added 'custom_quantity'
filename = "readvtk_boundary_example"
file = joinpath("examples", "preprocessing", "data", filename * ".vtu")

# Read the vtk file and convert it to an 'InitialCondition'
# 'system_type' must be "fluid" or "boundary"
system_type = "boundary"
ic = vtk2trixi(file, system_type)

trixi2vtk(ic; filename="readvtk", output_directory="out",
          custom_quantity=nothing)
