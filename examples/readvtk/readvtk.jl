# Example File to read a vtk file and convert it to InitialCondition
using TrixiParticles

# The example file is from the simulation “dam_break_2d.jl” at time step 70 (1.4s) with added 'custom_quantities'
filename = "readvtk_fluid_example"
file = joinpath("examples", "preprocessing", "data", filename * ".vtu")

# filename = "boundary_1_1"
# file = joinpath("out", filename * ".vtu")

# Read the vtk file and convert it to an 'InitialCondition'
# 'system_type' must be "fluid" or "boundary"
system_type = "fluid"
ic = vtk2trixi(file, system_type)

trixi2vtk(ic; filename="readvtk", output_directory="out",
          custom_quantity=nothing)
