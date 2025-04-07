# This example file demonstrates how to run an existing example file on a GPU.
# We simply define a GPU-compatible neighborhood search and `trixi_include` the example
# file with this neighborhood search.
# To run this example on a GPU, `data_type` needs to be changed to the array type of the
# installed GPU. See the docs on GPU support for more information.

using TrixiParticles

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, ode=nothing)

# ------------------------------------------------------------------------------
# GPU-Compatible Neighborhood Search Setup
# ------------------------------------------------------------------------------

# Compute the minimum and maximum corners of the tank's boundary to define the grid.
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)

# Create a full grid cell list using the computed corners.
cell_list = FullGridCellList(; min_corner, max_corner)

# Define the GPU-compatible neighborhood search using the cell list.
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

# ------------------------------------------------------------------------------
# Run Dam Break Simulation on GPU
# ------------------------------------------------------------------------------
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              neighborhood_search=neighborhood_search,
              fluid_particle_spacing=fluid_particle_spacing,
              tspan=tspan,
              density_diffusion=density_diffusion,
              boundary_layers=boundary_layers,
              spacing_ratio=spacing_ratio,
              boundary_model=boundary_model,
              data_type=nothing)
