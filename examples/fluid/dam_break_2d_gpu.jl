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

# Define a GPU-compatible neighborhood search
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

# Run the dam break simulation with the this neighborhood search
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              neighborhood_search=neighborhood_search,
              fluid_particle_spacing=fluid_particle_spacing,
              tspan=tspan,
              density_diffusion=density_diffusion,
              boundary_layers=boundary_layers, spacing_ratio=spacing_ratio,
              boundary_model=boundary_model,
              data_type=nothing)
