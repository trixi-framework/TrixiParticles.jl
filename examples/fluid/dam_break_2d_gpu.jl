# ==========================================================================================
# 2D Dam Break Simulation on GPU
#
# This example file demonstrates how to adapt an existing CPU-based example
# for execution on a GPU. It involves:
# 1. Including the base CPU example to load common parameters and object definitions
#    (without running the simulation by setting `sol=nothing`).
# 2. Defining a GPU-compatible neighborhood search.
# 3. Re-including the base example, overriding the neighborhood search and specifying
#    a GPU-compatible parallelization backend.
#
# Note: To run this example on an actual GPU, the `parallelization_backend`
# needs to be changed to a GPU-specific backend like `CUDABackend()` (for NVIDIA GPUs)
# or `ROCBackend()` (for AMD GPUs), and the corresponding GPU packages
# (e.g., `CUDA.jl`) must be installed.
# See the TrixiParticles.jl documentation on GPU support for more details.
# ==========================================================================================

using TrixiParticles

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=0.6 / 40,
              spacing_ratio=1, boundary_layers=4,
              coordinates_eltype=Float64,
              sol=nothing, ode=nothing)

# Define a GPU-compatible neighborhood search
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

# Run the dam break simulation with this neighborhood search
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              neighborhood_search=neighborhood_search,
              fluid_particle_spacing=fluid_particle_spacing,
              tspan=tspan, smoothing_length=smoothing_length,
              density_diffusion=density_diffusion,
              boundary_layers=boundary_layers, spacing_ratio=spacing_ratio,
              boundary_model=boundary_model,
              parallelization_backend=PolyesterBackend(),
              boundary_density_calculator=boundary_density_calculator,
              coordinates_eltype=Float64)
