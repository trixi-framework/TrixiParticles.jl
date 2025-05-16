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
# or `AMDGPUBackend()` (for AMD GPUs), and the corresponding GPU packages
# (e.g., `CUDA.jl`) must be installed.
# See the TrixiParticles.jl documentation on GPU support for more details.
# ==========================================================================================

using TrixiParticles

# ------------------------------------------------------------------------------
# Load Base Setup from CPU Dam Break Example
# ------------------------------------------------------------------------------
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=0.6 / 20,
              spacing_ratio=1,
              boundary_layers=4,
              sol=nothing, ode=nothing)

# ------------------------------------------------------------------------------
# GPU-Compatible Neighborhood Search Setup
# ------------------------------------------------------------------------------

# Define the search domain for the neighborhood search based on the tank geometry.
min_corner = minimum(tank.boundary.coordinates, dims=2)
max_corner = maximum(tank.boundary.coordinates, dims=2)

# Create a full grid cell list covering the domain.
cell_list = FullGridCellList(min_corner=min_corner, max_corner=max_corner)

# Define the GPU-compatible neighborhood search using this cell list.
# Note: `GridNeighborhoodSearch` itself is compatible with both CPU and GPU backends.
neighborhood_search_gpu = GridNeighborhoodSearch{2}(cell_list=cell_list)

# ------------------------------------------------------------------------------
# Run Dam Break Simulation with GPU-Specific Settings
# ------------------------------------------------------------------------------
# Re-include the 2D dam break example, this time overriding parameters
# relevant for GPU execution, including the neighborhood search and parallelization backend.
#
# Replace `PolyesterBackend()` with `CUDABackend()` or `AMDGPUBackend()` for actual GPU execution.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              neighborhood_search=neighborhood_search_gpu,
              fluid_particle_spacing=fluid_particle_spacing,
              tspan=tspan,
              density_diffusion=density_diffusion,
              boundary_layers=boundary_layers,
              spacing_ratio=spacing_ratio,
              boundary_model=boundary_model,
              parallelization_backend=PolyesterBackend())   # Replace with CUDABackend() or ROCBackend()
