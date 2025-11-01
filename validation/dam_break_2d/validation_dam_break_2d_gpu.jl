using TrixiParticles
using TrixiParticles.JSON
using CUDA

min_corner = (-1.5, -1.5)
max_corner = (6.5, 5.0)
cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=30)

neighborhood_search = GridNeighborhoodSearch{2}(; cell_list, update_strategy=ParallelUpdate())
# neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

trixi_include_changeprecision(Float32, @__MODULE__,
                                joinpath(validation_dir(),
                                        "dam_break_2d", "validation_dam_break_2d.jl");
                                parallelization_backend=CUDABackend(),
                                neighborhood_search=neighborhood_search)

# trixi_include_changeprecision(Float32, @__MODULE__,
#                                 joinpath(examples_dir(),
#                                         "fluid", "dam_break_ds.jl");)
