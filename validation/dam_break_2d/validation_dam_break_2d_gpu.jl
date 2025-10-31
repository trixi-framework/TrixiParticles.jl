using TrixiParticles
using TrixiParticles.JSON
using CUDA

min_corner = (-1.0, -1.0)
max_corner = (7.0, 5.0)
cell_list = FullGridCellList(; min_corner, max_corner)
neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

trixi_include_changeprecision(Float32, @__MODULE__,
                                joinpath(validation_dir(),
                                        "dam_break_2d", "validation_dam_break_2d.jl");
                                parallelization_backend=CUDABackend(),
                                neighborhood_search=neighborhood_search)
