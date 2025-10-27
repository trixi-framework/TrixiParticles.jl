# This file computes the pressure sensor data of the dam break setup described in
#
# J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
# "Incompressible δ-SPH via artificial compressibility".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
# https://doi.org/10.1016/j.cma.2023.116700

using TrixiParticles
using TrixiParticles.JSON
using CUDA

# When using data center CPUs with large numbers of cores, especially on multi-socket
# systems with multiple NUMA nodes, pinning threads to cores can significantly
# improve performance, even for low resolutions.
# using ThreadPinning
# pinthreads(:numa)

# `resolution` in this case is set relative to `H`, the initial height of the fluid.
# Use 40, 80 or 400 for validation.
# Note: 400 takes about 30 minutes on a large data center CPU (much longer with serial update)
resolution = 600

min_corner = (-1.5, -1.5)
max_corner = (6.5, 5.0)
cell_list = FullGridCellList(; min_corner, max_corner, max_points_per_cell=30)

neighborhood_search = GridNeighborhoodSearch{2}(; cell_list, update_strategy=ParallelUpdate())
# neighborhood_search = GridNeighborhoodSearch{2}(; cell_list)

# switch to physical viscosity model
viscosity_fluid = ViscosityMorris(nu = 8.9E-7)
# viscosity_fluid = ViscosityAdami(nu = 8.9E-7)
#  viscosity_fluid = nothing


# ==========================================================================================
# ==== WCSPH simulation
trixi_include(@__MODULE__,
              joinpath(validation_dir(), "dam_break_2d",
                       "setup_marrone_2011.jl"),
              use_edac=false,
              extra_string = "_phys_viscosity",
              viscosity_fluid=viscosity_fluid,
              particles_per_height=resolution,
              sound_speed=50 * sqrt(9.81 * 0.6), # This is used by De Courcy et al. (2024)
              tspan=(0.0, 7 / sqrt(9.81 / 0.6)), # This is used by De Courcy et al. (2024)
            #   tspan=(0.0, 0.0001), # This is used by De Courcy et al. (2024)
              parallelization_backend=CUDABackend(),
              neighborhood_search=neighborhood_search)
