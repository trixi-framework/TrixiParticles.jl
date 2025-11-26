# This file computes the pressure sensor data of the dam break setup described in
#
# J. J. De Courcy, T. C. S. Rendall, L. Constantin, B. Titurus, J. E. Cooper.
# "Incompressible δ-SPH via artificial compressibility".
# In: Computer Methods in Applied Mechanics and Engineering, Volume 420 (2024),
# https://doi.org/10.1016/j.cma.2023.116700

using TrixiParticles
using TrixiParticles.JSON
using CUDA

# ==========================================================================================
# ==== WCSPH simulation
trixi_include_changeprecision(Float32, @__MODULE__,
              joinpath(examples_dir(), "paper",
                       "dam_break_2d_val_600_phys_viscosity.jl"))
