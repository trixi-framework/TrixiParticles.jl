using OrdinaryDiffEq
using TrixiParticles
using TrixiParticles.JSON
using CUDA

trixi_include_changeprecision(Float32, @__MODULE__,
                              joinpath(examples_dir(), "paper",
                                       "dam_break_2d_val_600.jl"),
                              parallelization_backend=CUDABackend(),
                              case="600_val_gpu")
