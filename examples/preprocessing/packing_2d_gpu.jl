using TrixiParticles
using Metal

trixi_include_changeprecision(Float32, @__MODULE__,
                              joinpath(examples_dir(), "preprocessing", "packing_2d.jl");
                              steady_state=nothing, pp_cb_ekin=nothing,
                              parallelization_backend=MetalBackend())
