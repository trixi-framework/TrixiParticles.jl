# The same as `plot_vortex_street_reference_2d.jl`, but reading the output from the current
# local simulation from the `out` directory instead of using the provided reference data.
trixi_include(joinpath(dirname(@__FILE__), "plot_vortex_street_reference_2d.jl"),
              directory="out")
