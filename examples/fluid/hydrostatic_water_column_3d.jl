using TrixiParticles

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              tspan=(0.0, 0.1))
