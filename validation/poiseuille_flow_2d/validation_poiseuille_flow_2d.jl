using TrixiParticles

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              output_directory=joinpath(validation_dir(), "poiseuille_flow_2d"),
              saving_callback=nothing, tspan=(0.0, 0.1))
