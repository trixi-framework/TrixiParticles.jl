using TrixiParticles

fluid_density = 1000.0

surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.0005)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              surface_tension=surface_tension,
              correction=AkinciFreeSurfaceCorrection(fluid_density))
