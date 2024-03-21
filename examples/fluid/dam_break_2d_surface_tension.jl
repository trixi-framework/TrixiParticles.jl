using TrixiParticles

fluid_density = 1000.0

surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.0005)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              surface_tension=surface_tension,
              correction=AkinciFreeSurfaceCorrection(fluid_density), sol=nothing)

sol = solve(ode, RDPK3SpFSAL35(),
              abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
              reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
              dtmax=1e-2, # Limit stepsize to prevent crashing
              dt=1e-5,
              save_everystep=false, callback=callbacks);
