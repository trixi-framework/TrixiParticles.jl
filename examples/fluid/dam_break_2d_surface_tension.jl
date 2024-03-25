using TrixiParticles

fluid_density = 1000.0

H = 0.6
fluid_particle_spacing = H / 40

surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.05)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              surface_tension=surface_tension,
              smoothing_length=2.0 * fluid_particle_spacing,
              correction=AkinciFreeSurfaceCorrection(fluid_density), sol=nothing,
              density_diffusion=nothing, alpha=0.1)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            dt=1e-5,
            save_everystep=false, callback=callbacks);
