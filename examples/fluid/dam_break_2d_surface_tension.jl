# This example shows how surface tension can be applied to existing cases
# and which parameters have to be changed!
using TrixiParticles

fluid_density = 1000.0

H = 0.6
fluid_particle_spacing = H / 60

# Set the surface tension to a value that is accurate in your case.
# Note: This usually requires calibration to be physically accurate!
surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.25)

# `density_diffusion` is deactivated since the interaction with the surface tension model can
# cause stability problems.
# `adhesion_coefficient` needs to be set to a value so that the fluid doesn't separate
# from the boundary
# Note: The viscosity will be increased by the surface tension model hence we can reduce the artificial viscosity value.
# Note: The surface tension model leds to an increase in compressibility of the fluid
#       which needs to be rectified by an increase of the `sound_speed`.
# Note: The Wendland Kernels don't work very well here since the SurfaceTensionAkinci
#       model is optimized for smoothing_length=4r which is a too small smoothing_length for the Wendland Kernel functions.
# Note: Adhesion will result in friction at the boundary.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              surface_tension=surface_tension,
              fluid_particle_spacing=fluid_particle_spacing,
              smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
              smoothing_length=1.0 * fluid_particle_spacing,
              correction=AkinciFreeSurfaceCorrection(fluid_density),
              density_diffusion=nothing, adhesion_coefficient=0.5, alpha=0.001,
              sound_speed=100)

# sol = solve(ode, RDPK3SpFSAL35(),
#             abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
#             reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
#             dtmax=1e-2, # Limit stepsize to prevent crashing
#             dt=1e-5,
#             save_everystep=false, callback=callbacks);
