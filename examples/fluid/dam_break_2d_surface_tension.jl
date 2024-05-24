# This example shows how surface tension can be applied to existing cases
# and which parameters have to be changed!
using TrixiParticles

fluid_density = 1000.0

H = 0.6
fluid_particle_spacing = H / 60

# Set the surface tension to a value that is accurate in your case.
# Note: This usually requires calibration to be physically accurate!
surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.015)

# `density_diffusion` is deactivated since the interaction with the surface tension model can
# cause stability problems.
# `adhesion_coefficient` needs to be set to a value so that the fluid doesn't separate
# from the boundary
# Note: The surface tension model leads to an increase in compressibility of the fluid,
#       which needs to be rectified by an increase of the `sound_speed`.
# Note: The Wendland Kernels don't work very well here since the `SurfaceTensionAkinci`
#       model is optimized for a compact support of `2 * particle_spacing`, which would result
#       in a too small `smoothing_length` for the Wendland Kernel functions.
# Note: Adhesion will result in additional friction at the boundary.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              surface_tension=surface_tension,
              fluid_particle_spacing=fluid_particle_spacing,
              smoothing_kernel=WendlandC2Kernel{2}(),
              smoothing_length=2.5 * fluid_particle_spacing,
              correction=AkinciFreeSurfaceCorrection(fluid_density),
              density_diffusion=nothing, adhesion_coefficient=0.05,
              viscosity=ViscosityAdami(nu=0.00089),
              wall_viscosity=ViscosityAdami(nu=0.00089),
              sound_speed=100.0, tspan=(0.0, 2.0))
