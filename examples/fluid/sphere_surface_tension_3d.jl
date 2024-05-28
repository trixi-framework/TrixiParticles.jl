# In this example we can observe that the `SurfaceTensionAkinci` surface tension model correctly leads to a
# surface minimization of the water cube and approaches a sphere.
using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0

particle_spacing = 0.1
fluid_size = (0.9, 0.9, 0.9)

sound_speed = 20.0

# For all surface tension simulations, we need a compact support of `2 * particle_spacing`
smoothing_length = 1.0 * particle_spacing

nu = 0.01

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "sphere_surface_tension_2d.jl"),
              saving_interval=0.1, cfl=1.2, surface_tension_coefficient=0.1,
              tspan=(0.0, 10.0), nu=nu,
              alpha=10 * nu / (smoothing_length * sound_speed),
              smoothing_kernel=SchoenbergCubicSplineKernel{3}(),
              particle_spacing=particle_spacing, sound_speed=sound_speed,
              fluid_density=fluid_density, fluid_size=fluid_size)
