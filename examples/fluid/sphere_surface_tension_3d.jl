# In this example we can observe that the `SurfaceTensionAkinci` surface tension model correctly leads to a
# surface minimization of the water cube and approaches a sphere.
using TrixiParticles
using OrdinaryDiffEq

fluid_density = 1000.0

particle_spacing = 0.15
fluid_size = (0.9, 0.9, 0.9)

sound_speed = 20.0

# For all surface tension simulations, we need a compact support of `2 * particle_spacing`
smoothing_length = 1.0 * particle_spacing

nu = 0.3

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "sphere_surface_tension_2d.jl"),
              surface_tension_coefficient=0.15,
              tspan=(0.0, 100.0), nu=nu, smoothing_length = 3.0 * particle_spacing,
              fluid_smoothing_kernel=WendlandC2Kernel{3}(),
              particle_spacing=particle_spacing, sound_speed=sound_speed,
              fluid_density=fluid_density, fluid_size=fluid_size)
