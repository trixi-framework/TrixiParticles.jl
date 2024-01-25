using TrixiParticles
using OrdinaryDiffEq

# ==========================================================================================
# ==== Resolution
scale_factor = 0.015
particle_spacing = 0.01

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 10.0)

fluid_density = 1000.0
sound_speed = 20.0

filename_tank = joinpath(examples_dir(), "preprocessing", "tank_complex2.asc")
filename_fluid = joinpath(examples_dir(), "preprocessing", "fluid.asc")

tank = ComplexShape(; filename=filename_tank, particle_spacing,
                    density=fluid_density, scale_factor,
                    point_in_shape_algorithm=WindingNumberHorman())

fluid = ComplexShape(; filename=filename_fluid, particle_spacing,
                     density=fluid_density, scale_factor,
                     point_in_shape_algorithm=WindingNumberHorman())
fluid = setdiff(fluid, tank)

viscosity = ArtificialViscosityMonaghan(alpha=0.1, beta=0.0)
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
#density_diffusion = DensityDiffusionAntuono(fluid, delta=0.1)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=particle_spacing, fluid=fluid, boundary=tank,
              tspan=tspan, sound_speed=sound_speed, viscosity=viscosity,
              density_diffusion=density_diffusion)
