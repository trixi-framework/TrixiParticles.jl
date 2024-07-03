# 2D dam break simulation with an oil film on top

using TrixiParticles
using OrdinaryDiffEq

# Size parameters
H = 0.6
W = 2 * H

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = H / 60

# Change spacing ratio to 3 and boundary layers to 1 when using Monaghan-Kajtar boundary model
boundary_layers = 4
spacing_ratio = 1

boundary_particle_spacing = fluid_particle_spacing / spacing_ratio

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81

tspan = (0.0, 5.7 / sqrt(gravity))

# Boundary geometry and initial fluid particle positions
initial_fluid_size = (W, H)
oil_size = (W, 0.1 * H)
tank_size = (floor(5.366 * H / boundary_particle_spacing) * boundary_particle_spacing, 4.0)

fluid_density = 1000.0
oil_density = 700.0

sound_speed = 20 * sqrt(gravity * H)
state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1, clip_negative_pressure=false)

tank = RectangularTank(fluid_particle_spacing, initial_fluid_size, tank_size, fluid_density,
                       n_layers=boundary_layers, spacing_ratio=spacing_ratio,
                       acceleration=(0.0, -gravity), state_equation=state_equation)

oil = RectangularShape(fluid_particle_spacing,
                       round.(Int, oil_size ./ fluid_particle_spacing),
                       zeros(length(oil_size)), density=oil_density)

# move on top of the water
for i in axes(oil.coordinates, 2)
    oil.coordinates[:, i] .+= [0.0, H]
end

# ==========================================================================================
# ==== Fluid
smoothing_length = 3.5 * fluid_particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

nu = 0.02 * smoothing_length * sound_speed / 8
viscosity = ViscosityMorris(nu=nu)
oil_viscosity = ViscosityMorris(nu=10 * nu)

# Alternatively the density diffusion model by Molteni & Colagrossi can be used,
# which will run faster.
density_diffusion = DensityDiffusionMolteniColagrossi(delta=0.1)
# density_diffusion = DensityDiffusionAntuono(tank.fluid, delta=0.1)
# oil_density_diffusion = DensityDiffusionAntuono(oil, delta=0.1)

fluid_system = WeaklyCompressibleSPHSystem(tank.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           smoothing_length, viscosity=viscosity,
                                           density_diffusion=density_diffusion,
                                           acceleration=(0.0, -gravity), correction=nothing,
                                           surface_tension=nothing)

oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
                                         state_equation, smoothing_kernel,
                                         smoothing_length, viscosity=oil_viscosity,
                                         density_diffusion=density_diffusion,
                                         acceleration=(0.0, -gravity),
                                         surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.02),
                                         correction=AkinciFreeSurfaceCorrection(oil_density))

# ==========================================================================================
# ==== Boundary
boundary_density_calculator = AdamiPressureExtrapolation()
boundary_model = BoundaryModelDummyParticles(tank.boundary.density, tank.boundary.mass,
                                             state_equation=state_equation,
                                             boundary_density_calculator,
                                             smoothing_kernel, smoothing_length,
                                             correction=nothing)

boundary_system = BoundarySPHSystem(tank.boundary, boundary_model, adhesion_coefficient=0.0)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, oil_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch,
                          threaded_nhs_update=true)
ode = semidiscretize(semi, tspan, data_type=nothing)

info_callback = InfoCallback(interval=100)

solution_prefix = ""
saving_callback = SolutionSavingCallback(dt=0.02, prefix=solution_prefix)

# Save at certain timepoints which allows comparison to the results of Marrone et al.,
# i.e. (1.5, 2.36, 3.0, 5.7, 6.45).
# Please note that the images in Marrone et al. are obtained at a particle_spacing = H/320,
# which takes between 2 and 4 hours.
saving_paper = SolutionSavingCallback(save_times=[0.0, 0.371, 0.584, 0.743, 1.411, 1.597],
                                      prefix="marrone_times")

# This can be overwritten with `trixi_include`
extra_callback = nothing

use_reinit = false
density_reinit_cb = use_reinit ?
                    DensityReinitializationCallback(semi.systems[1], interval=10) :
                    nothing
stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(info_callback, saving_callback, stepsize_callback, extra_callback,
                        density_reinit_cb, saving_paper)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);