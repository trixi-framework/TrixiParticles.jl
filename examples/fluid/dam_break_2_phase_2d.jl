# 2D dam break simulation with an oil film on top

using TrixiParticles
using OrdinaryDiffEq

# Size parameters
H = 0.6
W = 2 * H

gravity = 9.81
tspan = (0.0, 0.2)

# Resolution
fluid_particle_spacing = H / 60

# Numerical settings
smoothing_length = 3.5 * fluid_particle_spacing
sound_speed = 100

nu = 0.02 * smoothing_length * sound_speed / 8
oil_viscosity = ViscosityMorris(nu=0.1 * nu)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, fluid_particle_spacing=fluid_particle_spacing,
              viscosity=ViscosityMorris(nu=nu), smoothing_length=smoothing_length,
              gravity=gravity, tspan=tspan,
              density_diffusion=nothing,
              sound_speed=sound_speed,
              surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.02))

# TODO: broken (fixed?)
# trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
# sol=nothing, fluid_particle_spacing=fluid_particle_spacing,
# viscosity=ViscosityMorris(nu=nu), smoothing_length=smoothing_length,
# gravity=gravity, tspan=tspan,
# density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
# sound_speed=sound_speed,
# surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.02),
# correction=AkinciFreeSurfaceCorrection(1000.0))

# ==========================================================================================
# ==== Setup oil layer

oil_size = (tank_size[1], 1.0 * H)
oil_size2 = (tank_size[1]-W, H)
oil_density = 200.0

oil = RectangularShape(fluid_particle_spacing,
                       round.(Int, oil_size ./ fluid_particle_spacing),
                       zeros(length(oil_size)), density=oil_density)

oil2 = RectangularShape(fluid_particle_spacing,
                       round.(Int, oil_size2 ./ fluid_particle_spacing),
                       (W, 0.0), density=oil_density)

# move on top of the water
for i in axes(oil.coordinates, 2)
    oil.coordinates[:, i] .+= [0.0, H]
end

oil = union(oil, oil2)

oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
                                         StateEquationCole(; sound_speed,
                                                           reference_density=oil_density,
                                                           exponent=1,
                                                           clip_negative_pressure=false),
                                         smoothing_kernel,
                                         smoothing_length, viscosity=oil_viscosity,
                                         #density_diffusion=density_diffusion,
                                         acceleration=(0.0, -gravity))

# oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
#                                          StateEquationIdealGas(; gas_constant=287.0, temperature=293.0, gamma=1.4),
#                                          smoothing_kernel,
#                                          smoothing_length, viscosity=oil_viscosity,
#                                          density_diffusion=density_diffusion,
#                                          acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, oil_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing))
ode = semidiscretize(semi, tspan, data_type=nothing)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
