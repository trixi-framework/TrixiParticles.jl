# 2D dam break simulation with an oil film on top

using TrixiParticles
using OrdinaryDiffEq

# Size parameters
H = 0.6
W = 2 * H

gravity = 9.81
tspan = (0.0, 5.7 / sqrt(gravity))

# Resolution
fluid_particle_spacing = H / 60

# Numerical settings
smoothing_length = 3.5 * fluid_particle_spacing
sound_speed = 20 * sqrt(gravity * H)

nu = 0.02 * smoothing_length * sound_speed / 8
oil_viscosity = ViscosityMorris(nu=10 * nu)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, fluid_particle_spacing=fluid_particle_spacing,
              viscosity=ViscosityMorris(nu=nu), smoothing_length=smoothing_length,
              gravity=gravity,
              density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
              sound_speed=sound_speed)

# ==========================================================================================
# ==== Setup oil layer

oil_size = (W, 0.1 * H)
oil_density = 700.0

oil = RectangularShape(fluid_particle_spacing,
                       round.(Int, oil_size ./ fluid_particle_spacing),
                       zeros(length(oil_size)), density=oil_density)

# move on top of the water
for i in axes(oil.coordinates, 2)
    oil.coordinates[:, i] .+= [0.0, H]
end

oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
                                         state_equation, smoothing_kernel,
                                         smoothing_length, viscosity=oil_viscosity,
                                         density_diffusion=density_diffusion,
                                         acceleration=(0.0, -gravity),
                                         surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.02),
                                         correction=AkinciFreeSurfaceCorrection(oil_density))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, oil_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing))
ode = semidiscretize(semi, tspan, data_type=nothing)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
