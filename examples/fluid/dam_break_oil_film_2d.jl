# ==========================================================================================
# 2D Dam Break Simulation with an Oil Film
#
# This example simulates a 2D dam break where a layer of oil sits on top of the water.
# It demonstrates a multi-fluid simulation with two immiscible fluids.
# The base water setup is loaded from the `dam_break_2d.jl` example.
# ==========================================================================================

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
smoothing_length = 1.75 * fluid_particle_spacing
sound_speed = 20 * sqrt(gravity * H)

# Physical values
nu_water = 8.9e-7
nu_oil = 6e-5
nu_ratio = nu_water / nu_oil

nu_sim_oil = max(0.01 * smoothing_length * sound_speed, nu_oil)
nu_sim_water = nu_ratio * nu_sim_oil

oil_viscosity = ViscosityMorris(nu=nu_sim_oil)

# TODO: broken if both systems use surface tension
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, fluid_particle_spacing=fluid_particle_spacing, tspan=tspan,
              viscosity_fluid=ViscosityMorris(nu=nu_sim_water),
              smoothing_length=smoothing_length,
              gravity=gravity, density_diffusion=nothing, sound_speed=sound_speed,
              prefix="", reference_particle_spacing=fluid_particle_spacing)

# ==========================================================================================
# ==== Setup oil layer

oil_size = (W, 0.1 * H)
oil_density = 700.0
oil_eos = StateEquationCole(; sound_speed, reference_density=oil_density, exponent=1,
                            clip_negative_pressure=false)

oil = RectangularShape(fluid_particle_spacing,
                       round.(Int, oil_size ./ fluid_particle_spacing),
                       zeros(length(oil_size)), density=oil_density)

# move on top of the water
for i in axes(oil.coordinates, 2)
    oil.coordinates[:, i] .+= [0.0, H]
end

oil_state_equation = StateEquationCole(; sound_speed, reference_density=oil_density,
                                       exponent=1, clip_negative_pressure=false)
oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
                                         oil_eos, smoothing_kernel,
                                         smoothing_length, viscosity=oil_viscosity,
                                         acceleration=(0.0, -gravity),
                                         surface_tension=SurfaceTensionAkinci(surface_tension_coefficient=0.01),
                                         correction=AkinciFreeSurfaceCorrection(oil_density),
                                         reference_particle_spacing=fluid_particle_spacing)

# oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
#                                          oil_eos, smoothing_kernel,
#                                          smoothing_length, viscosity=oil_viscosity,
#                                          acceleration=(0.0, -gravity),
#                                          surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.03),
#                                          reference_particle_spacing=fluid_particle_spacing)

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, oil_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This is overwritten by the stepsize callback
            save_everystep=false, callback=callbacks);
