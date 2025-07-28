# ==========================================================================================
# 2D Two-Phase Dam Break Simulation (Water and Air)
#
# This example simulates a 2D dam break with an air layer above the water.
# It demonstrates how to set up a multi-fluid simulation in TrixiParticles.jl.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# Size parameters
H = 0.6
W = 2 * H

# ==========================================================================================
# ==== Resolution

# Note: The resolution is very coarse. A better result is obtained with H/60 or higher (which takes over 1 hour)
fluid_particle_spacing = H / 20

# ==========================================================================================
# ==== Experiment Setup
gravity = 9.81
tspan = (0.0, 2.0)

# Numerical settings
smoothing_length = 3.5 * fluid_particle_spacing
sound_speed = 100.0
# when using the Ideal gas equation
# sound_speed = 343.0

# physical values
nu_water = 8.9E-7
nu_air = 1.544E-5
nu_ratio = nu_water / nu_air

nu_sim_air = 0.02 * smoothing_length * sound_speed
nu_sim_water = nu_ratio * nu_sim_air

air_viscosity = ViscosityMorris(nu=nu_sim_air)
water_viscosity = ViscosityMorris(nu=nu_sim_water)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, fluid_particle_spacing=fluid_particle_spacing,
              viscosity_fluid=water_viscosity, smoothing_length=smoothing_length,
              gravity=gravity, tspan=tspan, density_diffusion=nothing,
              sound_speed=sound_speed, exponent=7,
              tank_size=(floor(5.366 * H / fluid_particle_spacing) * fluid_particle_spacing,
                         2.6 * H))

# ==========================================================================================
# ==== Setup air_system layer

air_size = (tank_size[1], 1.5 * H)
air_size2 = (tank_size[1] - W, H)
air_density = 1.0

# Air above the initial water
air_system = RectangularShape(fluid_particle_spacing,
                              round.(Int, air_size ./ fluid_particle_spacing),
                              zeros(length(air_size)), density=air_density)

# Air for the rest of the empty volume
air_system2 = RectangularShape(fluid_particle_spacing,
                               round.(Int, air_size2 ./ fluid_particle_spacing),
                               (W, 0.0), density=air_density)

# move on top of the water
for i in axes(air_system.coordinates, 2)
    air_system.coordinates[:, i] .+= [0.0, H]
end

air_system = union(air_system, air_system2)

air_eos = StateEquationCole(; sound_speed, reference_density=air_density, exponent=1,
                            clip_negative_pressure=false)
#air_eos = StateEquationIdealGas(; sound_speed, reference_density=air_density, gamma=1.4)

air_system_system = WeaklyCompressibleSPHSystem(air_system, fluid_density_calculator,
                                                air_eos, smoothing_kernel, smoothing_length,
                                                viscosity=air_viscosity,
                                                acceleration=(0.0, -gravity))

# ==========================================================================================
# ==== Simulation
semi = Semidiscretization(fluid_system, air_system_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
