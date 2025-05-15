################################################################################
# Dam Break 2D with Air Layer Example
#
# This example simulates a 2D dam break with an air layer above the water.
# The simulation uses a coarse resolution; note that a finer resolution (e.g. tank_height/60)
# yields more accurate results but increases computation time significantly.
################################################################################

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Physical Size and Resolution Parameters
# ------------------------------------------------------------------------------
tank_height = 0.6
tank_width = 2 * tank_height

# Note: The resolution is very coarse.
# A better result is obtained with tank_height/60 or higher (which takes over 1 hour)
fluid_particle_spacing = tank_height / 20

# ------------------------------------------------------------------------------
# Experiment Setup: Physical and Numerical Parameters
# ------------------------------------------------------------------------------
gravity = 9.81
tspan = (0.0, 2.0)

# Numerical settings
smoothing_length = 3.5 * fluid_particle_spacing
sound_speed = 100.0
# Alternative sound speed for the ideal gas equation:
# sound_speed = 343.0

# physical values
nu_water = 8.9E-7
nu_air = 1.544E-5
nu_ratio = nu_water / nu_air

nu_sim_air = 0.02 * smoothing_length * sound_speed
nu_sim_water = nu_ratio * nu_sim_air

air_viscosity = ViscosityMorris(nu=nu_sim_air)
water_viscosity = ViscosityMorris(nu=nu_sim_water)

# ------------------------------------------------------------------------------
# Load Dam Break Simulation (Water System)
# ------------------------------------------------------------------------------
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, fluid_particle_spacing=fluid_particle_spacing,
              viscosity=water_viscosity, smoothing_length=smoothing_length,
              gravity=gravity, tspan=tspan, density_diffusion=nothing,
              sound_speed=sound_speed, exponent=7,
              reference_particle_spacing=fluid_particle_spacing,
              tank_size=(floor(5.366 * tank_height / fluid_particle_spacing) *
                         fluid_particle_spacing,
                         2.6 * tank_height))

# ------------------------------------------------------------------------------
# Setup Air System Layer
# ------------------------------------------------------------------------------
# Define air layer dimensions for two regions:
#   - air_system: air immediately above the initial water
#   - air_system2: air in the remaining empty volume

air_size = (tank_size[1], 1.5 * tank_height)
air_size2 = (tank_size[1] - tank_width, tank_height)
air_density = 1.0

# Air above the initial water
air_system = RectangularShape(fluid_particle_spacing,
                              round.(Int, air_size ./ fluid_particle_spacing),
                              zeros(length(air_size)), density=air_density)

# Air for the rest of the empty volume
air_system2 = RectangularShape(fluid_particle_spacing,
                               round.(Int, air_size2 ./ fluid_particle_spacing),
                               (tank_width, 0.0), density=air_density)

# move on top of the water
for i in axes(air_system.coordinates, 2)
    air_system.coordinates[:, i] .+= [0.0, tank_height]
end

air_system = union(air_system, air_system2)

air_eos = StateEquationCole(; sound_speed, reference_density=air_density, exponent=1,
                            clip_negative_pressure=false)
#air_eos = StateEquationIdealGas(; sound_speed, reference_density=air_density, gamma=1.4)

air_system_system = WeaklyCompressibleSPHSystem(air_system, fluid_density_calculator,
                                                air_eos, smoothing_kernel, smoothing_length,
                                                viscosity=air_viscosity,
                                                acceleration=(0.0, -gravity))

# ------------------------------------------------------------------------------
# Simulation: Combine Water and Air Systems and Solve
# ------------------------------------------------------------------------------
semi = Semidiscretization(fluid_system, air_system_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
