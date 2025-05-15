# ==========================================================================================
# 2D Two-Phase Dam Break Simulation (Water and Air)
#
# This example simulates a 2D dam break with an air layer above the water.
# It demonstrates how to set up a multi-fluid simulation in TrixiParticles.jl.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Physical and Resolution Parameters
# ------------------------------------------------------------------------------

tank_height = 0.6
tank_width = 2 * tank_height

# Particle spacing (resolution)
# Note: This is a very coarse resolution. For better results, decrease this value
# e.g., `tank_height / 60` but will take over 1 hour!
fluid_particle_spacing = tank_height / 20

# ------------------------------------------------------------------------------
# Experiment Setup:
# ------------------------------------------------------------------------------

# Gravitational acceleration
gravity = 9.81

# Simulation time span
tspan = (0.0, 2.0)

# SPH numerical parameters
smoothing_length = 3.5 * fluid_particle_spacing
sound_speed = 100.0
# Alternative sound speed for air if using StateEquationIdealGas:
# sound_speed_air_ideal_gas = 343.0 # Speed of sound of air at 20°C

# Kinematic viscosities (physical values)
nu_water_physical = 8.9E-7 # m^2/s at 20°C
nu_air_physical = 1.544E-5 # m^2/s at 20°C
nu_ratio = nu_water_physical / nu_air_physical

# Numerical viscosities for SPH
# Here, we set a numerical viscosity for air and scale it for water.
# This is done for stability reasons at the given coarse resolution.
nu_sim_air = max(0.02 * smoothing_length * sound_speed, nu_air_physical)
nu_sim_water = nu_ratio * nu_sim_air

air_viscosity = ViscosityMorris(nu=nu_sim_air)
water_viscosity = ViscosityMorris(nu=nu_sim_water)

# ------------------------------------------------------------------------------
# Load Base Dam Break Simulation for Water System
# ------------------------------------------------------------------------------

# Include the single-phase dam break setup to initialize water particles and boundaries.
# `sol=nothing` prevents the included simulation from running.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, # Don't run the included simulation
              fluid_particle_spacing=fluid_particle_spacing,
              viscosity=water_viscosity, # Use the two-phase water viscosity
              smoothing_length=smoothing_length,
              gravity=gravity,
              tspan=tspan,
              density_diffusion=nothing,
              sound_speed=sound_speed,
              exponent=7,
              reference_particle_spacing=fluid_particle_spacing,
              # Adjust tank size for the air layer by making the tank taller
              tank_size=(floor(5.366 * tank_height / fluid_particle_spacing) *
                         fluid_particle_spacing,
                         2.6 * tank_height))

# ------------------------------------------------------------------------------
# Setup Air System Layer
# ------------------------------------------------------------------------------
# Define air layer dimensions. The air occupies the space above the initial water
# and the initially empty part of the tank.
air_region_above_water_size = (initial_water_width, tank_size[2] - tank_height)
air_region_empty_tank_size = (tank_size[1] - initial_water_width, tank_size[2])

air_density = 1.0

# Create air particles for the region directly above the initial water column
air_particles_above_water = RectangularShape(fluid_particle_spacing,
                                             round.(Int,
                                                    air_region_above_water_size ./
                                                    fluid_particle_spacing),
                                             (0.0, tank_height), # Positioned on top of the water
                                             density=air_density)

# Create air particles for the rest of the empty volume in the tank
air_particles_empty_tank = RectangularShape(fluid_particle_spacing,
                                            round.(Int,
                                                   air_region_empty_tank_size ./
                                                   fluid_particle_spacing),
                                            (initial_water_width, 0.0), # Positioned next to the initial water
                                            density=air_density)

# Combine the two air particle sets
air_particles = union(air_particles_above_water, air_particles_empty_tank)

# Equation of state for air
air_eos = StateEquationCole(sound_speed=sound_speed, reference_density=air_density,
                            exponent=1, clip_negative_pressure=false)
# Alternative: Ideal Gas Law
# air_eos = StateEquationIdealGas(sound_speed=sound_speed_air_ideal_gas,
#                                 reference_density=air_density, gamma=1.4)

air_sph_system = WeaklyCompressibleSPHSystem(air_particles, fluid_density_calculator,
                                             air_eos,
                                             smoothing_kernel, smoothing_length,
                                             viscosity=air_viscosity,
                                             acceleration=(0.0, -gravity),
                                             reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Simulation:
# ------------------------------------------------------------------------------

semi = Semidiscretization(fluid_system, air_sph_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())
ode = semidiscretize(semi, tspan)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-5, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);
