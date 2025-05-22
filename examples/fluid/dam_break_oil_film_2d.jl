# ==========================================================================================
# 2D Dam Break Simulation with an Oil Film
#
# This example simulates a 2D dam break where a layer of oil sits on top of the water.
# It demonstrates a multi-fluid simulation with two immiscible fluids.
# The base water setup is loaded from the `dam_break_2d.jl` example.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Physical and Resolution Parameters for the Oil Film Extension
# ------------------------------------------------------------------------------

initial_water_height = 0.6
initial_water_width = 2 * initial_water_height

# Particle spacing or resolution
fluid_particle_spacing = initial_water_height / 60

# Gravitational acceleration
gravity = 9.81

# SPH numerical parameters
smoothing_length = 1.75 * fluid_particle_spacing
sound_speed = 20 * sqrt(gravity * initial_water_height)

# Physical kinematic viscosities
nu_water_physical = 8.9e-7  # m^2/s at 20°C
nu_oil_physical = 6.0e-5    # m^2/s for a generic oil

# Numerical viscosities for SPH simulation
# Artificial viscosity for oil, scaled for water based on viscosity ratio.
# This helps maintain stability, especially at coarser resolutions.
nu_ratio_physical = nu_water_physical / nu_oil_physical
nu_sim_oil = max(0.01 * smoothing_length * sound_speed, nu_oil_physical)
nu_sim_water = nu_ratio_physical * nu_sim_oil

# Viscosity models for each fluid
water_viscosity_model = ViscosityMorris(nu=nu_sim_water)
oil_viscosity_model = ViscosityMorris(nu=nu_sim_oil)

# Simulation time span
tspan = (0.0, 5.7 / sqrt(gravity))

# ------------------------------------------------------------------------------
# Load Base Dam Break Simulation for Water System
# ------------------------------------------------------------------------------

# Include the single-phase dam break setup to initialize water particles and boundaries.
# `sol=nothing` prevents the included simulation from running.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, # Don't run the included simulation yet
              ode=nothing,
              fluid_particle_spacing=fluid_particle_spacing,
              viscosity=water_viscosity_model,
              smoothing_length=smoothing_length,
              gravity=gravity,
              density_diffusion=nothing, # No density diffusion for water in this setup
              sound_speed=sound_speed,
              prefix="",
              reference_particle_spacing=fluid_particle_spacing,
              tspan=tspan)

# ------------------------------------------------------------------------------
# Setup Oil Layer System
# ------------------------------------------------------------------------------
oil_layer_height = 0.1 * initial_water_height
oil_initial_size = (initial_water_width, oil_layer_height)
oil_density = 700.0 # kg/m^3

# Equation of state for oil
oil_state_equation = StateEquationCole(sound_speed=sound_speed,
                                       reference_density=oil_density,
                                       exponent=1, # Typically 1 for nearly incompressible fluids like oil
                                       clip_negative_pressure=false)

# Create oil particles as a rectangular shape
oil_particles = RectangularShape(fluid_particle_spacing,
                                 round.(Int, oil_initial_size ./ fluid_particle_spacing),
                                 (0.0, 0.0), # Initial position, will be moved
                                 density=oil_density)

# Position the oil layer on top of the initial water column
initial_oil_position_offset = SVector(0.0, initial_water_height)
for i in axes(oil_particles.coordinates, 2)
    oil_particles.coordinates[:, i] .+= initial_oil_position_offset
end

oil_surface_tension = SurfaceTensionAkinci(surface_tension_coefficient=0.01)

oil_system = WeaklyCompressibleSPHSystem(oil_particles, fluid_density_calculator,
                                         oil_state_equation, smoothing_kernel,
                                         smoothing_length,
                                         viscosity=oil_viscosity_model,
                                         acceleration=(0.0, -gravity),
                                         surface_tension=oil_surface_tension,
                                         correction=AkinciFreeSurfaceCorrection(oil_density),
                                         reference_particle_spacing=fluid_particle_spacing)

# Alternative oil system with physical surface tension model
# oil_system = WeaklyCompressibleSPHSystem(oil, fluid_density_calculator,
#                                          oil_eos, smoothing_kernel,
#                                          smoothing_length, viscosity=oil_viscosity,
#                                          acceleration=(0.0, -gravity),
#                                          surface_tension=SurfaceTensionMorris(surface_tension_coefficient=0.03),
#                                          reference_particle_spacing=fluid_particle_spacing)

# ------------------------------------------------------------------------------
# Simulation:
# ------------------------------------------------------------------------------

semi = Semidiscretization(fluid_system, oil_system, boundary_system,
                          neighborhood_search=GridNeighborhoodSearch{2}(update_strategy=nothing),
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # This will be overwritten by the stepsize callback
            save_everystep=false,
            callback=callbacks);
