# 2D dam break simulation using implicit incompressible SPH (IISPH)
using TrixiParticles

fluid_particle_spacing = 0.6 / 40

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing,
              sol=nothing, ode=nothing)

# IISPH doesn't require a large compact support like WCSPH and performs worse with a typical
# smoothing length used for WCSPH.
smoothing_length = 1.0 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()
# This kernel slightly overestimates the density, so we reduce the mass slightly
# to obtain a density slightly below the reference density.
# Otherwise, the fluid will jump slightly at the beginning of the simulation.
tank.fluid.mass .*= 0.995

# Calculate kinematic viscosity for the viscosity model.
# Only ViscosityAdami and ViscosityMorris can be used for IISPH simulations since they don't
# require a speed of sound.
nu = 0.02 * smoothing_length * sound_speed / 8
viscosity = ViscosityAdami(; nu)

# Use IISPH as fluid system
time_step = 1e-3
fluid_system = ImplicitIncompressibleSPHSystem(tank.fluid, smoothing_kernel,
                                               smoothing_length, fluid_density,
                                               viscosity=ViscosityAdami(nu=nu),
                                               acceleration=(0.0, -gravity),
                                               min_iterations=2,
                                               max_iterations=30,
                                               time_step=time_step)

# Run the dam break simulation with these changes
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              neighborhood_search=GridNeighborhoodSearch{2}(),
              viscosity_fluid=ViscosityAdami(nu=nu),
              smoothing_kernel=smoothing_kernel,
              smoothing_length=smoothing_length,
              fluid_system=fluid_system,
              boundary_density_calculator=PressureZeroing(),
              tspan=tspan,
              state_equation=nothing,
              callbacks=CallbackSet(info_callback, saving_callback),
              time_integration_scheme=SymplecticEuler(), dt=time_step)
