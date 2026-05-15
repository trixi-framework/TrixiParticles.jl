# 2D dam break simulation using implicit incompressible SPH (IISPH) with pressure boundaries
using TrixiParticles
using OrdinaryDiffEqSymplecticRK

tspan = (0.0, 5.7 / sqrt(9.81 / 0.6))

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl");
              tspan, sol=nothing, ode=nothing)

# Change smoothing kernel and length
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Calculate kinematic viscosity for the viscosity model
nu = 0.02 * smoothing_length * sound_speed / 8
viscosity = ViscosityAdami(; nu)

# Use IISPH as fluid system
time_step = 1e-3
# Reduce omega when using pressure boundaries to ensure numerical stability
omega = 0.4

iisph_system = ImplicitIncompressibleSPHSystem(tank.fluid; smoothing_kernel,
                                               smoothing_length,
                                               reference_density=fluid_density, viscosity,
                                               acceleration=(0.0, -gravity),
                                               min_iterations=2, max_iterations=30, omega,
                                               time_step)

# Run the dam break simulation with these changes
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl");
              viscosity_fluid=viscosity, smoothing_kernel, smoothing_length,
              fluid_system=iisph_system,
              boundary_density_calculator=PressureBoundaries(; time_step, omega), tspan,
              state_equation=nothing, callbacks=CallbackSet(info_callback, saving_callback),
              time_integration_scheme=SymplecticEuler(), dt=time_step)
