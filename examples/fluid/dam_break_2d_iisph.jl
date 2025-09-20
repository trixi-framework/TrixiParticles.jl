# 2D dam break simulation using implicit incompressible SPH (IISPH)
using TrixiParticles

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, ode=nothing)

# Change smoothing kernel and length to get a stable simulation
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_kernel = SchoenbergCubicSplineKernel{2}()

# Calculate kinematic viscosity for the viscosity model
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
              viscosity_fluid=ViscosityAdami(nu=nu),
              smoothing_kernel=smoothing_kernel,
              smoothing_length=smoothing_length,
              fluid_system=fluid_system,
              boundary_density_calculator=PressureZeroing(),
              tspan=tspan,
              state_equation=nothing,
              callbacks=CallbackSet(info_callback, saving_callback),
              time_integration_scheme=SymplecticEuler(), dt=time_step)