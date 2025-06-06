# 2D dam break simulation using an ImplicitIncompressible SPH system
using TrixiParticles

# Load setup from dam break example
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing, ode=nothing)

# Change smoohing kernel and length
smoothing_length = 1.25 * fluid_particle_spacing
smoothing_kernel = GaussianKernel{2}()

# Calculate nu for the viscosity model
nu = 0.02 * smoothing_length * sound_speed/8

# Use IISPH as fluid system
time_step=0.001
min_iterations=10
max_iterations=30
IISPH_system = ImplicitIncompressibleSPHSystem(tank.fluid, smoothing_kernel,
                                               smoothing_length, fluid_density,
                                               viscosity=ViscosityAdami(nu=nu),
                                               acceleration=(0.0, -gravity),
                                               min_iterations=min_iterations,
                                               max_iterations=max_iterations,
                                               time_step=time_step)

# Run the dam break simulation with these changes
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              viscosity=ViscosityAdami(nu=nu),
              fluid_system=IISPH_system,
              boundary_density_calculator=PressureZeroing(),
              state_equation=nothing,
              callbacks=CallbackSet(info_callback, saving_callback),
              solver_function=SymplecticEuler(), dt=time_step)
