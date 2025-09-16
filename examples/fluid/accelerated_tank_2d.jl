# ==========================================================================================
# 2D Accelerated Tank Example
#
# This setup is identical to `hydrostatic_water_column_2d.jl`, except that there is
# no gravity, and the tank is accelerated upwards instead.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# Resolution
fluid_particle_spacing = 0.05

# Function for moving boundaries
movement_function(x, t) = x + SVector(0.0, 0.5 * 9.81 * t^2)

is_moving(t) = true

boundary_movement = PrescribedMotion(movement_function, is_moving)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              fluid_particle_spacing=fluid_particle_spacing,
              prescribed_motion=boundary_movement,
              tspan=(0.0, 1.0), system_acceleration=(0.0, 0.0));
