# This setup is identical to `hydrostatic_water_column_2d.jl`, except that now there is
# no gravity, and the tank is accelerated upwards instead.
# Note that the two setups are physically identical, but produce different numerical errors.
using TrixiParticles
using OrdinaryDiffEq

# Resolution
fluid_particle_spacing = 0.05

# Function for moving boundaries
movement_function(t) = SVector(0.0, 0.5 * 9.81 * t^2)

is_moving(t) = true

boundary_movement = BoundaryMovement(movement_function, is_moving)

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              fluid_particle_spacing = fluid_particle_spacing, movement = boundary_movement,
              tspan = (0.0, 1.0), system_acceleration = (0.0, 0.0));
