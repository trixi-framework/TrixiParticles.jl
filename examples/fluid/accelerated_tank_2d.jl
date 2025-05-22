# ==========================================================================================
# 2D Accelerated Tank Example
#
# This setup is identical to `hydrostatic_water_column_2d.jl`, except that there is
# no gravity, and the tank is accelerated upwards instead.
# ==========================================================================================

using TrixiParticles
using OrdinaryDiffEq

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
# Resolution of the fluid
particle_spacing = 0.05

# Function defining the boundary movement: tank accelerates upwards
# S(t) = 0.5 * a * t^2, with a = g
movement_function(t) = SVector(0.0, 0.5 * 9.81 * t^2)

# Function indicating if the boundary is moving at a given time
# In this example, the boundary is always moving.
is_moving(t) = true

# Create a BoundaryMovement object
boundary_movement = BoundaryMovement(movement_function, is_moving)

# ------------------------------------------------------------------------------
# Simulation Setup
# ------------------------------------------------------------------------------
# Include the hydrostatic water column setup from the 2D examples.
# We override specific parameters:
# - `fluid_particle_spacing`: To use the resolution defined above.
# - `movement`: To apply the upward acceleration of the tank.
# - `tspan`: To set the simulation time span for this example.
# - `system_acceleration`: Set to zero, as the physical effect of gravity
#                          is modeled by the tank's upward acceleration.
trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              particle_spacing=particle_spacing,
              movement=boundary_movement,
              tspan=(0.0, 1.0),
              system_acceleration=(0.0, 0.0))
