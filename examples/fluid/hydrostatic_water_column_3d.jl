# ==========================================================================================
# 3D Hydrostatic Water Column Simulation
#
# This example sets up a 3D hydrostatic water column by including and modifying
# the 2D `hydrostatic_water_column_2d.jl` setup.
# ==========================================================================================

using TrixiParticles

trixi_include(@__MODULE__,
              joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"),
              fluid_particle_spacing=0.05, initial_fluid_size=(1.0, 1.0, 0.9),
              tank_size=(1.0, 1.0, 1.2), acceleration=(0.0, 0.0, -9.81),
              smoothing_kernel=SchoenbergCubicSplineKernel{3}(), tspan=(0.0, 1.0),
              maxiters=10^5, fluid_density_calculator=ContinuityDensity(),
              clip_negative_pressure=false)
