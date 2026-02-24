# ==========================================================================================
# 3D Pulsative Channel Flow Simulation (Weakly Compressible SPH)
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 3D pulsative channel flow simulation in a circular pipe
# including open boundary conditions.
# ==========================================================================================
using TrixiParticles

particle_spacing_factor = 30

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_3d.jl"),
              tspan=(0.0, 1.0), solution=nothing,
              particle_spacing_factor=particle_spacing_factor)

v_max = channel_diameter^2 * imposed_pressure_drop / (8 * dynamic_viscosity * channel_length)

function dynamic_pressure_drop(pos, t)
    return imposed_pressure_drop +
           (channel_length - pos[1]) / channel_length * imposed_pressure_drop * cos(t)
end

saving_callback = SolutionSavingCallback(dt=0.01, prefix="", output_directory="out")
extra_callback = nothing

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_3d.jl"),
              tspan=(0.0, 12.6), particle_spacing_factor=particle_spacing_factor,
              extra_callback=extra_callback, saving_callback=saving_callback, v_max=v_max,
              inlet_reference_pressure=dynamic_pressure_drop,
              outlet_reference_pressure=dynamic_pressure_drop)
