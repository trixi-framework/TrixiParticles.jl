# ==========================================================================================
# 3D Pulsatile Channel Flow Simulation (Weakly Compressible SPH)
#
# Based on:
#   Zhan, X., et al. "Dynamical pressure boundary condition for weakly compressible smoothed particle hydrodynamics"
#   Physics of Fluids, Volume 37
#   https://doi.org/10.1063/5.0254575
#
# This example sets up a 3D pulsatile channel flow simulation in a circular pipe
# including open boundary conditions.
# ==========================================================================================
using TrixiParticles

particle_spacing_factor = 30

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
              tspan=(0.0, 1.0), sol=nothing,
              particle_spacing_factor=particle_spacing_factor)

v_max = pipe_diameter^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

function dynamic_pressure_drop(pos, t)
    return pressure_drop + (flow_length - pos[1]) / flow_length * pressure_drop * cos(t)
end

saving_callback = SolutionSavingCallback(dt=0.01, prefix="", output_directory="out")
extra_callback = nothing

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
              tspan=(0.0, 12.6), particle_spacing_factor=particle_spacing_factor,
              extra_callback=extra_callback, saving_callback=saving_callback, v_max=v_max,
              reference_pressure_in=dynamic_pressure_drop,
              reference_pressure_out=dynamic_pressure_drop)
