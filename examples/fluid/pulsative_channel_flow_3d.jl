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

# A more reasonable particle spacing factor is 30, which takes around an hour to 0.5s simulation time
particle_spacing_factor = 15

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_3d.jl"),
              sol=nothing, particle_spacing_factor=particle_spacing_factor)

v_max = channel_diameter^2 * imposed_pressure_drop /
        (8 * dynamic_viscosity * channel_length)

# Use `let` block to define the function with the *current values* of the global variables,
# instead of reading the globals every time it is called, which would make it slow.
let imposed_pressure_drop = imposed_pressure_drop, channel_length = channel_length
    global function dynamic_pressure_drop(pos, t)
        return imposed_pressure_drop +
               (channel_length - pos[1]) / channel_length * imposed_pressure_drop * cos(t)
    end
end

saving_callback = SolutionSavingCallback(dt=0.01, prefix="", output_directory="out")
extra_callback = nothing

# simulation end time has been shortened to avoid long runtimes in the example
# a more full pulse is seen with a runtime of 3.0, which takes around 30m at particle_spacing_factor = 15 and around 6h at particle_spacing_factor = 30
simulation_end_time = 0.5
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_3d.jl"),
              tspan=(0.0, simulation_end_time),
              particle_spacing_factor=particle_spacing_factor,
              extra_callback=extra_callback, saving_callback=saving_callback, v_max=v_max,
              inlet_reference_pressure=dynamic_pressure_drop,
              outlet_reference_pressure=dynamic_pressure_drop)
