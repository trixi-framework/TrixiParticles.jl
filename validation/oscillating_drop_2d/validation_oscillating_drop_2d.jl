# ==========================================================================================
# 2D Oscillating Drop Energy Validation
#
# Based on:
#   M. Antuono, S. Marrone, A. Colagrossi, B. Bouscasse.
#   "Energy balance in the δ-SPH scheme"
#   Computer Methods in Applied Mechanics and Engineering, 289 (2015), pp. 209-226.
#   https://doi.org/10.1016/j.cma.2015.02.004
#
# The validation extracts the energy components plotted in Fig. 3 of the paper. The
# default resolution is intentionally lower than the paper's R / dx = 200 setup.
# ==========================================================================================

include("../validation_util.jl")

using TrixiParticles
using OrdinaryDiffEqLowStorageRK
using Printf

include("energy_quantities.jl")

# ==========================================================================================
# ==== Resolution
fluid_particle_spacing = 0.05

# ==========================================================================================
# ==== Experiment Setup
n_periods = 12
omega = 1.0

# `VoxelSphere` matches the initial drop in mechanical energy seen in the paper,
# while `RoundSphere` starts with an initial configuration close to equilibrium,
# resulting in a cleaner energy plot.
sphere_type = VoxelSphere()

# The paper's energy balance is for the inviscid oscillating drop with density diffusion.
viscosity = nothing

formatted_spacing = replace(@sprintf("%.4f", fluid_particle_spacing), "." => "p")
filename = "validation_result_oscillating_drop_2d_dx_$formatted_spacing"

q_delta = DeltaSPHHeat()

# Note that `interval` also controls the time step size for the integration of Q_δ,
# not just the output frequency.
extra_callback = PostprocessCallback(; filename, output_directory="out",
                                     interval=10, write_file_interval=500,
                                     kinetic_energy,
                                     potential_energy=potential_energy(omega),
                                     compressible_energy, q_delta)

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "oscillating_drop_2d.jl");
              fluid_particle_spacing, n_periods, omega, viscosity, extra_callback,
              info_callback = InfoCallback(interval=500), saving_callback=nothing,
              parallelization_backend=PolyesterBackend())

println("Oscillating drop energy validation written to out/$filename.json")
