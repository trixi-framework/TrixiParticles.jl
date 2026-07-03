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
period = 4.567375
n_periods = 12
tspan = (0.0, n_periods * period)

fluid_density = 1000.0
sound_speed = 10.0

# The compressible energy diagnostic below uses the closed form for the linear equation
# of state used in the paper.
state_equation = StateEquationCole(; sound_speed, exponent=1,
                                   reference_density=fluid_density)

# The paper's energy balance is for the inviscid oscillating drop with density diffusion.
viscosity = nothing

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "oscillating_drop_2d.jl");
              fluid_particle_spacing, tspan, fluid_density, sound_speed,
              state_equation, viscosity, #density_diffusion=nothing,
              sol=nothing, error_A=nothing)

formatted_spacing = replace(@sprintf("%.4f", fluid_particle_spacing), "." => "p")
filename = "validation_result_oscillating_drop_2d_dx_$formatted_spacing"

q_delta = DeltaSPHHeat()

postprocess_callback = PostprocessCallback(; output_directory="out",
                                           filename,
                                           write_file_interval=1000,
                                           interval=1,
                                           kinetic_energy,
                                           potential_energy,
                                           compressible_energy,
                                           delta_sph_diffusive_power,
                                           q_delta)

info_callback = InfoCallback(interval=500)
callbacks = CallbackSet(info_callback, postprocess_callback)

sol = solve(ode, RDPK3SpFSAL49(),
            abstol=1e-7,
            reltol=1e-4,
            save_everystep=false, callback=callbacks)

println("Oscillating drop energy validation written to out/$filename.json")
