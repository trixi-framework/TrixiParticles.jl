using TrixiParticles

include("poiseuille_flow_util.jl")

# Reference files are available for 30 and 50.
# Use `particle_spacing_factor = 50` to match the resolution used by Zhang et al. (2025).
particle_spacing_factor = 50

# Set to true to plot the results of a validation run in the `out` folder
# instead of the reference file.
use_sim_results = false

# Import variables into scope
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"),
              particle_spacing_factor=particle_spacing_factor, sol=nothing)

# Analytical velocity evolution given in eq. 16 (Zhang et al., 2025)
function poiseuille_velocity(y, t)

    # Base profile (stationary part)
    base_profile = (imposed_pressure_drop / (2 * dynamic_viscosity * channel_length)) * y *
                   (y - channel_height)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 0:10  # Limit to 11 terms for convergence
        coefficient = (4 * imposed_pressure_drop * channel_height^2) /
                      (dynamic_viscosity * channel_length * pi^3 * (2 * n + 1)^3)

        sine_term = sin(pi * y * (2 * n + 1) / channel_height)

        exp_term = exp(-((2 * n + 1)^2 * pi^2 * dynamic_viscosity * t) /
                       (fluid_density * channel_height^2))

        transient_sum += coefficient * sine_term * exp_term
    end

    # Total velocity
    v_x = base_profile + transient_sum

    return v_x
end

# Load results
input_file = use_sim_results ?
             joinpath("out",
                      "validation_run_poiseuille_flow_2d_$particle_spacing_factor.json") :
             joinpath(validation_dir(), "poiseuille_flow_2d",
                      "validation_reference_$particle_spacing_factor.json")
times_ref = [0.1, 0.3, 0.6, 0.9, 2.0]
v_x_vector = load_velocity_profiles(input_file, times_ref)
positions = range(0, channel_height, length=length(first(v_x_vector)))
data_range = 2:98

v_analytical(y, t) = -poiseuille_velocity(y, t)
rmsep_run = rmsep(v_x_vector, times_ref, positions, data_range, v_analytical)

# RMSEP (%) received by Zhang et al. (2025)
rmsep_reference = [1.81, 0.95, 0.67, 0.86, 1.22]

p_rmsep = plot_rmsep(times_ref, rmsep_run, rmsep_reference,
                     title="Poiseuille Flow 2D", xlims=(0, 2.05))
display(p_rmsep)

p = plot_velocity_profiles(positions, v_x_vector, times_ref, v_analytical,
                           title="Poiseuille Flow 2D", ylims=(-0.002, 0.014))
display(p)
