using TrixiParticles
using Bessels

include(joinpath("..", "poiseuille_flow_2d", "poiseuille_flow_util.jl"))

# Reference files are available for 10, 20, 30, 40 and 50.
# Use `particle_spacing_factor = 50` to match the resolution used by Zhang et al. (2025).
particle_spacing_factor = 50

# Set to true to plot the results of a validation run in the `out` folder
# instead of the reference file.
use_sim_results = false

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "poiseuille_flow_3d.jl"),
              particle_spacing_factor=particle_spacing_factor, sol=nothing)

# First five roots of the Bessel function of the first kind, J₀(x),
# required for the analytical solution of the transient velocity profile in axisymmetric pipe flow.
roots_J_0 = [2.4048255577, 5.5200781103, 8.6537279129, 11.7915344391, 14.9309177086]

# Analytical velocity evolution given in eq. 18 (Zhang et al., 2025)
function hagen_poiseuille_velocity(r, t)
    # Base profile (stationary part)
    base_profile = (imposed_pressure_drop / (4 * dynamic_viscosity * channel_length)) *
                   (channel_radius^2 - r^2)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in eachindex(roots_J_0)
        alpha = roots_J_0[n]
        J_1 = besselj(1, alpha)
        J_2 = besselj(2, alpha)

        coefficient = (imposed_pressure_drop * channel_radius^2 * J_2) /
                      (dynamic_viscosity * channel_length * alpha^2 * J_1^2)

        exp_term = exp(-t * (dynamic_viscosity * alpha^2) /
                       (fluid_density * channel_radius^2))

        transient_sum += coefficient * besselj0(r * alpha / channel_radius) * exp_term
    end

    # Total velocity
    v_x = base_profile - transient_sum

    return v_x
end

input_file = use_sim_results ?
             joinpath("out",
                      "validation_run_poiseuille_flow_3d_$particle_spacing_factor.json") :
             joinpath(validation_dir(), "poiseuille_flow_3d",
                      "validation_reference_$particle_spacing_factor.json")
times_ref = [0.03, 0.05, 0.07, 0.14, 0.3, 1.0]
v_x_vector = load_velocity_profiles(input_file, times_ref)
positions = range(-channel_radius, channel_radius, length=length(first(v_x_vector)))
data_range = 10:90

rmsep_run = rmsep(v_x_vector, times_ref, positions, data_range, hagen_poiseuille_velocity)

# RMSEP error (%) received by Zhang et al. (2025)
rmsep_reference = [2.97, 1.88, 1.61, 1.5, 0.74, 0.89]

p_rmsep = plot_rmsep(times_ref, rmsep_run, rmsep_reference,
                     title="Poiseuille Flow 3D", xlims=(0, 1.1))
display(p_rmsep)

p = plot_velocity_profiles(positions, v_x_vector, times_ref, hagen_poiseuille_velocity,
                           title="Poiseuille Flow 3D", ylims=(-0.001, 0.005))
display(p)
