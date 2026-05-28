using TrixiParticles
using CSV
using DataFrames
using Plots
using Statistics

# In Tafuni et al. (2018), the resolution is `0.01` (5M particles).
# `resolution_factor = 0.02` results in 1.3M particles.
resolution_factor = 0.05
cylinder_diameter = 0.1
prescribed_velocity = 1.0

output_directory = "out"

# ======================================================================================
# ==== Read results
data = CSV.read(joinpath(output_directory, "resulting_force.csv"), DataFrame)

step = 1
times = data[!, "time"][1:step:end]

f_lift = data[!, "f_l_fluid_1"][1:step:end]
f_drag = data[!, "f_d_fluid_1"][1:step:end]

# ======================================================================================
# ==== Compute strouhal number
t_start = 6.0
start_index = findfirst(t -> t >= t_start, times)
isnothing(start_index) && error("t_start = $t_start is larger than the maximum time in the data.")

times_cut = times[start_index:end]
length(times_cut) >= 4 || error("need at least four force samples after t_start = $t_start")
dt = times_cut[2] - times_cut[1]

f_lift_cut = f_lift[start_index:end]
f_drag_cut = f_drag[start_index:end]

# Compute the frequency bins for the discrete Fourier transform.
# For N time samples with uniform time steps dt, the corresponding frequencies are:
# f_k = k / (N * dt), where k = 0, 1, ..., N-1.
# This gives the frequency bins in Hz, matching the order of the spectrum below.
N = length(f_lift_cut)
sample_indices = 0:(N - 1)
frequencies = sample_indices / (N * dt)
half_spectrum = 1:div(N, 2)
frequencies_half = frequencies[half_spectrum]

lift_signal = f_lift_cut .- mean(f_lift_cut)
# Avoid an extra FFT dependency for this lightweight validation plot.
spectrum_half = [abs(sum(lift_signal .*
                         cis.(-2pi * (frequency_index - 1) / N .* sample_indices)))
                 for frequency_index in half_spectrum]

# For real-valued signals, the Fourier spectrum is symmetric.
# Only the first half (up to the Nyquist frequency) contains unique, physically meaningful frequency components.
# We therefore analyze only the first N/2 values of the frequency spectrum.
f_dominant = frequencies_half[argmax(spectrum_half)]

# Verify whether the dominant frequency is indeed unique.
# In theory, for a purely harmonic oscillation, the spectrum should exhibit only a single dominant frequency component.
delta = 2 * (frequencies_half[2] - frequencies_half[1])
frequency_band = abs.(frequencies_half .- f_dominant) .< delta

strouhal_number = f_dominant * cylinder_diameter / prescribed_velocity

# ======================================================================================
# ==== Validate the frequency spectrum
integral_total = sum(spectrum_half)
integral_peak = sum(spectrum_half[frequency_band])
dominant_band_fraction = integral_total > 0 ? integral_peak / integral_total : NaN

@info "Strouhal number" round(strouhal_number, digits=3)
@info "Fraction of the dominant frequency band in the total spectrum" round(dominant_band_fraction,
                                                                            digits=3)
@info "C_L_max for the periodic shedding" round(maximum(f_lift_cut), digits=3)
@info "C_D_max for the periodic shedding" round(maximum(f_drag_cut), digits=3)

dp = round(Int, 1 / resolution_factor)
plot_title = "Drag and lift force coefficients (Δx = d/$(dp))"
pC = plot(times, f_lift, ylims=(-1, 3), xlims=(0, 20), label="C_L", color=:red, linewidth=2)
plot!(pC, times, f_drag, ylims=(-1, 3), xlims=(0, 20), label="C_D", color=:blue,
      linewidth=2, title=plot_title, xlabel="t (s)")
plot!(pC, top_margin=2Plots.mm)

pS = plot(frequencies_half, spectrum_half, xlabel="Frequency (Hz)", size=(400, 200),
          ylabel="Amplitude",
          title="Frequency Spectrum (St = $(round(strouhal_number, digits=4)))",
          label=nothing, linewidth=2)
plot!(pS, top_margin=2Plots.mm)

p = plot(pC, pS, layout=@layout([a; b{0.3h}]), size=(800, 800))
plot!(p, right_margin=5Plots.mm)
