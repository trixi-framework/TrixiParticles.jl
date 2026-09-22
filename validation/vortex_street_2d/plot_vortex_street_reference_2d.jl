using TrixiParticles
using TrixiParticles.JSON
using Plots
using Statistics

# In Tafuni et al. (2018), the resolution is `0.01` (5M particles).
# `resolution_factor = 0.02` results in 1.3M particles.
resolution_factor = 0.05
cylinder_diameter = 0.1
prescribed_velocity = 1.0

directory = joinpath(validation_dir(), "vortex_street_2d")

# Particle spacing relative to the cylinder diameter, e.g. `20` for a spacing of `d/20`
dp = round(Int, 1 / resolution_factor)

# ======================================================================================
# ==== Read results
data = JSON.parsefile(joinpath(directory, "resulting_force_dp$dp.json"))

times = Float64.(data["lift_force_fluid_1"]["time"])

f_lift = Float64.(data["lift_force_fluid_1"]["values"])
f_drag = Float64.(data["drag_force_fluid_1"]["values"])
# Transverse velocity measured by the sensor in the wake of the cylinder
v_y = Float64.(data["wake_velocity_y_fluid_1"]["values"])

# ======================================================================================
# ==== Compute the frequency spectrum
# The flow is still developing at the beginning of the simulation, so only analyze the
# signals after the vortex shedding has become periodic.
t_start = 6.0
start_index = findfirst(t -> t >= t_start, times)
isnothing(start_index) &&
    error("t_start = $t_start is larger than the maximum time in the data.")

times_cut = times[start_index:end]
length(times_cut) >= 4 || error("need at least four samples after t_start = $t_start")
dt = times_cut[2] - times_cut[1]

f_lift_cut = f_lift[start_index:end]
f_drag_cut = f_drag[start_index:end]
v_y_cut = v_y[start_index:end]

# Compute the frequency bins for the discrete Fourier transform.
# For N time samples with uniform time steps dt, the corresponding frequencies are:
# f_k = k / (N * dt), where k = 0, 1, ..., N-1.
# This gives the frequency bins in Hz, matching the order of the spectrum below.
#
# For real-valued signals, the Fourier spectrum is symmetric.
# Only the first half (up to the Nyquist frequency) contains unique,
# physically meaningful frequency components.
# We therefore analyze only the first N/2 values of the frequency spectrum.
function frequency_spectrum(signal, dt)
    N = length(signal)
    sample_indices = 0:(N - 1)
    frequencies = sample_indices / (N * dt)
    half_spectrum = 1:div(N, 2)

    signal_centered = signal .- mean(signal)
    # Avoid an extra FFT dependency for this lightweight validation plot.
    spectrum = [abs(sum(signal_centered .*
                        cis.(-2pi * (frequency_index - 1) / N .* sample_indices)))
                for frequency_index in half_spectrum]

    return frequencies[half_spectrum], spectrum
end

# Return the dominant frequency and the fraction of the total spectrum that is contained
# in the frequency band around the dominant frequency.
# In theory, for a purely harmonic oscillation, the spectrum should exhibit only a single
# dominant frequency component, so this fraction indicates how clean the oscillation is.
function dominant_frequency(frequencies, spectrum)
    f_dominant = frequencies[argmax(spectrum)]

    delta = 2 * (frequencies[2] - frequencies[1])
    frequency_band = abs.(frequencies .- f_dominant) .< delta

    integral_total = sum(spectrum)
    integral_peak = sum(spectrum[frequency_band])
    band_fraction = integral_total > 0 ? integral_peak / integral_total : NaN

    return f_dominant, band_fraction
end

# The force coefficients are computed from the pressure at the cylinder surface, which is
# very noisy at low resolutions. The transverse velocity in the wake of the cylinder yields
# a much cleaner signal, so the Strouhal number is computed from `v_y`.
frequencies_v_y, spectrum_v_y = frequency_spectrum(v_y_cut, dt)
f_dominant_v_y, band_fraction_v_y = dominant_frequency(frequencies_v_y, spectrum_v_y)
strouhal_number = f_dominant_v_y * cylinder_diameter / prescribed_velocity

# For comparison, also compute the Strouhal number from the lift coefficient.
# This only yields a meaningful value at sufficiently high resolutions.
frequencies_lift, spectrum_lift = frequency_spectrum(f_lift_cut, dt)
f_dominant_lift, band_fraction_lift = dominant_frequency(frequencies_lift, spectrum_lift)
strouhal_number_lift = f_dominant_lift * cylinder_diameter / prescribed_velocity

@info "Strouhal number (from the wake velocity v_y)" round(strouhal_number, digits=3)
@info "Dominant frequency band fraction of the v_y spectrum" round(band_fraction_v_y,
                                                                   digits=3)
@info "Strouhal number (from the lift coefficient)" round(strouhal_number_lift, digits=3)
@info "Dominant frequency band fraction of the C_L spectrum" round(band_fraction_lift,
                                                                   digits=3)
@info "C_L_max for the periodic shedding" round(maximum(f_lift_cut), digits=3)
@info "C_D_max for the periodic shedding" round(maximum(f_drag_cut), digits=3)

# ======================================================================================
# ==== Plot the results
plot_title = "Drag and lift force coefficients (Δx = d/$(dp))"
time_limits = (0, maximum(times))
pC = plot(times, f_lift, ylims=(-1, 3), xlims=time_limits, label="C_L", color=:red,
          linewidth=2)
plot!(pC, times, f_drag, ylims=(-1, 3), xlims=time_limits, label="C_D", color=:blue,
      linewidth=2, title=plot_title, xlabel="t (s)")
plot!(pC, top_margin=2Plots.mm)

pV = plot(times, v_y, xlims=time_limits, label="v_y", color=:green, linewidth=2,
          title="Transverse velocity in the wake of the cylinder", xlabel="t (s)",
          ylabel="v_y (m/s)")
plot!(pV, top_margin=2Plots.mm)

pS = plot(frequencies_v_y, spectrum_v_y, xlabel="Frequency (Hz)", size=(400, 200),
          ylabel="Amplitude",
          title="Frequency Spectrum of v_y (St = $(round(strouhal_number, digits=4)))",
          label=nothing, linewidth=2)
plot!(pS, top_margin=2Plots.mm)

p = plot(pC, pV, pS, layout=@layout([a; b; c{0.25h}]), size=(800, 1000), dpi=600)
plot!(p, right_margin=5Plots.mm)
