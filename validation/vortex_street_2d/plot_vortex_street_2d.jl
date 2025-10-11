using TrixiParticles
using CSV, DataFrames, Plots
using FFTW

# Results in [90k particles, 340k particles, 1.2M particles, 5M particles]
# In the Tafuni et al. (2018), the resolution is `0.01` (5M particles).
resolution_factor = 0.04 # [0.08, 0.04, 0.02, 0.01]

cylinder_diameter = 0.1
prescribed_velocity = 1.0

reynolds_number = 200

# open_boundary_model = BoundaryModelMirroringTafuni(; mirror_method=ZerothOrderMirroring())
open_boundary_model = BoundaryModelDynamicalPressureZhang()

model = nameof(typeof(open_boundary_model))
output_directory = joinpath(validation_dir(), "vortex_street_2d",
                            "$(model)_dp_$(resolution_factor)D_Re_$reynolds_number")

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
start_index = findfirst(t -> t ≥ t_start, times)
times_cut = times[start_index:end]
dt = times_cut[2] - times_cut[1]

f_lift_cut = f_lift[start_index:end]

# Compute the frequency for the FFT.
# For N time samples with uniform time steps dt, the corresponding frequencies are:
# f_k = k / (N * dt), where k = 0, 1, ..., N-1.
# This gives the frequency bins in Hz, matching the order of FFT.
N = length(f_lift_cut)
frequencies = (0:(N - 1)) / (N * dt)

spectrum = abs.(fft(f_lift_cut))
spectrum_half = spectrum[1:div(N, 2)]

# For real-valued signals, the FFT output is symmetric.
# Only the first half (up to the Nyquist frequency) contains unique, physically meaningful frequency components.
# We therefore analyze only the first N/2 values of the frequency spectrum.
f_dominant = frequencies[argmax(spectrum_half)]

# Verify whether the dominant frequency is indeed unique.
# In theory, for a purely harmonic oscillation, the spectrum should exhibit only a single dominant frequency component.
delta = 2 * (frequencies[2] - frequencies[1])
frequency_band = (abs.(frequencies[1:div(N, 2)] .- f_dominant) .< delta)

strouhal_number = f_dominant * cylinder_diameter / prescribed_velocity

# ======================================================================================
# ==== Validate the frequency spectrum
integral_total = sum(spectrum_half)
integral_peak = sum(spectrum_half[frequency_band])

@info "Strouhal number" strouhal_number
@info "Fraction of the dominant frequency band in the total spectrum" integral_peak /
                                                                      integral_total
@info "C_L_max for the unsteady state" maximum(f_lift)

dp = round(Int, 1 / resolution_factor)
plot_title = "$(model), Δx = d/$(dp), St = $(round(strouhal_number, digits=4))"
pC = plot(times, f_lift, ylims=(-1, 3), xlims=(0, 20), label="C_L", color=:red, linewidth=2)
plot!(pC, times, f_drag, ylims=(-1, 3), xlims=(0, 20), label="C_D", color=:blue,
      linewidth=2, title=plot_title, xlabel="t (s)")
plot!(pC, top_margin=2Plots.mm)

pS = plot(frequencies[1:div(N, 2)], spectrum_half, xlabel="Frequency (Hz)", size=(400, 200),
          ylabel="Amplitude", title="Frequency Spectrum", label=nothing, linewidth=2)
plot!(pC, top_margin=2Plots.mm)

p = plot(pC, pS, layout=@layout([a; b{0.3h}]), size=(800, 800))
plot!(p, right_margin=5Plots.mm)
