using TrixiParticles
using FFTW
using CSV, DataFrames

strouhal_numbers = Float64[]

# Results in [90k particles, 220k particles, 1.2M particles, 5M particles]
resolution_factors = [0.08, 0.05, 0.02, 0.01]

for resolution_factor in resolution_factors
    # ======================================================================================
    # ==== Run the simulation
    trixi_include(joinpath(validation_dir(), "vortex_street_2d", "vortex_street_2d.jl"),
                  factor_d=resolution_factor, saving_callback=nothing, tspan=(0.0, 50.0))

    # ======================================================================================
    # ==== Read results and compute the Strouhal number
    data = CSV.read(joinpath(output_directory, "resulting_force.csv"), DataFrame)

    time = data[!, "time"]
    dt = time[2] - time[1]

    f_lift = data[!, "f_l_fluid_1"]

    # Compute the frequency for the FFT.
    # For N time samples with uniform time steps dt, the corresponding frequencies are:
    # f_k = k / (N * dt), where k = 0, 1, ..., N-1.
    # This gives the frequency bins in Hz, matching the order of FFT.
    N = length(f_lift)
    frequencies = (0:(N - 1)) / (N * dt)

    spectrum = abs.(fft(f_lift))

    # For real-valued signals, the FFT output is symmetric.
    # Only the first half (up to the Nyquist frequency) contains unique, physically meaningful frequency components.
    # We therefore analyze only the first N/2 values of the frequency spectrum.
    f_dominant = frequencies[argmax(spectrum[1:div(N, 2)])]

    strouhal_number = f_dominant * cylinder_diameter / prescribed_velocity

    push!(strouhal_numbers, strouhal_number)
end

# ======================================================================================
# ==== Save the strouhal numbers
df = DataFrame(Resolution=resolution_factors, StrouhalNumber=strouhal_numbers)

CSV.write(joipnath(validation_dir(), "strouhal_numbers.csv"), df)
