using TrixiParticles
using FFTW
using CSV, DataFrames
using Test

# Results in [90k particles, 340k particles, 1.2M particles, 5M particles]
# In the Tafuni et al. (2018), the resolution is `0.01` (5M particles).
resolution_factor = 0.04 # [0.08, 0.04, 0.02, 0.01]

reynolds_number = 200

output_directory = joinpath(validation_dir(), "vortex_street_2d",
                            "out_dp_$(resolution_factor)D_Re_$reynolds_number")

# ==========================================================================================
# ==== Postprocessing
trixi_include(joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              reynolds_number=reynolds_number, saving_callback=nothing,
              output_directory=output_directory, factor_d=resolution_factor, sol=nothing)

circle = SphereShape(particle_spacing, (cylinder_diameter + particle_spacing) / 2,
                     cylinder_center, fluid_density, n_layers=1,
                     sphere_type=RoundSphere())

# Points for pressure interpolation, located at the wall interface
const data_points = copy(circle.coordinates)
const center = SVector(cylinder_center)

calculate_lift_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function calculate_lift_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                              v_ode, u_ode, semi, t)
    force = zero(SVector{ndims(system), eltype(system)})

    values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                clip_negative_pressure=false)
    pressure = Array(values.pressure)

    for i in axes(data_points, 2)
        point = TrixiParticles.current_coords(data_points, system, i)

        # F = ∑ -p_i * A_i * n_i
        force -= pressure[i] * particle_spacing .*
                 TrixiParticles.normalize(point - center)
    end

    return 2 * force[2] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

calculate_drag_force(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function calculate_drag_force(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode,
                              v_ode,
                              u_ode, semi, t)
    force = zero(SVector{ndims(system), eltype(system)})

    values = interpolate_points(data_points, semi, system, v_ode, u_ode; cut_off_bnd=false,
                                clip_negative_pressure=false)
    pressure = Array(values.pressure)

    for i in axes(data_points, 2)
        point = TrixiParticles.current_coords(data_points, system, i)

        # F = ∑ -p_i * A_i * n_i
        force -= pressure[i] * particle_spacing .*
                 TrixiParticles.normalize(point - center)
    end

    return 2 * force[1] / (fluid_density * prescribed_velocity^2 * cylinder_diameter)
end

pp_callback = PostprocessCallback(; dt=0.02,
                                  f_l=calculate_lift_force, f_d=calculate_drag_force,
                                  output_directory, filename="resulting_force",
                                  write_csv=true, write_file_interval=10)

# ======================================================================================
# ==== Run the simulation
trixi_include(joinpath(examples_dir(), "fluid", "vortex_street_2d.jl"),
              parallelization_backend=PolyesterBackend(), reynolds_number=reynolds_number,
              saving_callback=nothing, output_directory=output_directory,
              factor_d=resolution_factor, tspan=(0.0, 20.0), extra_callback=pp_callback)

# ======================================================================================
# ==== Read results and compute the Strouhal number
data = CSV.read(joinpath(output_directory, "resulting_force.csv"), DataFrame)

time = data[!, "time"]
t_start = 6.0
start_index = findfirst(t -> t ≥ t_start, time)
times_cut = time[start_index:end]
dt = times_cut[2] - times_cut[1]

f_lift = data[!, "f_l_fluid_1"][start_index:end]

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

# Verify whether the dominant frequency is indeed unique.
# In theory, for a purely harmonic oscillation, the spectrum should exhibit only a single dominant frequency component.
delta = 2 * (frequencies[2] - frequencies[1])
frequency_band = (abs.(frequencies[1:div(N, 2)] .- f_dominant) .< delta)

# ======================================================================================
# ==== Save the strouhal numbers
strouhal_number = f_dominant * cylinder_diameter / prescribed_velocity

df = DataFrame(Resolution=resolution_factor, t_max=last(tspan),
               frequency=f_dominant, StrouhalNumber=strouhal_number)

CSV.write(joinpath(output_directory, "strouhal_number.csv"), df)

# ======================================================================================
# ==== Validate the frequency spectrum
spectrum_half = spectrum[1:div(N, 2)]
integral_total = sum(spectrum_half)
integral_peak = sum(spectrum_half[frequency_band])

# TODO: 0.4 is sufficient? Check for higher resolution.
@test 0.4 < integral_peak / integral_total
