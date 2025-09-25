using TrixiParticles
using Plots, FFTW, Statistics
using OrdinaryDiffEq

struct FluidSystemMock2 <: TrixiParticles.AbstractFluidSystem{2}
    pressure_acceleration_formulation::Nothing
    density_diffusion::Nothing
end
TrixiParticles.initial_smoothing_length(system::FluidSystemMock2) = 1.0
TrixiParticles.nparticles(system::FluidSystemMock2) = 1
TrixiParticles.system_smoothing_kernel(system::FluidSystemMock2) = nothing

# R_1: peripheral resistance
# R_2: characteristic resistance
# C: compliance

# Two-element Windkessel model (RC-WK)
# q(t) = p(t) / R + C * dp(t) / dt

# Three-element Windkessel model (RCR-WK)
# dp(t) / dt + p(t) / (R_1 * C) = R_2 * dq(t) / dt + (R_1 + R_2) / (R_1 * C) * q(t)

# Measurements of aortic presure and flow in the ascending ferret aorta (Gasser, 2021):
Δt = 0.0125
times_exp = collect(0:Δt:0.375)
SI = true
factor_flow = SI ? 10e-6 : 1
factor_pressure = SI ? 133.322387415 : 1

flow_rates_exp = [0.00, 0.00, 0.04, 0.23, 0.48, 3.00, 4.61, 4.50, 3.96, 3.20, 2.10, 0.00,
    -0.32, -0.08, 0.14, 0.15, 0.16, 0.20, 0.30, 0.20, 0.16, 0.13, 0.10, 0.08, 0.07, 0.06,
    0.06, 0.04, 0.02, 0.00, 0.00] .* factor_flow

pressures_exp = [78, 78, 77, 77, 78, 95, 97, 97, 96, 93, 90, 81, 83, 85, 87, 89,
    90, 90, 89, 87, 86, 84, 84, 83, 82, 81, 81, 80, 80, 79, 78.0] .* factor_pressure
# pressures_exp .-= minimum(pressures_exp)

function compute_fourier_coefficients_fft(signal_data, T_period, M_harmonics=10)
    N = length(signal_data)

    # FFT berechnen
    fft_result = fft(signal_data)

    # Complex coefficients extrahieren
    c_coeffs = zeros(ComplexF64, M_harmonics + 1)

    # DC component (k=0)
    c_coeffs[1] = fft_result[1] / N

    # Harmonics (k=1 to M)
    for k in 1:M_harmonics
        if k + 1 <= N
            c_coeffs[k + 1] = fft_result[k + 1] / N
        end
    end

    omega_1 = 2pi / T_period

    return c_coeffs, omega_1
end

function fourier_signal_from_fft(t, c_coeffs, omega_1)
    # Rekonstruiert Signal aus FFT-Koeffizienten
    # ỹ(t) = Re[∑(k=0 to M) c_k exp(iωₖt)]
    M = length(c_coeffs) - 1
    signal = real(c_coeffs[1])  # DC component

    for k in 1:M
        omega_k = k * omega_1
        signal += 2 * real(c_coeffs[k + 1] * exp(1im * omega_k * t))
    end

    return signal
end

function pulsatile_flow_table(t)
    # Fourier coefficients from Table 2.5
    # Period T = 0.375 s
    T = 0.375
    omega_1 = 2π / T

    # Coefficients [ml s^-1]
    c0 = 0.786333
    c1 = 0.0276938 - 0.646413im
    c2 = -0.555816 - 0.112961im
    c3 = -0.0896779 + 0.408525im
    c4 = 0.216635 + 0.0557698im

    # Convert to m³/s if SI units are used
    factor = 1

    # Calculate signal using complex exponentials
    # q(t) = Re[c0 + 2*Re(c1*exp(iω₁t)) + 2*Re(c2*exp(i2ω₁t)) + ...]
    signal = c0
    signal += 2 * real(c1 * exp(1im * omega_1 * t))
    signal += 2 * real(c2 * exp(1im * 2 * omega_1 * t))
    signal += 2 * real(c3 * exp(1im * 3 * omega_1 * t))
    signal += 2 * real(c4 * exp(1im * 4 * omega_1 * t))

    return signal * factor
end

T = 0.375
c_flow_fft, omega_flow = compute_fourier_coefficients_fft(flow_rates_exp, T, 10)
c_pressure_fft, omega_pressure = compute_fourier_coefficients_fft(pressures_exp, T, 10)
pulsatile_flow_fft(t) = fourier_signal_from_fft(t, c_flow_fft, omega_flow)
pulsatile_pressure_fft(t) = fourier_signal_from_fft(t, c_pressure_fft, omega_pressure)

p1 = plot(pulsatile_flow_fft, xlims=(0, 0.375), color=:red, linewidth=2,
          label="fourier-approx")
plot!(p1, times_exp, flow_rates_exp, color=:black, linestyle=:dash, linewidth=2,
      label="experiment")
p2 = plot(pulsatile_pressure_fft, xlims=(0, 0.375), color=:red, linewidth=2,
          label="fourier-approx")
plot!(p2, times_exp, pressures_exp, color=:black, linestyle=:dash, linewidth=2,
      label="experiment")

function pressure_RC_ode!(dp, p, params, t)
    (; R, C, q_func) = params

    dp[1] = (q_func(t) - p[1] / R) / C
end

# dp/dt = - p * 1 / (R1 * C)  + R2 * dq/dt + q * (R1+ R2) / (R1 * C)
function pressure_RCR_ode!(dp, p, params, t)
    (; R_1, R_2, C, q_func, dt) = params

    dq_dt = (q_func(t) - q_func(t - dt)) / dt

    dp[1] = -p[1] / (R_1 * C) + R_2 * dq_dt + q_func(t) * (R_1 + R_2) / (R_1 * C)
end
function simulate_rcr(R1, R2, C, tspan, dt, p_0, func)
    pressure_model = RCRWindkesselModel(; peripheral_resistance=R2,
                                        compliance=C,
                                        characteristic_resistance=R1)
    # Define a boundary zone with height=1.0 to ensure a unit volume,
    # so velocity directly corresponds to flow rate.
    reference_velocity(pos, t) = SVector(func(t), 0.0)
    boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                                 particle_spacing=0.1, face_normal=(1.0, 0.0),
                                 density=1000.0, reference_velocity, pressure_model,
                                 open_boundary_layers=1, rest_pressure=p_0)

    system = OpenBoundarySystem(boundary_zone; buffer_size=0, boundary_model=nothing,
                                fluid_system=FluidSystemMock2(nothing, nothing))

    u = system.initial_condition.coordinates
    v = system.initial_condition.velocity

    times = collect(tspan[1]:dt:tspan[2])
    p_calculated = zero(times)
    q_calculated = zero(times)
    for (i, t) in enumerate(times)
        v[1, :] .= func(t)
        TrixiParticles.calculate_flow_rate_and_pressure!(system, v, u, dt)

        p_calculated[i] = system.pressure_model_values[1].pressure[]
        q_calculated[i] = system.pressure_model_values[1].flow_rate[]
    end

    return p_calculated, q_calculated, times
end

params_RC = (R=108.41, C=9.8143e-3, q_func=pulsatile_flow_table)
params_RCR = (R_1=104.04, R_2=4.37, C=84.787e-3, q_func=pulsatile_flow_table, dt=Δt)

p0 = [pressures_exp[1]]
tspan = (0.0, 20T)

sol_RC = solve(ODEProblem(pressure_RC_ode!, p0, tspan, params_RC), RK4(), adaptive=false,
               dt=Δt)
sol_RCR = solve(ODEProblem(pressure_RCR_ode!, p0, tspan, params_RCR), RK4(), adaptive=false,
                dt=Δt)

pressures, flow_rates,
times = simulate_rcr(104.04, 4.37, 84.787e-3, tspan, Δt/2,
                     pressures_exp[1], pulsatile_flow_table)

x_lims = (10T, 11T)
p = plot(sol_RC, label="RC-WK (time integrator: RK4)", color=:blue, xlims=x_lims)
plot!(p, sol_RCR, label="RCR-WK (time integrator: RK4)", color=:red, xlims=x_lims)
plot!(p, times, pressures, color=:black, xlims=x_lims, label="TP")
