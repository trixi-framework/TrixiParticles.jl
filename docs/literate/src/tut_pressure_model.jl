using TrixiParticles
using Plots, DSP, LaTeXStrings

function pulsatile_velocity_sin(t)
    amplitude = 1
    frequency = 3 / T

    t_periodic = mod(t, T)
    t_periodic > 1 / frequency && return 0.0

    return amplitude * sin(pi * frequency * t_periodic)^4
end
function pulsatile_velocity_fourier(t)
    # Fourier coefficients from empirical data (Table III)
    a_0 = 0.3782
    omega = 10 # 8.302  # Angular frequency

    # Eq. 21: V_inlet(t) = a₀ + Σ aₙ cos(nωt) + Σ bₙ sin(nωt)
    v_inlet = a_0

    # Cosine and sine terms with coefficients directly embedded
    v_inlet += (-0.1812) * cos(1 * omega * t) + (-0.07725) * sin(1 * omega * t)
    v_inlet += 0.1276 * cos(2 * omega * t) + 0.01466 * sin(2 * omega * t)
    v_inlet += (-0.08981) * cos(3 * omega * t) + 0.04295 * sin(3 * omega * t)
    v_inlet += 0.04347 * cos(4 * omega * t) + (-0.06679) * sin(4 * omega * t)
    v_inlet += (-0.05412) * cos(5 * omega * t) + 0.05679 * sin(5 * omega * t)
    v_inlet += 0.02642 * cos(6 * omega * t) + (-0.01878) * sin(6 * omega * t)
    v_inlet += 0.008946 * cos(7 * omega * t) + 0.01869 * sin(7 * omega * t)
    v_inlet += (-0.009005) * cos(8 * omega * t) + (-0.01888) * sin(8 * omega * t)

    return v_inlet
end

function simulate_rcr(R1, R2, C, tspan, dt, func)
    pressure_model = RCRWindkesselModel(; peripheral_resistance=R2,
                                        compliance=C,
                                        characteristic_resistance=R1)

    reference_velocity(pos, t) = SVector(func(t), 0.0)
    particle_spacing = 0.2
    boundary_zone = BoundaryZone(; boundary_face=([0.0, 0.0], [0.0, 1.0]),
                                 particle_spacing, face_normal=(1.0, 0.0), density=1000.0,
                                 reference_velocity, pressure_model, open_boundary_layers=2)

    trixi_include(joinpath(examples_dir(), "fluid", "poiseuille_flow_2d.jl"), sol=nothing,
                  domain_size=(1, 1), inflow=boundary_zone, outflow=nothing, tspan=tspan,
                  n_buffer_particles=0, open_boundary_layers=2, boundary_system=nothing,
                  particle_spacing=particle_spacing, nhs=nothing)

    times = collect(tspan[1]:dt:tspan[2])

    pressures = zero(times)
    flow_rates = zero(times)

    v_ode, u_ode = ode.u0.x
    for (i, t) in enumerate(times)
        TrixiParticles.update_open_boundary_eachstep!(open_boundary, v_ode, u_ode,
                                                      semi, t, dt)

        pressures[i] = first(open_boundary.cache.pressure_boundary)
        flow_rates[i] = open_boundary.pressure_model_values[1].flow_rate[]
    end

    return pressures, flow_rates, times
end

# T = 1
T = 1 # 0.8302
dt = T / 100
tspan = (0, 5T)

tau = 0
beta = 1.0 # R1 / R2 ratio

R2 = 1.0
C = tau / R2
R1 = beta * R2

# func = pulsatile_velocity_fourier
func = pulsatile_velocity_sin
pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, dt, func)
pressures_lp = filtfilt(digitalfilter(Lowpass(dt / 2.1), Butterworth(3)), pressures)

p_Q = plot(times, flow_rates, color=:red, label=nothing, linewidth=2, ylims=(0, 1),
           title="Pulsatile Flow (sine pulse)", ylabel="Q", xlabel="t")
p_P1 = plot(times, pressures, color=:blue, label=nothing, linewidth=2, xlabel="t")
plot!(p_P1, times, pressures_lp, linewidth=2, linestyle=:dash, color=:gray,
      label=nothing, title="R1 = $R1, R2 = $R2, C = $C", ylabel="p")

tau = T / 4
beta = 2.0 # R1 / R2 ratio

R2 = 1.0
C = tau / R2
R1 = beta * R2

pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, dt, func)
pressures_lp = filtfilt(digitalfilter(Lowpass(dt / 2.1), Butterworth(3)), pressures)

p_P2 = plot(times, pressures, color=:blue, label=nothing, linewidth=2, xlabel="t")
plot!(p_P2, times, pressures_lp, linewidth=2, linestyle=:dash, color=:gray,
      label=nothing, title="R1 = $R1, R2 = $R2, C = $C", ylabel="p")

tau = T / 10
beta = 10.0 # R1 / R2 ratio

R2 = 1.0
C = tau / R2
R1 = beta * R2

pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, dt, func)
pressures_lp = filtfilt(digitalfilter(Lowpass(dt / 2.1), Butterworth(3)), pressures)

p_P3 = plot(times, pressures, color=:blue, label=nothing, linewidth=2, xlabel="t")
plot!(p_P3, times, pressures_lp, linewidth=2, linestyle=:dash, color=:gray,
      label=nothing, title="R1 = $R1, R2 = $R2, C = $C", ylabel="p")
p_1 = plot(p_Q, p_P1, p_P2, p_P3, layout=(4, 1), size=(600, 800), xlims=tspan)

dt = T / 100
tspan = (0, 5T)

tau = 0
beta = 1.0 # R1 / R2 ratio

R2 = 1.0
C = tau / R2
R1 = beta * R2

# R1 = 8500.0
# R2 = 500.0
# C = 1.368e-4

func = pulsatile_velocity_fourier
pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, dt, func)
pressures_lp = filtfilt(digitalfilter(Lowpass(dt / 2.1), Butterworth(3)), pressures)

p_Q = plot(times, flow_rates, color=:red, label=nothing, linewidth=2, ylims=(0, 1.0),
           title="Physiological Flow (Fourier)", ylabel="Q", xlabel="t")
p_P1 = plot(times, pressures, color=:blue, label=nothing, linewidth=2, xlabel="t")
plot!(p_P1, times, pressures_lp, linewidth=2, linestyle=:dash, color=:gray,
      label=nothing, title="R1 = $R1, R2 = $R2, C = $C", ylabel="p")

tau = T / 4
beta = 2.0 # R1 / R2 ratio

R2 = 1.0
C = tau / R2
R1 = beta * R2

pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, dt, func)
pressures_lp = filtfilt(digitalfilter(Lowpass(dt / 2.1), Butterworth(3)), pressures)

p_P2 = plot(times, pressures, color=:blue, label=nothing, linewidth=2, xlabel="t")
plot!(p_P2, times, pressures_lp, linewidth=2, linestyle=:dash, color=:gray,
      label=nothing, title="R1 = $R1, R2 = $R2, C = $C", ylabel="p")

tau = T / 10
beta = 10.0 # R1 / R2 ratio

R2 = 1.0
C = tau / R2
R1 = beta * R2

pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, dt, func)
pressures_lp = filtfilt(digitalfilter(Lowpass(dt / 2.1), Butterworth(3)), pressures)

p_P3 = plot(times, pressures, color=:blue, label=nothing, linewidth=2, xlabel="t")
plot!(p_P3, times, pressures_lp, linewidth=2, linestyle=:dash, color=:gray,
      label=nothing, title="R1 = $R1, R2 = $R2, C = $C", ylabel="p")
p_2 = plot(p_Q, p_P1, p_P2, p_P3, layout=(4, 1), size=(600, 800), xlims=tspan)

p = plot(p_1, p_2, layout=(1, 2), size=(800, 800))
