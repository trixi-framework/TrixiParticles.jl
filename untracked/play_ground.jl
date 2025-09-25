using TrixiParticles
using OrdinaryDiffEq
using Plots

# Mock fluid system
struct FluidSystemMock2 <: TrixiParticles.AbstractFluidSystem{2}
    pressure_acceleration_formulation::Nothing
    density_diffusion::Nothing
end
TrixiParticles.initial_smoothing_length(system::FluidSystemMock2) = 1.0
TrixiParticles.nparticles(system::FluidSystemMock2) = 1
TrixiParticles.system_smoothing_kernel(system::FluidSystemMock2) = nothing

function pulsatile_velocity_sin(t)
    amplitude = 1
    frequency = 3 / T

    t_periodic = mod(t, T)
    t_periodic > 1 / frequency && return 0.0

    return amplitude * sin(pi * frequency * t_periodic)^4
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

function pressure_RCR_ode!(dp, p, params, t)
    (; R_1, R_2, C, q_func, dt) = params

    dq_dt = (q_func(t) - q_func(t - dt)) / dt

    dp[1] = -p[1] / (R_1 * C) + R_2 * dq_dt + (R_1 + R_2) / (R_1 * C) * q_func(t)
end

# p_0 = 0.0

# T = 1
# Δt = T / 100
# tspan = (0, 5T)

# R2 = 1.0
# C = 1.0
# R1 = 1.0
# func = pulsatile_velocity_sin

# pressures, flow_rates, times = simulate_rcr(R1, R2, C, tspan, Δt, p_0, func)

# params_RCR = (R_1=R1, R_2=R2, C=C, q_func=func, dt=Δt)

# p0 = [0]

# sol_RCR = solve(ODEProblem(pressure_RCR_ode!, p0, tspan, params_RCR), RK4(), dt=Δt)

# plot(sol_RCR, color=:black, linewidth=2, label="1D")
# plot!(times, pressures, color=:red, linewidth=2, label="TP")
