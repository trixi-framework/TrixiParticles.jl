using TrixiParticles
using OrdinaryDiffEq
using Plots

# The following test example is adapted from a case study presented
# in Gasser (2021, https://link.springer.com/book/10.1007/978-3-030-70966-2),
# which uses data from Burattini et al. (2002, https://doi.org/10.1152/ajpheart.2002.282.1.h244)
# to simulate a ferret vascular system.
function pulsatile_flow(t)
    # Fourier coefficients from Table 2.5
    omega_1 = 2π / T

    # Coefficients [ml s^-1]
    c0 = 0.786333
    c1 = 0.0276938 - 0.646413im
    c2 = -0.555816 - 0.112961im
    c3 = -0.0896779 + 0.408525im
    c4 = 0.216635 + 0.0557698im

    # Calculate signal using complex exponentials
    # q(t) = Re[c0 + 2*Re(c1*exp(iω₁t)) + 2*Re(c2*exp(i2ω₁t)) + ...]
    signal = c0
    signal += 2 * real(c1 * exp(1im * omega_1 * t))
    signal += 2 * real(c2 * exp(1im * 2 * omega_1 * t))
    signal += 2 * real(c3 * exp(1im * 3 * omega_1 * t))
    signal += 2 * real(c4 * exp(1im * 4 * omega_1 * t))

    return signal
end

# Mock fluid system
struct FluidSystemMockRCR <: TrixiParticles.AbstractFluidSystem{2}
    pressure_acceleration_formulation::Nothing
    density_diffusion::Nothing
end
TrixiParticles.initial_smoothing_length(system::FluidSystemMockRCR) = 1.0
TrixiParticles.nparticles(system::FluidSystemMockRCR) = 1
TrixiParticles.system_smoothing_kernel(system::FluidSystemMockRCR) = nothing

# Encapsulated simulation updating only `calculate_flow_rate_and_pressure!`
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

    system = OpenBoundarySystem(boundary_zone; buffer_size=0,
                                boundary_model=nothing,
                                fluid_system=FluidSystemMockRCR(nothing, nothing))

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

    return p_calculated
end

T = 0.375
n_cycles = 100
Δt = T / 1000
tspan = (0.0, n_cycles * T)
p0 = 78.0

R1 = 108.41
R2 = 4.37
C = 84.787e-3

# The reference pressure values are computed using an ODE that describes
# the behavior of the RCR Windkessel model. The governing equation is:
#
#   dp/dt + p / (R_1 * C) = R_2 * dq/dt (R_1 + R_2) / (R_1 * C) * q
#
# where
#   - p: pressure
#   - q: time-dependent flow, provided by `pulsatile_flow`
#   - R_1: characteristic resistance
#   - R_2: peripheral resistance
#   - C: compliance
#
# The function `pressure_RCR_ode!` implements this equation for numerical solution:
function pressure_RCR_ode!(dp, p, params, t)
    (; R_1, R_2, C, q_func, dt) = params
    dq_dt = (q_func(t) - q_func(t - dt)) / dt  # numerical derivative of inflow
    dp[1] = -p[1] / (R_1 * C) + R_2 * dq_dt + q_func(t) * (R_1 + R_2) / (R_1 * C)
end
#
# The solution is obtained using an explicit Runge-Kutta method (RK4) over the time interval `tspan`.
# Reference pressure values are evaluated at discrete time points (`validation_times`):
#
params_RCR = (R_1=R1, R_2=R2, C=C, q_func=pulsatile_flow, dt=Δt)
sol_RCR = solve(ODEProblem(pressure_RCR_ode!, [p0], tspan, params_RCR), RK4())

pressures = simulate_rcr(R1, R2, C, tspan, Δt, p0, pulsatile_flow)

p1 = plot(pulsatile_flow, xlims=(0, T), color=:red, linewidth=3, xlabel="t [s]",
          ylabel="ml/s", label=nothing)
p2 = plot(sol_RCR, xlims=(T * (n_cycles - 1), T * n_cycles), linewidth=3, color=:black,
          linestyle=:dash, label="ODE")
p3 = plot(range(tspan[1], tspan[2], length=length(pressures)), pressures,
          linewidth=3, xlims=(T * (n_cycles - 1), T * n_cycles), label="simulation")
