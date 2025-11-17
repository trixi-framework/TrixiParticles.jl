@testset verbose=true "`RCRWindkesselModel`" begin
    @testset verbose=true "Show" begin
        pressure_model = RCRWindkesselModel(; peripheral_resistance=1.2,
                                            compliance=0.5,
                                            characteristic_resistance=2.3)

        show_box = """
            ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
            │ RCRWindkesselModel                                                                               │
            │ ══════════════════                                                                               │
            │ characteristic_resistance: ………… 2.3                                                              │
            │ peripheral_resistance: …………………… 1.2                                                              │
            │ compliance: ………………………………………………… 0.5                                                              │
            └──────────────────────────────────────────────────────────────────────────────────────────────────┘"""

        @test repr("text/plain", pressure_model) == show_box
    end

    @testset verbose=true "Validation" begin
        # The following test example is adapted from a case study presented
        # in Gasser (2021, https://link.springer.com/book/10.1007/978-3-030-70966-2),
        # which uses data from Burattini et al. (2002, https://doi.org/10.1152/ajpheart.2002.282.1.h244)
        # to simulate a ferret vascular system.
        function pulsatile_flow(t)
            c_q = [
                0.0136667 - 0.00144338im,   # k = -10
                0.00722562 - 0.0347752im,   # k = -9
                -0.0593984 + 0.0217432im,   # k = -8
                -0.0233298 + 0.0691505im,   # k = -7
                0.0250477 + 0.0231058im,    # k = -6
                0.0369504 + 0.0725im,       # k = -5
                0.216635 - 0.0557698im,     # k = -4
                -0.0896779 - 0.408525im,    # k = -3
                -0.555816 + 0.112961im,     # k = -2
                0.0276938 + 0.646413im,     # k = -1
                0.786333,                   # k = 0
                0.0276938 - 0.646413im,     # k = 1
                -0.555816 - 0.112961im,     # k = 2
                -0.0896779 + 0.408525im,    # k = 3
                0.216635 + 0.0557698im,     # k = 4
                0.0369504 - 0.0725im,       # k = 5
                0.0250477 - 0.0231058im,    # k = 6
                -0.0233298 - 0.0691505im,   # k = 7
                -0.0593984 - 0.0217432im,   # k = 8
                0.00722562 + 0.0347752im,   # k = 9
                0.0136667 + 0.00144338im    # k = 10
            ]

            signal = 0.0
            for (k, c) in zip(-10:10, c_q)
                signal += real(c * exp(1im * k * 2 * pi / T * t))
            end

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
                                         particle_spacing=0.1, face_normal=(-1.0, 0.0),
                                         density=1000.0, reference_velocity,
                                         reference_pressure=pressure_model,
                                         open_boundary_layers=1, rest_pressure=p_0)

            system = OpenBoundarySystem(boundary_zone; buffer_size=0,
                                        boundary_model=nothing,
                                        fluid_system=FluidSystemMockRCR(nothing, nothing))
            system.boundary_zone_indices .= 1

            u = system.initial_condition.coordinates
            v = system.initial_condition.velocity

            times = collect(tspan[1]:dt:tspan[2])
            p_calculated = empty(times)
            for t in times
                v[1, :] .= func(t)
                TrixiParticles.calculate_flow_rate_and_pressure!(system, v, u, dt)

                # Store only values after the seventh cycle
                if t >= 7T
                    p = TrixiParticles.reference_pressure(boundary_zone, v, system, 1, 0, t)
                    push!(p_calculated, p)
                end
            end

            return p_calculated
        end

        T = 0.375
        dt = T / 500
        tspan = (0.0, 8T)
        p0 = 78.0

        R1 = 1.7714
        R2 = 106.66
        C = 1.1808e-2

        # The reference pressure values are computed using an ODE that describes
        # the behavior of the RCR Windkessel model. The governing equation is:
        #
        #   dp/dt + p / (R_2 * C) = R_1 * dq/dt + (R_2 + R_1) / (R_2 * C) * q
        #
        # where
        #   - p: pressure
        #   - q: time-dependent flow, provided by `pulsatile_flow`
        #   - R_1: characteristic resistance
        #   - R_2: peripheral resistance
        #   - C: compliance
        #
        # The function `pressure_RCR_ode!` implements this equation for numerical solution:
        # function pressure_RCR_ode!(dp, p, params, t)
        #     (; R_1, R_2, C, q_func, dt) = params
        #     dq_dt = (q_func(t) - q_func(t - dt)) / dt  # numerical derivative of inflow
        #     dp[1] = -p[1] / (R_2 * C) + R_1 * dq_dt + q_func(t) * (R_2 + R_1) / (R_2 * C)
        # end
        #
        # The solution is obtained using simple Euler integration over the time interval `tspan`.
        # Reference pressure values are evaluated at discrete time points (`validation_times`):
        # params_RCR = (R_2=R2, R_1=R1, C=C, q_func=pulsatile_flow, dt=dt)
        # sol_RCR = solve(ODEProblem(pressure_RCR_ode!, [p0], tspan, params_RCR), Euler(),
        #                 dt=dt, adaptive=false)
        # validation_times = collect(range(7T, 8T, step=50 * dt))
        # pressures_ref = vec(stack(sol_RCR(validation_times)))
        #
        # The values listed below are taken from this ODE simulation and serve as a baseline
        # for validating the implementation of `RCRWindkesselModel` in the test.
        pressures_ref = [
            78.23098908459171,
            76.0357395466389,
            86.63098598789874,
            96.04303886358863,
            91.35701317358787,
            88.89163496483422,
            87.1572215816713,
            84.95271784314382,
            82.79365887019439,
            80.25102392189406,
            78.23943618409275
        ]

        pressures = simulate_rcr(R1, R2, C, tspan, dt, p0, pulsatile_flow)

        @test isapprox(pressures_ref, pressures[1:50:end], rtol=5e-3)
    end
end
