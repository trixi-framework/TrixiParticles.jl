@testset verbose=true "WCSPH RHS" begin
    @testset verbose=true "`pressure_acceleration`" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "Symmetry" begin
            density_calculators = [ContinuityDensity(), SummationDensity()]
            masses = [[0.01, 0.01], [0.73, 0.31]]
            densities = [
                [1000.0, 1000.0],
                [1000.0, 1000.0],
                [900.0, 1201.0],
                [1003.0, 353.4],
            ]
            pressures = [
                [0.0, 0.0],
                [10_000.0, 10_000.0],
                [10.0, 10_000.0],
                [1000.0, -1000.0],
            ]
            grad_kernels = [0.3, 104.0]
            particle = 2
            neighbor = 3

            # Not used for fluid-fluid interaction
            pos_diff = 0
            distance = 0

            @testset "`$(nameof(typeof(density_calculator)))`" for density_calculator in density_calculators
                for (m_a, m_b) in masses, (rho_a, rho_b) in densities,
                    (p_a, p_b) in pressures, grad_kernel in grad_kernels

                    # Partly copied from constructor test, just to create a WCSPH system
                    coordinates = zeros(2, 3)
                    velocity = zeros(2, 3)
                    mass = zeros(3)
                    density = zeros(3)
                    state_equation = Val(:state_equation)
                    smoothing_kernel = Val(:smoothing_kernel)
                    TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
                    smoothing_length = -1.0
                    correction = Nothing()

                    initial_condition = InitialCondition(coordinates, velocity, mass,
                                                         density)
                    system = WeaklyCompressibleSPHSystem(initial_condition,
                                                         density_calculator,
                                                         state_equation, smoothing_kernel,
                                                         smoothing_length)

                    # `system` is only used for the pressure
                    system.pressure .= [0.0, p_a, p_b]

                    # Compute accelerations a -> b and b -> a
                    dv1 = TrixiParticles.pressure_acceleration(1.0, m_b, particle, neighbor,
                                                               system, system, rho_a, rho_b,
                                                               pos_diff, distance,
                                                               grad_kernel,
                                                               density_calculator)

                    dv2 = TrixiParticles.pressure_acceleration(1.0, m_a, neighbor, particle,
                                                               system, system, rho_b, rho_a,
                                                               -pos_diff, distance,
                                                               -grad_kernel,
                                                               density_calculator)

                    # Test that both forces are identical but in opposite directions
                    @test isapprox(m_a * dv1, -m_b * dv2, rtol=2eps())
                end
            end
        end
    end

    @testset verbose=true "`interact!`" begin
        # The following tests for linear and angular momentum and total energy conservation
        # are based on Sections 3.3.4 and 3.4.2 of
        # Daniel J. Price. "Smoothed Particle Hydrodynamics and Magnetohydrodynamics."
        # In: Journal of Computational Physics 231.3 (2012), pages 759–94.
        # https://doi.org/10.1016/j.jcp.2010.12.011
        @testset verbose=true "Momentum and Total Energy Conservation" begin
            # We are testing the momentum conservation of SPH with random initial configurations
            density_calculators = [ContinuityDensity(), SummationDensity()]

            # Random initial configuration
            mass = [
                [3.11, 1.55, 2.22, 3.48, 0.21, 3.73, 0.21, 3.45],
                [0.82, 1.64, 1.91, 0.02, 0.08, 1.58, 4.94, 0.7],
            ]
            density = [
                [914.34, 398.36, 710.22, 252.54, 843.81, 694.73, 670.5, 539.14],
                [280.15, 172.25, 267.1, 130.42, 382.3, 477.21, 848.31, 188.62],
            ]
            pressure = [
                [91438.0, 16984.0, 58638.0, 10590.0, 92087.0, 66586.0, 64723.0, 49862.0],
                [31652.0, -21956.0, 2874.0, -12489.0, 27206.0, 32225.0, 42848.0, 3001.0],
            ]
            coordinates = [
                [0.16 0.55 0.08 0.58 0.52 0.26 0.32 0.99;
                 0.76 0.6 0.47 0.4 0.25 0.79 0.45 0.63],
                [0.4 0.84 0.47 0.02 0.64 0.85 0.02 0.15;
                 0.83 0.62 0.99 0.57 0.25 0.72 0.34 0.69],
            ]
            velocity = [
                [1.05 0.72 0.12 1.22 0.67 0.85 1.42 -0.57;
                 1.08 0.68 0.74 -0.27 -1.22 0.43 1.41 1.25],
                [-1.84 -1.4 5.21 -5.99 -5.02 9.5 -4.51 -8.28;
                 0.78 0.1 9.67 8.46 9.29 5.18 -4.83 -4.87]]

            # The state equation is only needed to unpack `sound_speed`, so we can mock
            # it by using a `NamedTuple`.
            state_equation = (; sound_speed=0.0)
            smoothing_kernel = SchoenbergCubicSplineKernel{2}()
            smoothing_length = 0.3
            search_radius = TrixiParticles.compact_support(smoothing_kernel,
                                                           smoothing_length)

            @testset "`$(nameof(typeof(density_calculator)))`" for density_calculator in density_calculators
                for i in eachindex(mass)
                    initial_condition = InitialCondition(coordinates[i], velocity[i],
                                                         mass[i], density[i])
                    system = WeaklyCompressibleSPHSystem(initial_condition,
                                                         density_calculator,
                                                         state_equation, smoothing_kernel,
                                                         smoothing_length)

                    # Overwrite `system.pressure`
                    system.pressure .= pressure[i]

                    u = coordinates[i]
                    if density_calculator isa SummationDensity
                        # Density is stored in the cache
                        v = velocity[i]
                        system.cache.density .= density[i]
                    else
                        # Density is integrated with `ContinuityDensity`
                        v = vcat(velocity[i], density[i]')
                    end

                    nhs = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius,
                                                                      TrixiParticles.eachparticle(system))

                    # Result
                    dv = zeros(3, 8)
                    TrixiParticles.interact!(dv, v, u, v, u, nhs, system, system)

                    # Linear momentum conservation
                    # ∑ m_a dv_a
                    deriv_linear_momentum = sum(mass[i]' .* view(dv, 1:2, :), dims=2)

                    @test isapprox(deriv_linear_momentum, zeros(2, 1), atol=3e-14)

                    # Angular momentum conservation
                    # m_a (r_a × dv_a)
                    function deriv_angular_momentum(particle)
                        r_a = SVector(u[1, particle], u[2, particle], 0.0)
                        dv_a = SVector(dv[1, particle], dv[2, particle], 0.0)

                        return mass[i][particle] * cross(r_a, dv_a)
                    end

                    # ∑ m_a (r_a × dv_a)
                    deriv_angular_momentum = sum(deriv_angular_momentum, 1:8)

                    @test isapprox(deriv_angular_momentum, zeros(3), atol=2e-14)

                    # Total energy conservation
                    drho(::ContinuityDensity, particle) = dv[3, particle]

                    # Derivative of the density summation. This is a slightly different
                    # formulation of the continuity equation.
                    function drho_particle(particle, neighbor)
                        m_b = mass[i][neighbor]
                        vdiff = TrixiParticles.current_velocity(v, system, particle) -
                                TrixiParticles.current_velocity(v, system, neighbor)

                        pos_diff = TrixiParticles.current_coords(u, system, particle) -
                                   TrixiParticles.current_coords(u, system, neighbor)
                        distance = norm(pos_diff)

                        # Only consider particles with a distance > 0
                        distance < sqrt(eps()) && return 0.0

                        grad_kernel = TrixiParticles.smoothing_kernel_grad(system,
                                                                           pos_diff,
                                                                           distance)

                        return m_b * dot(vdiff, grad_kernel)
                    end

                    function drho(::SummationDensity, particle)
                        return sum(neighbor -> drho_particle(particle, neighbor), 1:8)
                    end

                    # m_a (v_a ⋅ dv_a + dte_a),
                    # where `te` is the thermal energy, called `u` in the Price paper.
                    function deriv_energy(particle)
                        dte_a = pressure[i][particle] / density[i][particle]^2 *
                                drho(density_calculator, particle)
                        v_a = TrixiParticles.extract_svector(v, system, particle)
                        dv_a = TrixiParticles.extract_svector(dv, system, particle)

                        return mass[i][particle] * (dot(v_a, dv_a) + dte_a)
                    end

                    # ∑ m_a (v_a ⋅ dv_a + dte_a)
                    deriv_total_energy = sum(deriv_energy, 1:8)

                    @test isapprox(deriv_total_energy, 0.0, atol=4e-14)
                end
            end
        end
    end
end
