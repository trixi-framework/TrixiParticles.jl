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

            @testset "$(nameof(typeof(density_calculator)))" for density_calculator in density_calculators
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

                    initial_condition = InitialCondition(coordinates, velocity, mass,
                                                         density)
                    system = WeaklyCompressibleSPHSystem(initial_condition,
                                                         density_calculator,
                                                         state_equation, smoothing_kernel,
                                                         smoothing_length)

                    # `system` is only used for the pressure
                    system.pressure .= [0.0, p_a, p_b]

                    # Compute accelerations a -> b and b -> a
                    dv1 = TrixiParticles.pressure_acceleration(1.0, m_b, particle,
                                                               neighbor,
                                                               system, system, rho_a, rho_b,
                                                               pos_diff, distance,
                                                               grad_kernel,
                                                               density_calculator)

                    dv2 = TrixiParticles.pressure_acceleration(1.0, m_a, neighbor,
                                                               particle,
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
end
