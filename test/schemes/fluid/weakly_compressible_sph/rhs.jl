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
                    density = ones(3)
                    state_equation = Val(:state_equation)
                    smoothing_kernel = Val(:smoothing_kernel)
                    TrixiParticles.ndims(::Val{:smoothing_kernel}) = 2
                    smoothing_length = -1.0

                    fluid = InitialCondition(; coordinates, velocity, mass, density)
                    system = WeaklyCompressibleSPHSystem(fluid,
                                                         density_calculator,
                                                         state_equation, smoothing_kernel,
                                                         smoothing_length)

                    # `system` is only used for the pressure
                    system.pressure .= [0.0, p_a, p_b]

                    # Compute accelerations a -> b and b -> a
                    dv1 = TrixiParticles.pressure_acceleration(system, system, -1,
                                                               m_a, m_b, p_a, p_b,
                                                               rho_a, rho_b, pos_diff,
                                                               distance, grad_kernel, 1.0,
                                                               nothing)

                    dv2 = TrixiParticles.pressure_acceleration(system, system, -1,
                                                               m_b, m_a, p_b, p_a,
                                                               rho_b, rho_a, -pos_diff,
                                                               distance, -grad_kernel, 1.0,
                                                               nothing)

                    # Test that both forces are identical but in opposite directions
                    @test isapprox(m_a * dv1, -m_b * dv2, rtol=2eps())
                end
            end
        end
    end

    # The following tests for linear and angular momentum and total energy conservation
    # are based on Sections 3.3.4 and 3.4.2 of
    # Daniel J. Price. "Smoothed Particle Hydrodynamics and Magnetohydrodynamics."
    # In: Journal of Computational Physics 231.3 (2012), pages 759–94.
    # https://doi.org/10.1016/j.jcp.2010.12.011
    @testset verbose=true "Momentum and Total Energy Conservation" begin
        # We are testing the momentum conservation of SPH with random initial configurations
        density_calculators = [ContinuityDensity(), SummationDensity()]

        particle_spacing = 0.1

        # The state equation is only needed to unpack `sound_speed`, so we can mock
        # it by using a `NamedTuple`.
        state_equation = (; sound_speed=0.0)
        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        @testset "`$(nameof(typeof(density_calculator)))`" for density_calculator in density_calculators
            # Run three times with different seed for the random initial condition
            for seed in 1:3
                # A larger number of particles will increase accumulated errors in the
                # summation. A larger tolerance has to be used for the tests below.
                fluid = rectangular_patch(particle_spacing, (3, 3), seed=seed)
                system = WeaklyCompressibleSPHSystem(fluid, density_calculator,
                                                     state_equation, smoothing_kernel,
                                                     smoothing_length)
                n_particles = TrixiParticles.nparticles(system)

                # Overwrite `system.pressure` because we skip the update step
                system.pressure .= fluid.pressure

                u = fluid.coordinates
                if density_calculator isa SummationDensity
                    # Density is stored in the cache
                    v = fluid.velocity
                    system.cache.density .= fluid.density
                else
                    # Density is integrated with `ContinuityDensity`
                    v = vcat(fluid.velocity, fluid.density')
                end

                nhs = TrixiParticles.TrivialNeighborhoodSearch{2}(search_radius,
                                                                  TrixiParticles.eachparticle(system))

                # Result
                dv = zero(v)
                TrixiParticles.interact!(dv, v, u, v, u, nhs, system, system)

                # Linear momentum conservation
                # ∑ m_a dv_a
                deriv_linear_momentum = sum(fluid.mass' .* view(dv, 1:2, :), dims=2)

                @test isapprox(deriv_linear_momentum, zeros(2, 1), atol=5e-14)

                # Angular momentum conservation
                # m_a (r_a × dv_a)
                function deriv_angular_momentum(particle)
                    r_a = SVector(u[1, particle], u[2, particle], 0.0)
                    dv_a = SVector(dv[1, particle], dv[2, particle], 0.0)

                    return fluid.mass[particle] * cross(r_a, dv_a)
                end

                # ∑ m_a (r_a × dv_a)
                deriv_angular_momentum = sum(deriv_angular_momentum, 1:n_particles)

                # Cross product is always 3-dimensional
                @test isapprox(deriv_angular_momentum, zeros(3), atol=4e-15)

                # Total energy conservation
                drho(::ContinuityDensity, particle) = dv[end, particle]

                function drho(::SummationDensity, particle)
                    return sum(neighbor -> drho_particle(particle, neighbor), 1:n_particles)
                end

                # Derivative of the density summation. This is a slightly different
                # formulation of the continuity equation.
                function drho_particle(particle, neighbor)
                    m_b = TrixiParticles.hydrodynamic_mass(system, neighbor)
                    v_diff = TrixiParticles.current_velocity(v, system, particle) -
                             TrixiParticles.current_velocity(v, system, neighbor)

                    pos_diff = TrixiParticles.current_coords(u, system, particle) -
                               TrixiParticles.current_coords(u, system, neighbor)
                    distance = norm(pos_diff)

                    # Only consider particles with a distance > 0
                    distance < sqrt(eps()) && return 0.0

                    grad_kernel = TrixiParticles.smoothing_kernel_grad(system, pos_diff,
                                                                       distance)

                    return m_b * dot(v_diff, grad_kernel)
                end

                # m_a (v_a ⋅ dv_a + dte_a),
                # where `te` is the thermal energy, called `u` in the Price paper.
                function deriv_energy(particle)
                    p_a = fluid.pressure[particle]
                    rho_a = fluid.density[particle]
                    dte_a = p_a / rho_a^2 * drho(density_calculator, particle)
                    v_a = TrixiParticles.extract_svector(v, system, particle)
                    dv_a = TrixiParticles.extract_svector(dv, system, particle)

                    return fluid.mass[particle] * (dot(v_a, dv_a) + dte_a)
                end

                # ∑ m_a (v_a ⋅ dv_a + dte_a)
                deriv_total_energy = sum(deriv_energy, 1:n_particles)

                @test isapprox(deriv_total_energy, 0.0, atol=6e-15)
            end
        end
    end
end
