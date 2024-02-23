@testset verbose=true "EDAC RHS" begin
    # The following tests for linear and angular momentum and total energy conservation
    # are based on Sections 3.3.4 and 3.4.2 of
    # Daniel J. Price. "Smoothed Particle Hydrodynamics and Magnetohydrodynamics."
    # In: Journal of Computational Physics 231.3 (2012), pages 759–94.
    # https://doi.org/10.1016/j.jcp.2010.12.011
    @testset verbose=true "Momentum and Total Energy Conservation" begin
        # We are testing the momentum conservation of SPH with random initial configurations

        particle_spacing = 0.1

        smoothing_kernel = SchoenbergCubicSplineKernel{2}()
        smoothing_length = 1.2particle_spacing
        search_radius = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        # Run three times with different seed for the random initial condition
        for seed in 1:3
            # A larger number of particles will increase accumulated errors in the
            # summation. A larger tolerance has to be used for the tests below.
            fluid = rectangular_patch(particle_spacing, (3, 3), seed=seed)
            system = EntropicallyDampedSPHSystem(fluid, smoothing_kernel, smoothing_length,
                                                 0.0)
            n_particles = TrixiParticles.nparticles(system)

            u = fluid.coordinates

            # pressure is integrated
            v = vcat(fluid.velocity, fluid.pressure')

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
            deriv_angular_momentum_ = sum(deriv_angular_momentum, 1:n_particles)

            # Cross product is always 3-dimensional
            @test isapprox(deriv_angular_momentum_, zeros(3), atol=4e-15)

            # Total energy conservation
            function drho(particle)
                return sum(neighbor -> drho_particle(particle, neighbor), 1:n_particles)
            end

            # Derivative of the density summation.
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
                dte_a = p_a / rho_a^2 * drho(particle)
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
