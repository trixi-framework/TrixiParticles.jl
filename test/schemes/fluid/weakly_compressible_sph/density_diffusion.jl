@testset verbose=true "Density Diffusion" begin
    @testset verbose=true "DensityDiffusionAntuono" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "show" begin
            density_diffusion = DensityDiffusionAntuono(delta=0.1)

            @test repr(density_diffusion) ==
                  "DensityDiffusionAntuono(delta=0.1, update_everystage=true)"

            callback_density_diffusion = DensityDiffusionAntuono(delta=0.1,
                                                                 update_everystage=false)
            @test repr(callback_density_diffusion) ==
                  "DensityDiffusionAntuono(delta=0.1, update_everystage=false)"
            @test !TrixiParticles.requires_update_callback(density_diffusion)
            @test TrixiParticles.requires_update_callback(callback_density_diffusion)
        end

        @trixi_testset "Update from `UpdateCallback` with `update_everystage=false`" begin
            using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54

            # Independent reference implementation of the renormalized density gradient
            # as defined in the docstring of `DensityDiffusionAntuono`.
            # It loops over all pairs of particles instead of using the NHS.
            function reference_density_gradient(system, v, u)
                NDIMS = ndims(system)
                result = zeros(NDIMS, TrixiParticles.nparticles(system))

                for particle in TrixiParticles.eachparticle(system)
                    rho_a = TrixiParticles.current_density(v, system, particle)
                    x_a = TrixiParticles.current_coords(u, system, particle)

                    L = zeros(NDIMS, NDIMS)
                    gradient_a = zeros(NDIMS)
                    for neighbor in TrixiParticles.eachparticle(system)
                        pos_diff = x_a -
                                   TrixiParticles.current_coords(u, system, neighbor)
                        distance = norm(pos_diff)

                        if neighbor == particle ||
                           distance > TrixiParticles.compact_support(system, system)
                            continue
                        end

                        rho_b = TrixiParticles.current_density(v, system, neighbor)
                        volume_b = system.mass[neighbor] / rho_b
                        grad_kernel = TrixiParticles.smoothing_kernel_grad(system,
                                                                           pos_diff,
                                                                           distance,
                                                                           particle)

                        L -= volume_b * grad_kernel * pos_diff'
                        gradient_a -= (rho_a - rho_b) * volume_b * grad_kernel
                    end

                    result[:, particle] = inv(L) * gradient_a
                end

                return result
            end

            # 6x6 particles with a non-uniform density and mass.
            # The compressive velocity field makes the density change in every step,
            # and the particles move, so the gradient changes in every step as well.
            particle_spacing = 0.1
            center = SVector(0.3, 0.3)
            coordinates = RectangularShape(particle_spacing, (6, 6), (0.0, 0.0),
                                           density=1000.0).coordinates
            velocity = -0.5 * (coordinates .- center)
            density = 1000 .+ 10 * sin.(5 * coordinates[1, :]) .+
                      5 * cos.(7 * coordinates[2, :])
            initial_condition = InitialCondition(; coordinates, velocity, density,
                                                 particle_spacing)

            density_diffusion = DensityDiffusionAntuono(delta=0.1,
                                                        update_everystage=false)
            state_equation = StateEquationCole(sound_speed=10.0,
                                               reference_density=1000.0, exponent=7)
            system = WeaklyCompressibleSPHSystem(initial_condition;
                                                 smoothing_kernel=WendlandC2Kernel{2}(),
                                                 smoothing_length=1.2 * particle_spacing,
                                                 density_calculator=ContinuityDensity(),
                                                 state_equation, density_diffusion)
            semi = Semidiscretization(system)
            ode = semidiscretize(semi, (0.0, 1.0))

            # Refresh the cache at initialization and after every second time step.
            # The time step is well below the CFL limit.
            # Note that a staggered scheme like `SymplecticEuler` cannot be used here,
            # since its initialization modifies `v` after the callbacks are initialized.
            integrator = init(ode, CarpenterKennedy2N54(williamson_condition=false),
                              dt=1e-4, adaptive=false,
                              save_everystep=false,
                              callback=UpdateCallback(interval=2))

            semi = integrator.p.semi
            system = semi.systems[1]
            cached_gradient = system.cache.density_diffusion_normalized_density_gradient

            function current_reference(integrator)
                v_ode, u_ode = integrator.u.x
                v = TrixiParticles.wrap_v(v_ode, system, semi)
                u = TrixiParticles.wrap_u(u_ode, system, semi)

                return reference_density_gradient(system, v, u)
            end

            # `initial_update!` computes the gradient from the initial state.
            # Since the density is non-uniform, a correct gradient is non-zero.
            @test all(isfinite, cached_gradient)
            @test !iszero(cached_gradient)
            @test isapprox(cached_gradient, current_reference(integrator), rtol=1e-12)
            cached_gradient_initial = copy(cached_gradient)

            # After one step, the `UpdateCallback` is not triggered
            # (number of accepted steps 1 % 2 != 0), so the cache must not be changed,
            # although a refresh would now yield a different gradient.
            step!(integrator)
            @test integrator.stats.naccept == 1
            @test cached_gradient == cached_gradient_initial
            @test !isapprox(cached_gradient, current_reference(integrator), rtol=1e-6)

            # After the second step, the scheduled refresh must yield the gradient
            # of the current state.
            step!(integrator)
            @test integrator.stats.naccept == 2
            @test cached_gradient != cached_gradient_initial
            @test isapprox(cached_gradient, current_reference(integrator), rtol=1e-12)
        end
    end
end
