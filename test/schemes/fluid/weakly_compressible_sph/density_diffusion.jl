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
            function semidiscretize_patch(; shifting_technique=nothing)
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
                                                     smoothing_length=1.2 *
                                                                      particle_spacing,
                                                     density_calculator=ContinuityDensity(),
                                                     state_equation, density_diffusion,
                                                     shifting_technique)
                semi = Semidiscretization(system)

                return semidiscretize(semi, (0.0, 1.0))
            end

            # The time step is well below the CFL limit.
            # Note that a staggered scheme like `SymplecticEuler` cannot be used here,
            # since its initialization modifies `v` after the callbacks are initialized.
            function init_integrator(ode, callback)
                return init(ode, CarpenterKennedy2N54(williamson_condition=false),
                            dt=1e-4, adaptive=false, save_everystep=false, callback=callback)
            end

            @testset verbose=true "Refresh only when the `UpdateCallback` is triggered" begin
                # Refresh the cache at initialization and after every second time step.
                integrator = init_integrator(semidiscretize_patch(),
                                             UpdateCallback(interval=2))

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
                @test isapprox(cached_gradient, current_reference(integrator),
                               rtol=1e-12)
                cached_gradient_initial = copy(cached_gradient)

                # After one step, the `UpdateCallback` is not triggered
                # (number of accepted steps 1 % 2 != 0), so the cache must not be
                # changed, although a refresh would now yield a different gradient.
                step!(integrator)
                @test integrator.stats.naccept == 1
                @test cached_gradient == cached_gradient_initial
                @test !isapprox(cached_gradient, current_reference(integrator),
                                rtol=1e-6)

                # After the second step, the scheduled refresh must yield the gradient
                # of the current state.
                step!(integrator)
                @test integrator.stats.naccept == 2
                @test cached_gradient != cached_gradient_initial
                @test isapprox(cached_gradient, current_reference(integrator),
                               rtol=1e-12)
            end

            @testset verbose=true "Callback evaluating the RHS before the `UpdateCallback`" begin
                # The `PostprocessCallback` evaluates the RHS at initialization.
                # When it comes before the `UpdateCallback` in the `CallbackSet`,
                # the `UpdateCallback` has not been initialized yet, but the RHS must
                # already use the initialized caches.
                # This also covers the shifting velocity of PST with
                # `update_everystage=false`, which is used in the RHS when it is
                # integrated (`integrate_shifting_velocity=true`).
                shifting_technique = ParticleShiftingTechnique(update_everystage=false)
                ode = semidiscretize_patch(; shifting_technique)

                # Record the RHS that the `PostprocessCallback` sees.
                recorded_dv = []
                function record_dv(system, dv_ode, du_ode, v_ode, u_ode, semi, t)
                    push!(recorded_dv, copy(TrixiParticles.wrap_v(dv_ode, system, semi)))

                    # Don't store any values in the `PostprocessCallback`.
                    return nothing
                end
                postprocess_callback = PostprocessCallback(; interval=1000,
                                                           write_csv=false,
                                                           write_json=false, record_dv)

                integrator = init_integrator(ode,
                                             CallbackSet(postprocess_callback,
                                                         UpdateCallback()))
                semi = integrator.p.semi
                system = semi.systems[1]

                # Make sure that both caches are non-trivial, so that the test is
                # meaningful.
                @test !iszero(system.cache.density_diffusion_normalized_density_gradient)
                @test !iszero(system.cache.delta_v)

                # The RHS after all callbacks have been initialized.
                v_ode, u_ode = integrator.u.x
                dv_ode = similar(v_ode)
                TrixiParticles.kick!(dv_ode, v_ode, u_ode, integrator.p, integrator.t)

                @test length(recorded_dv) == 1
                @test isapprox(only(recorded_dv),
                               TrixiParticles.wrap_v(dv_ode, system, semi), rtol=1e-12)
            end
        end

        # The renormalized density gradient is defined with the uncorrected kernel gradient.
        # It must neither depend on the correction of the fluid nor on its correction
        # values, which are only computed after the density diffusion in each update.
        @trixi_testset "Uncorrected Kernel Gradient" begin
            # Irregular particle positions with a nonlinear density field
            coordinates = [0.0 0.8 1.7 0.3 1.1 2.1 0.6; 0.0 0.1 0.2 0.9 1.0 1.2 1.8]
            density = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1000.0, 1002.0]
            initial_condition = InitialCondition(; coordinates, density, mass=1.0,
                                                 particle_spacing=1.0)
            state_equation = StateEquationCole(; sound_speed=10.0,
                                               reference_density=1000.0, exponent=1)
            density_diffusion = DensityDiffusionAntuono(delta=0.1)

            corrections = (nothing, KernelCorrection(), GradientCorrection(),
                           BlendedGradientCorrection(0.3), MixedKernelGradientCorrection())
            results = map(corrections) do correction
                system = WeaklyCompressibleSPHSystem(initial_condition;
                                                     smoothing_kernel=SchoenbergCubicSplineKernel{2}(),
                                                     smoothing_length=1.0, state_equation,
                                                     density_calculator=ContinuityDensity(),
                                                     density_diffusion, correction)
                ode = semidiscretize(Semidiscretization(system;
                                                        parallelization_backend=SerialBackend()),
                                     (0.0, 0.01))
                semi = ode.p.semi
                system = semi.systems[1]
                v_ode, u_ode = ode.u0.x

                # The correction values of the fluid must not be used
                for field in (:kernel_correction_coefficient, :dw_gamma,
                              :correction_matrix)
                    haskey(system.cache, field) || continue
                    fill!(getproperty(system.cache, field), NaN)
                end

                TrixiParticles.update!(density_diffusion,
                                       TrixiParticles.wrap_v(v_ode, system, semi),
                                       TrixiParticles.wrap_u(u_ode, system, semi),
                                       system, semi)

                return (copy(system.cache.density_diffusion_correction_matrix),
                        copy(system.cache.density_diffusion_normalized_density_gradient))
            end

            matrix, normalized_gradient = first(results)
            @test all(isfinite, matrix)
            @test all(isfinite, normalized_gradient)
            @test !iszero(normalized_gradient)

            @testset "$(nameof(typeof(correction)))" for (correction,
                                                          result) in
                                                         zip(corrections[2:end],
                                                             results[2:end])
                @test result[1] == matrix
                @test result[2] == normalized_gradient
            end
        end
    end
end
