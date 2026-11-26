@testset verbose=true "Density Diffusion" begin
    @testset verbose=true "DensityDiffusionAntuono" begin
        # Use `@trixi_testset` to isolate the mock functions in a separate namespace
        @trixi_testset "show" begin
            density_diffusion = DensityDiffusionAntuono(delta=0.1)

            @test repr(density_diffusion) == "DensityDiffusionAntuono(0.1)"
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
                                                     density_diffusion,
                                                     gradient_correction=correction)
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
