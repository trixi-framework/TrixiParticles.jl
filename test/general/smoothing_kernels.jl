@testset verbose=true "Smoothing Kernels" begin
    # Don't show all kernel tests in the final overview
    @testset verbose=false "Integral" begin
        # All smoothing kernels should integrate to something close to 1
        function integrate_kernel_2d(smk)
            integral_2d_radial,
            _ = quadgk(r -> r * TrixiParticles.kernel(smk, r, 1.0), 0,
                       TrixiParticles.compact_support(smk, 1.0),
                       rtol=1e-15)
            return 2 * pi * integral_2d_radial
        end

        function integrate_kernel_3d(smk)
            integral_3d_radial,
            _ = quadgk(r -> r^2 * TrixiParticles.kernel(smk, r, 1.0), 0,
                       TrixiParticles.compact_support(smk, 1.0),
                       rtol=1e-15)
            return 4 * pi * integral_3d_radial
        end

        # Treat the truncated Gaussian kernel separately
        @testset "GaussianKernel" begin
            error_2d = abs(integrate_kernel_2d(GaussianKernel{2}()) - 1.0)
            error_3d = abs(integrate_kernel_3d(GaussianKernel{3}()) - 1.0)

            # Large error due to truncation
            @test 1e-4 < error_2d < 1e-3
            @test 1e-4 < error_3d < 1e-3
        end

        # All other kernels
        kernels = [
            SchoenbergCubicSplineKernel,
            SchoenbergQuarticSplineKernel,
            SchoenbergQuinticSplineKernel,
            WendlandC2Kernel,
            WendlandC4Kernel,
            WendlandC6Kernel,
            SpikyKernel,
            Poly6Kernel
        ]

        @testset "$kernel" for kernel in kernels
            # The integral should be 1 for all kernels
            error_2d = abs(integrate_kernel_2d(kernel{2}()) - 1.0)
            error_3d = abs(integrate_kernel_3d(kernel{3}()) - 1.0)

            @test error_2d <= 1e-15
            @test error_3d <= 1e-15
        end
    end

    # Don't show all kernel tests in the final overview
    @testset verbose=false "Kernel Derivative" begin
        # Test `kernel_deriv` against automatic differentiation of `kernel`
        kernels = [
            GaussianKernel,
            SchoenbergCubicSplineKernel,
            SchoenbergQuarticSplineKernel,
            SchoenbergQuinticSplineKernel,
            WendlandC2Kernel,
            WendlandC4Kernel,
            WendlandC6Kernel,
            SpikyKernel,
            Poly6Kernel
        ]

        # Test 4 different smoothing lengths
        smoothing_lengths = 0.25:0.25:1

        @testset "$kernel_type" for kernel_type in kernels
            for ndims in 2:3
                kernel_ = kernel_type{ndims}()

                for h in smoothing_lengths
                    # Test 11 different radii
                    radii = 0:(0.1h):(h + eps())

                    for r in radii
                        # Automatic differentation of `kernel`
                        fun = x -> TrixiParticles.kernel(kernel_, x, h)
                        automatic_deriv = TrixiParticles.ForwardDiff.derivative(fun, r)

                        # Analytical derivative with `kernel_deriv`
                        analytic_deriv = TrixiParticles.kernel_deriv(kernel_, r, h)

                        # This should work with very tight tolerances
                        @test isapprox(analytic_deriv, automatic_deriv,
                                       rtol=5e-15, atol=4 * eps())
                    end
                end
            end
        end
    end

    @testset verbose=false "Return Type" begin
        # Test that the return type of the kernel and kernel derivative preserve
        # the input type. We don't want to return `Float64` when working with `Float32`.
        kernels = [
            GaussianKernel,
            SchoenbergCubicSplineKernel,
            SchoenbergQuarticSplineKernel,
            SchoenbergQuinticSplineKernel,
            WendlandC2Kernel,
            WendlandC4Kernel,
            WendlandC6Kernel,
            SpikyKernel,
            Poly6Kernel
        ]

        # Test different smoothing length types
        smoothing_lengths = (0.5, 0.5f0)

        @testset "$kernel_type" for kernel_type in kernels
            for ndims in 2:3
                kernel_ = kernel_type{ndims}()

                for h in smoothing_lengths
                    result = TrixiParticles.kernel(kernel_, h / 2, h)
                    @test typeof(result) == typeof(h)

                    result = TrixiParticles.kernel_deriv(kernel_, h / 2, h)
                    @test typeof(result) == typeof(h)
                end
            end
        end
    end
end;
