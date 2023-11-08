using QuadGK

@testset verbose=true "Smoothing Kernels" begin
    function integrate_kernel_2d(smk)
        integral_2d_radial, _ = quadgk(r -> r * TrixiParticles.kernel(smk, r, 1.0), 0,
                                       TrixiParticles.compact_support(smk, 1.0), rtol=1e-15)
        return 2 * π * integral_2d_radial
    end

    function integrate_kernel_3d(smk)
        integral_3d_radial, _ = quadgk(r -> r^2 * TrixiParticles.kernel(smk, r, 1.0), 0,
                                       TrixiParticles.compact_support(smk, 1.0), rtol=1e-15)
        return 4 * π * integral_3d_radial
    end

    # All smoothing kernels should integrate to something close to 1
    # Don't show all kernel tests in the final overview
    @testset verbose=false "Integral" begin
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
            Poly6Kernel,
        ]

        @testset "$kernel" for kernel in kernels
            # The integral should be 1 for all kernels
            error_2d = abs(integrate_kernel_2d(kernel{2}()) - 1.0)
            error_3d = abs(integrate_kernel_3d(kernel{3}()) - 1.0)

            @test error_2d <= 1e-15
            @test error_3d <= 1e-15
        end
    end
end
