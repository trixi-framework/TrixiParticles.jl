using QuadGK
#include("../../src/general/smoothing_kernels.jl")

function integrate_kernel_2d(smk)
    integral_2d_radial, _ = quadgk(r -> r * TrixiParticles.kernel(smk, r, 1.0), 0,
                                   TrixiParticles.compact_support(smk, 1.0))
    return 2 * π * integral_2d_radial
end

function integrate_kernel_3d(smk)
    integral_3d_radial, _ = quadgk(r -> r^2 * TrixiParticles.kernel(smk, r, 1.0), 0,
                                   TrixiParticles.compact_support(smk, 1.0))
    return 4 * π * integral_3d_radial
end

@testset verbose=true "SmoothingKernel" begin

    # All smoothing kernels should integrate to something close to 1
    smk_2d = GaussianKernel{2}()
    smk_3d = GaussianKernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    # large error due to truncation
    @test error_2d <= 1e-3
    @test error_3d <= 1e-3

    smk_2d = SchoenbergCubicSplineKernel{2}()
    smk_3d = SchoenbergCubicSplineKernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-15
    @test error_3d <= 1e-15

    smk_2d = SchoenbergQuarticSplineKernel{2}()
    smk_3d = SchoenbergQuarticSplineKernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-9
    @test error_3d <= 1e-9

    smk_2d = SchoenbergQuinticSplineKernel{2}()
    smk_3d = SchoenbergQuinticSplineKernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-9
    @test error_3d <= 1e-9

    smk_2d = WendlandC2Kernel{2}()
    smk_3d = WendlandC2Kernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-15
    @test error_3d <= 1e-15

    smk_2d = WendlandC4Kernel{2}()
    smk_3d = WendlandC4Kernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-15
    @test error_3d <= 1e-15

    smk_2d = WendlandC6Kernel{2}()
    smk_3d = WendlandC6Kernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-15
    @test error_3d <= 1e-15

    smk_2d = SpikyKernel{2}()
    smk_3d = SpikyKernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-15
    @test error_3d <= 1e-15

    smk_2d = Poly6Kernel{2}()
    smk_3d = Poly6Kernel{3}()
    error_2d = abs(integrate_kernel_2d(smk_2d) - 1.0)
    error_3d = abs(integrate_kernel_3d(smk_3d) - 1.0)

    @test error_2d <= 1e-15
    @test error_3d <= 1e-15
end
