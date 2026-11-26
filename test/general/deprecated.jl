# Tests of the deprecated kernel functions in `src/general/deprecated.jl`.
# TODO remove this file in the next major release.

# A custom kernel as described in the custom kernel tutorial of previous versions,
# which only defines the deprecated `_unsafe` functions.
struct DeprecatedGaussianKernel <: TrixiParticles.AbstractSmoothingKernel{2} end

function TrixiParticles.kernel_unsafe(kernel::DeprecatedGaussianKernel, r::Real, h)
    q = r / h

    return 1 / (pi * h^2) * exp(-q^2)
end

function TrixiParticles.kernel_deriv_div_r_unsafe(kernel::DeprecatedGaussianKernel,
                                                  r::Real, h)
    q = r / h

    kernel_deriv = 1 / (pi * h^2) * (-2 * q) * exp(-q^2) / h
    return kernel_deriv / r
end

TrixiParticles.compact_support(::DeprecatedGaussianKernel, h) = 2 * h

# A system with a single particle and the given kernel.
function deprecated_kernel_test_system(smoothing_kernel, smoothing_length)
    initial_condition = InitialCondition(; coordinates=zeros(2, 1), velocity=zeros(2, 1),
                                         mass=[1.0], density=[1000.0])
    state_equation = StateEquationCole(; sound_speed=10.0, reference_density=1000.0,
                                       exponent=7)

    return WeaklyCompressibleSPHSystem(initial_condition; smoothing_kernel,
                                       smoothing_length,
                                       density_calculator=ContinuityDensity(),
                                       state_equation)
end

@testset verbose=true "Deprecated Kernel Functions" begin
    h = 1.0
    distance = 0.5
    pos_diff = SVector(0.3, 0.4)
    zero_pos_diff = zero(pos_diff)

    @testset "Custom Kernel With `_unsafe` Functions" begin
        kernel = DeprecatedGaussianKernel()

        expected_kernel = 1 / pi * exp(-distance^2)
        expected_deriv = 1 / pi * (-2 * distance) * exp(-distance^2)

        @test TrixiParticles.kernel(kernel, distance, h) ≈ expected_kernel
        @test TrixiParticles.kernel_deriv(kernel, distance, h) ≈ expected_deriv
        @test TrixiParticles.kernel_grad(kernel, pos_diff, distance, h) ≈
              expected_deriv / distance * pos_diff

        # Zero outside of the compact support
        @test iszero(TrixiParticles.kernel(kernel, 2.5, h))
        @test iszero(TrixiParticles.kernel_deriv(kernel, 2.5, h))
        @test iszero(TrixiParticles.kernel_grad(kernel, SVector(1.5, 2.0), 2.5, h))

        # No division by zero for zero distance
        @test TrixiParticles.kernel(kernel, 0.0, h) ≈ 1 / pi
        @test iszero(TrixiParticles.kernel_deriv(kernel, 0.0, h))
        @test iszero(TrixiParticles.kernel_grad(kernel, zero_pos_diff, 0.0, h))

        # A system with this kernel
        system = deprecated_kernel_test_system(kernel, h)

        @test TrixiParticles.smoothing_kernel(system, distance, 1) ≈ expected_kernel
        @test TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance, 1) ≈
              expected_deriv / distance * pos_diff
        @test iszero(TrixiParticles.smoothing_kernel_grad(system, zero_pos_diff, 0.0, 1))
    end

    # Smoke test of the forwarding `_unsafe` functions and of `skip_zero_distance`
    @testset "`_unsafe` Functions and `skip_zero_distance`" begin
        kernel = SchoenbergCubicSplineKernel{2}()
        system = deprecated_kernel_test_system(kernel, h)

        @test TrixiParticles.kernel_unsafe(kernel, distance, h) ==
              TrixiParticles.kernel(kernel, distance, h)
        @test TrixiParticles.kernel_deriv_div_r_unsafe(kernel, distance, h) ==
              TrixiParticles.kernel_deriv_div_r(kernel, distance, h)
        @test TrixiParticles.kernel_grad_unsafe(kernel, pos_diff, distance, h) ==
              TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)
        @test TrixiParticles.corrected_kernel_grad_unsafe(kernel, pos_diff, distance, h,
                                                          nothing, nothing, 1) ==
              TrixiParticles.kernel_grad(kernel, pos_diff, distance, h)
        @test TrixiParticles.smoothing_kernel_unsafe(system, distance, 1) ==
              TrixiParticles.smoothing_kernel(system, distance, 1)
        @test TrixiParticles.smoothing_kernel_grad_unsafe(system, pos_diff, distance, 1) ==
              TrixiParticles.smoothing_kernel_grad(system, pos_diff, distance, 1)

        @test TrixiParticles.skip_zero_distance(system)
        @test !TrixiParticles.skip_zero_distance(KernelCorrection())
    end
end
