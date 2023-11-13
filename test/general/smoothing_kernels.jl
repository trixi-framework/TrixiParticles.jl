@trixi_testset "Density Estimation Kernel" begin
    particle_spacing = 0.1
    density_ref = 1.0
    domain_size = 1.5
    nx = Int(round(domain_size / particle_spacing))
    smoothing_lengths = particle_spacing * [0.8; 1.0; 1.2; 1.4]
    dimensions = [1, 2, 3]

    @testset "$n_dims-D" for n_dims in dimensions
        mass = density_ref * particle_spacing^n_dims

        setup = RectangularShape(particle_spacing, ntuple(_ -> nx, n_dims),
                                 ntuple(_ -> 0.0, n_dims), density_ref)
        coords = setup.coordinates

        smoothing_kernels = [SchoenbergCubicSplineKernel{n_dims}()
                             SchoenbergQuarticSplineKernel{n_dims}()
                             SchoenbergQuinticSplineKernel{n_dims}()]

        normalized_w = zeros(size(coords, 2), 1)
        density = zeros(size(coords, 2), 1)

        @testset "$(TrixiParticles.type2string(smoothing_kernel)), h = $(round(smoothing_length, digits=2))" for smoothing_kernel in smoothing_kernels,
                                                                                                                 smoothing_length in smoothing_lengths

            compact_support = TrixiParticles.compact_support(smoothing_kernel,
                                                             smoothing_length)

            TrixiParticles.set_zero!(normalized_w)
            TrixiParticles.set_zero!(density)

            nhs = GridNeighborhoodSearch{n_dims}(compact_support, size(coords, 2);
                                                 min_corner=[0.0 for i in 1:n_dims],
                                                 max_corner=[domain_size
                                                             for i in 1:n_dims])
            TrixiParticles.initialize!(nhs, coords)

            # Compute summation density
            TrixiParticles.for_particle_neighbor(nothing, nothing,
                                                 coords, coords, nhs,
                                                 particles=axes(coords, 2)) do particle,
                                                                               neighbor,
                                                                               pos_diff,
                                                                               distance
                density[particle] += TrixiParticles.kernel(smoothing_kernel, distance,
                                                           smoothing_length) * mass
            end

            # Computed normalized kernel.
            TrixiParticles.for_particle_neighbor(nothing, nothing, coords, coords, nhs,
                                                 particles=axes(coords, 2)) do particle,
                                                                               neighbor,
                                                                               pos_diff,
                                                                               distance
                normalized_w[particle] += TrixiParticles.kernel(smoothing_kernel,
                                                                distance,
                                                                smoothing_length) *
                                          mass / density[neighbor]
            end

            # Test normalization condition
            @test isapprox(normalized_w, ones(length(normalized_w)))
        end
    end
end
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

    # All smoothing kernels should integrate to something close to 1.
    # Don't show all kernel tests in the final overview.
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
