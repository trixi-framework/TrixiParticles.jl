@trixi_testset "Normalization Condition Kernel" begin
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
