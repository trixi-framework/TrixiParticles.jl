#@testset verbose=true "GridNeighborhoodSearch" begin
using TrixiParticles
using LinearAlgebra, Random

particle_spacing = 0.1
density_ref = 1.0
domain_size = 1.5
nx = Int(round(domain_size / particle_spacing))
smoothing_lengths = particle_spacing * [0.8; 1.0; 1.2; 1.4]
dimensions = [1, 2, 3]

# Expected `rtol`s of exact solutions:
# `h => [normalize_w density;   dim 1
#        normalize_w density;   dim 2
#        normalize_w density]`  dim 3
cubic = Dict(smoothing_lengths[1] => [5eps() 1e-2;  # dim 1
                                      5eps() 2e-2;  # dim 2
                                      5eps() 4e-2], # dim 3
             smoothing_lengths[2] => [5eps() 10eps();
                                      5eps() 9e-4;
                                      5eps() 3e-5],
             smoothing_lengths[3] => [5eps() 2e-3;
                                      5eps() 3e-4;
                                      5eps() 9e-4],
             smoothing_lengths[4] => [5eps() 5e-3;
                                      5eps() 3e-3;
                                      5eps() 2e-3])
quartic = Dict(smoothing_lengths[1] => [5eps() 1e-3;  # dim 1
                                        5eps() 5e-3;  # dim 2
                                        5eps() 9e-3], # dim 3
               smoothing_lengths[2] => [5eps() 5eps();
                                        5eps() 4e-4;
                                        10eps() 2e-4],
               smoothing_lengths[3] => [5eps() 2e-4;
                                        5eps() 3e-4;
                                        5eps() 2e-5],
               smoothing_lengths[4] => [5eps() 1e-3;
                                        5eps() 5e-4;
                                        15eps() 2e-4])

quintic = Dict(smoothing_lengths[1] => [5eps() 5e-4;  # dim 1
                                        5eps() 1e-3;  # dim 2
                                        5eps() 3e-3], # dim 3
               smoothing_lengths[2] => [5eps() 5eps();
                                        5eps() 7e-5;
                                        5eps() 3e-5],
               smoothing_lengths[3] => [5eps() 4e-5;
                                        5eps() 6e-5;
                                        10eps() 7e-6],
               smoothing_lengths[4] => [5eps() 3e-4;
                                        5eps() 9e-5;
                                        15eps() 5e-5])

for n_dims in dimensions
    mass = density_ref * particle_spacing^n_dims

    setup = RectangularShape(particle_spacing, ntuple(_ -> nx, n_dims),
                             ntuple(_ -> 0.0, n_dims), density_ref)
    coords = setup.coordinates

    smoothing_kernels = [SchoenbergCubicSplineKernel{n_dims}()
                         SchoenbergQuarticSplineKernel{n_dims}()
                         SchoenbergQuinticSplineKernel{n_dims}()]

    kernel_dict = Dict(SchoenbergCubicSplineKernel{n_dims}() => cubic,
                       SchoenbergQuarticSplineKernel{n_dims}() => quartic,
                       SchoenbergQuinticSplineKernel{n_dims}() => quintic)

    w = zeros(size(coords, 2), 1)
    normalize_w = zeros(size(coords, 2), 1)
    density = zeros(size(coords, 2), 1)

    # Find particle in the center of the domain.
    indices = LinearIndices(ntuple(_ -> nx, n_dims))
    nx_half = round(Int, nx / 2)
    particle_ = indices[CartesianIndex(ntuple(_ -> nx_half, n_dims))]

    for smoothing_kernel in smoothing_kernels, smoothing_length in smoothing_lengths
        compact_support = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)

        TrixiParticles.set_zero!(w)
        TrixiParticles.set_zero!(normalize_w)
        TrixiParticles.set_zero!(density)

        nhs = GridNeighborhoodSearch{n_dims}(compact_support, size(coords, 2);
                                             min_corner=[0.0 for i in 1:n_dims],
                                             max_corner=[domain_size for i in 1:n_dims])
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

        TrixiParticles.for_particle_neighbor(nothing, nothing, coords, coords, nhs,
                                             particles=axes(coords, 2)) do particle,
                                                                           neighbor,
                                                                           pos_diff,
                                                                           distance
            normalize_w[particle] += TrixiParticles.kernel(smoothing_kernel, distance,
                                                           smoothing_length) * mass /
                                     density[neighbor]
        end

        for neighbor in axes(coords, 2)
            pos_diff = coords[:, particle_] - coords[:, neighbor]
            w[neighbor] = TrixiParticles.kernel(smoothing_kernel, norm(pos_diff),
                                                smoothing_length)
        end

        dict = kernel_dict[smoothing_kernel]

        @info isapprox(normalize_w, ones(length(normalize_w));
                       rtol=dict[smoothing_length][n_dims, 1])

        @info isapprox(density, density_ref * ones(length(density));
                       rtol=dict[smoothing_length][n_dims, 2])

    end
end

#end
