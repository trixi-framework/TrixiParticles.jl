using LinearAlgebra
using Printf
using Statistics
using TrixiParticles

function akinci_boundary_hydrodynamic_mass(initial_condition, smoothing_kernel,
                                           smoothing_length, reference_density)
    coordinates = initial_condition.coordinates
    particle_spacing = initial_condition.particle_spacing
    dimensions = size(coordinates, 1)
    support = TrixiParticles.compact_support(smoothing_kernel, smoothing_length)
    search_radius = ceil(Int, support / particle_spacing)
    origin = coordinates[:, 1]

    particle_at = Dict{NTuple{dimensions, Int}, Int}()
    for particle in axes(coordinates, 2)
        key = ntuple(dimensions) do dimension
            round(Int,
                  (coordinates[dimension, particle] - origin[dimension]) /
                  particle_spacing)
        end
        particle_at[key] = particle
    end

    offset_range = (-search_radius):search_radius
    offsets = Iterators.product(ntuple(_ -> offset_range, dimensions)...)
    mass = similar(initial_condition.mass)
    for particle in axes(coordinates, 2)
        key = ntuple(dimensions) do dimension
            round(Int,
                  (coordinates[dimension, particle] - origin[dimension]) /
                  particle_spacing)
        end
        number_density = zero(eltype(mass))
        for offset in offsets
            neighbor_key = ntuple(dimension -> key[dimension] + offset[dimension],
                                  dimensions)
            neighbor = get(particle_at, neighbor_key, 0)
            iszero(neighbor) && continue
            distance = norm(coordinates[:, particle] - coordinates[:, neighbor])
            distance < support || continue
            number_density += TrixiParticles.kernel(smoothing_kernel, distance,
                                                    smoothing_length)
        end
        mass[particle] = reference_density / number_density
    end

    return mass
end

function print_boundary_volume_summary(initial_condition, hydrodynamic_mass,
                                       reference_density)
    coordinates = initial_condition.coordinates
    particle_spacing = initial_condition.particle_spacing
    dimensions = size(coordinates, 1)
    nominal_mass = reference_density * particle_spacing^dimensions
    ratio = hydrodynamic_mass ./ nominal_mass
    z_values = sort(unique(coordinates[end, :]); rev=true)

    @printf("boundary particles: %d, nominal mass %.6g\n", length(ratio), nominal_mass)
    for z in z_values
        mask = isapprox.(coordinates[end, :], z;
                         atol=10eps(abs(z) + particle_spacing))
        layer_ratio = ratio[mask]
        @printf("  z=%9.6f: n=%5d ratio median=%7.4f range=[%7.4f, %7.4f]\n",
                z, count(mask), median(layer_ratio), minimum(layer_ratio),
                maximum(layer_ratio))
    end

    center_mask = trues(length(ratio))
    for dimension in 1:(dimensions - 1)
        center_mask .&= abs.(coordinates[dimension, :]) .< 2particle_spacing
    end
    top = maximum(coordinates[end, :])
    center_mask .&= isapprox.(coordinates[end, :], top;
                              atol=10eps(abs(top) + particle_spacing))
    @printf("  exposed center ratio: median=%7.4f range=[%7.4f, %7.4f]\n",
            median(ratio[center_mask]), minimum(ratio[center_mask]),
            maximum(ratio[center_mask]))

    return ratio
end
