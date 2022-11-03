function deformation_gradient(current_coordinates, i, j, particle, semi)
    @unpack cache, neighborhood_search, smoothing_kernel, smoothing_length = semi
    @unpack mass, solid_density, initial_coordinates, correction_matrix = cache

    result = zero(eltype(mass))

    for neighbor in eachneighbor(particle, initial_coordinates, neighborhood_search, semi)
        volume = mass[neighbor] / solid_density[neighbor]
        pos_diff = get_particle_coords(current_coordinates, semi, neighbor) -
            get_particle_coords(current_coordinates, semi, particle)

        initial_pos_diff = get_particle_coords(initial_coordinates, semi, particle) -
            get_particle_coords(initial_coordinates, semi, neighbor)
        initial_distance = norm(initial_pos_diff)

        if initial_distance > eps()
            grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
                dot(get_correction_matrix_column(semi, j, particle), initial_pos_diff) / initial_distance

            result += volume * pos_diff[i] * grad_kernel
        end
    end

    return result
end

# We cannot use a variable for the number of dimensions here, it has to be hardcoded
@inline function deformation_gradient(current_coordinates, particle, semi::SPHSemidiscretization{2})
    return @SMatrix [deformation_gradient(current_coordinates, i, j, particle, semi) for i in 1:2, j in 1:2]
end

@inline function deformation_gradient(current_coordinates, particle, semi::SPHSemidiscretization{3})
    return @SMatrix [deformation_gradient(current_coordinates, i, j, particle, semi) for i in 1:3, j in 1:3]
end


# First Piola-Kirchhoff stress tensor
function pk1_stress_tensor(current_coordinates, particle, semi)
    J = deformation_gradient(current_coordinates, particle, semi)

    S = pk2_stress_tensor(J, semi)

    return J * S
end

# Second Piola-Kirchhoff stress tensor
@inline function pk2_stress_tensor(J, semi)
    @unpack cache = semi
    @unpack lame_lambda, lame_mu = cache

    # Compute the Green-Lagrange strain
    E = 0.5 * (transpose(J) * J - I)

    return lame_lambda * tr(E) * I + 2 * lame_mu * E
end

include("sph_rhs.jl")
