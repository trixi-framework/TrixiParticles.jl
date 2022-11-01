function deformation_gradient(u, i, j, particle, semi)
    @unpack cache, neighborhood_search, smoothing_kernel, smoothing_length = semi
    @unpack mass, solid_density, initial_coordinates, correction_matrix = cache

    return sum(eachneighbor(particle, initial_coordinates, neighborhood_search, semi)) do neighbor
        volume = mass[neighbor] / solid_density[neighbor]
        pos_diff = get_particle_coords(u, semi, neighbor) -
            get_particle_coords(u, semi, particle)

        initial_pos_diff = get_particle_coords(initial_coordinates, semi, particle) -
            get_particle_coords(initial_coordinates, semi, neighbor)
        initial_distance = norm(initial_pos_diff)

        if initial_distance < eps()
            return zero(eltype(mass))
        end

        grad_kernel = kernel_deriv(smoothing_kernel, initial_distance, smoothing_length) *
            dot(view(correction_matrix, :, j, particle), initial_pos_diff) / initial_distance

        return volume * pos_diff[i] * grad_kernel
    end
end
