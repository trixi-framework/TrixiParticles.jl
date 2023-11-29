struct system_properties
end

function interpolate_point(point_coords, semi)
    density = 0.0

    foreach_system(semi) do system
        u_neighbor_system = wrap_u(u_ode, neighbor_system, semi)

        system_coords = current_coordinates(u, system)
        nhs = get_neighborhood_search(system, semi)

        for neighbor in eachneighbor(point_coords, nhs)
            distance = norm(point_coords - system_coords[neighbor])
            density += mass[neighbor] * smoothing_kernel(system, distance)
        end
    end

    return density
end
