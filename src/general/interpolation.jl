struct system_properties
end

function interpolate_point(point_coords, semi, sol, smoothing_length)
    density = 0.0
    neighbor_count = 0

    foreach_system(semi) do system
        #v = wrap_v(sol[end].x[1], system, semi)
        u = wrap_u(sol[end].x[2], system, semi)

        system_coords = current_coordinates(u, system)
        nhs = get_neighborhood_search(system, semi)
        #search_radius2 = system.smoothing_length^2
        search_radius2 = smoothing_length^2

        for particle in eachneighbor(point_coords, nhs)
            coords = extract_svector(system_coords, Val(ndims(system)), particle)

            pos_diff = point_coords - coords
            distance2 = dot(pos_diff, pos_diff)
            if distance2 > search_radius2
                continue
            end

            mass = hydrodynamic_mass(system, particle)
            density += mass * smoothing_kernel(system, sqrt(distance2))
            neighbor_count += 1
        end
    end

    return density
end
