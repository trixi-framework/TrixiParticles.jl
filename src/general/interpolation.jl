function interpolate_line(start, end_, no_points, semi, ref_system, sol; endpoint=true, smoothing_length=ref_system.smoothing_length)
    if endpoint
        # Include both start and end points
        points_coords = [start + (end_ - start) * t / (no_points - 1) for t in 0:(no_points - 1)]
    else
        # Exclude start and end points
        points_coords = [start + (end_ - start) * (t + 1) / (no_points + 1) for t in 0:(no_points - 1)]
    end

    return interpolate_point(points_coords, semi, ref_system, sol, smoothing_length=smoothing_length)
end


function interpolate_point(points_coords::Array{Array{Float64,1},1}, semi, ref_system, sol; smoothing_length=ref_system.smoothing_length)
    results = []

    for point in points_coords
        result = interpolate_point(point, semi, ref_system, sol, smoothing_length=smoothing_length)
        push!(results, result)
    end

    return results
end

function interpolate_point(point_coords, semi, ref_system, sol; smoothing_length=ref_system.smoothing_length)
    density = 0.0
    shepard_coefficient = 0.0
    ref_id = system_indices(ref_system, semi)
    neighbor_count = 0
    ref_density = 0.0
    other_density = 0.0
    ref_smoothing_kernel = ref_system.smoothing_kernel
    search_radius = compact_support(ref_smoothing_kernel, smoothing_length)
    search_radius2 = search_radius^2

    foreach_system(semi) do system
        system_id = system_indices(system, semi)
        v = wrap_v(sol[end].x[1], system, semi)
        u = wrap_u(sol[end].x[2], system, semi)

        system_coords = current_coordinates(u, system)
        nhs = get_neighborhood_search(system, semi)
        nhs = create_neighborhood_search(system, nhs, search_radius)

        for particle in eachneighbor(point_coords, nhs)
            coords = extract_svector(system_coords, Val(ndims(system)), particle)

            pos_diff = point_coords - coords
            distance2 = dot(pos_diff, pos_diff)
            pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, nhs)
            if distance2 > search_radius2
                continue
            end

            distance = sqrt(distance2)
            mass = hydrodynamic_mass(system, particle)
            volume = mass/particle_density(v, system, particle)
            kernel_value = kernel(ref_smoothing_kernel, distance, smoothing_length)


            m_W = mass * kernel_value
            density += m_W
            shepard_coefficient += volume * kernel_value

            if system_id === ref_id
                ref_density += m_W
            else
                other_density += m_W
            end

            neighbor_count +=1
        end
    end

    # point is not within the ref_system
    if other_density > ref_density
        return (density=0.0, neighbor_count=0, coord=point_coords)
    end

    return (density=density/shepard_coefficient, neighbor_count=neighbor_count, coord=point_coords)
end
