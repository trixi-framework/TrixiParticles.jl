@doc raw"""
    interpolate_plane(lower_left, top_right, resolution, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length)

Interpolates properties across a plane or a volume in an SPH simulation environment.
The region for interpolation is defined by its lower left and top right corners, with a specified resolution determining the density of the interpolation points.

The function generates a grid of points within the defined region, spaced uniformly according to the given resolution. In 2D, this grid lies within a plane,
 while in 3D, it fills a volumetric space. The function can handle both 2D and 3D interpolations by adapting to the dimensionality of the provided corner coordinates.

### Parameters:
- `lower_left`: The lower left corner of the interpolation region.
- `top_right`: The top right corner of the interpolation region.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`: The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`: The solution state from which the properties are interpolated.
- `smoothing_length`: Optional. The smoothing length used in the interpolation. Defaults to `ref_system.smoothing_length`.

### Returns:
An array of interpolated properties at each point within the defined region. Points with a neighbor count of 0 are excluded from the results.

!!! note
    - This function is especially useful for analyzing spatial variations of properties within a specified region in the SPH simulation domain.
    - The accuracy of the interpolation is influenced by the local particle density, the chosen smoothing length, and the resolution of the interpolation grid.

## Example
```julia
# Interpolating across a plane from [0.0, 0.0] to [1.0, 1.0] with a resolution of 0.2
results = interpolate_plane([0.0, 0.0], [1.0, 1.0], 0.2, semi, ref_system, sol)
```
"""
function interpolate_plane(lower_left, top_right, resolution, semi, ref_system, sol;
                           smoothing_length=ref_system.smoothing_length,
                           calculate_other_system_density=false)
    dims = length(lower_left)
    if dims != length(top_right)
        error("Dimensions of lower_left and top_right must match")
    end

    # Check that lower_left is indeed lower and to the left of top_right
    for i in 1:dims
        if lower_left[i] >= top_right[i]
            error("lower_left should be lower and to the left of top_right in all dimensions")
        end
    end

    # Calculate the number of points in each dimension based on the resolution
    no_points_x = ceil(Int, (top_right[1] - lower_left[1]) / resolution) + 1
    no_points_y = ceil(Int, (top_right[2] - lower_left[2]) / resolution) + 1
    no_points_z = dims == 3 ? ceil(Int, (top_right[3] - lower_left[3]) / resolution) + 1 : 1

    x_range = range(lower_left[1], top_right[1], length=no_points_x)
    y_range = range(lower_left[2], top_right[2], length=no_points_y)
    z_range = dims == 3 ? range(lower_left[3], top_right[3], length=no_points_z) : 1:1

    points_coords = dims == 2 ? [[x, y] for x in x_range, y in y_range] :
                    [[x, y, z] for x in x_range, y in y_range, z in z_range]

    results = []
    for point in points_coords
        result = interpolate_point(point, semi, ref_system, sol,
                                   smoothing_length=smoothing_length,
                                   calculate_other_system_density=calculate_other_system_density)
        push!(results, result)
    end

    # Filter out results with neighbor_count of 0
    results = filter(r -> r.neighbor_count > 0, results)

    return results
end

@doc raw"""
    interpolate_line(start, end_, no_points, semi, ref_system, sol; endpoint=true, smoothing_length=ref_system.smoothing_length)

Interpolates properties along a line in an SPH simulation environment.
The line is defined by its start and end points, and the number of points to interpolate along this line is specified.
The function can optionally include or exclude the line's endpoint in the interpolation process.

The line interpolation is accomplished by generating a series of evenly spaced points between `start` and `end_`.
The number of these points is determined by `no_points`. When `endpoint` is `true`, both the start and end points are included in the interpolation.
If `endpoint` is `false`, the line is interpolated between the start and end points but does not include these points.

The function relies on the existing `interpolate_point` function for performing the actual interpolation at each point.

### Parameters:
- `start`: The starting point of the line.
- `end_`: The ending point of the line.
- `no_points`: The number of points to interpolate along the line.
- `semi`: The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`: The solution state from which the properties are interpolated.
- `endpoint`: Optional. A boolean to include (`true`) or exclude (`false`) the end point in the interpolation. Default is `true`.
- `smoothing_length`: Optional. The smoothing length used in the interpolation. Default is `ref_system.smoothing_length`.

### Returns:
An array of interpolated properties at each point along the line.

!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.

## Example
```julia
# Interpolating along a line from [1.0, 0.0] to [1.0, 1.0] with 5 points
results = interpolate_line([1.0, 0.0], [1.0, 1.0], 5, semi, ref_system, sol)
```
"""
function interpolate_line(start, end_, no_points, semi, ref_system, sol; endpoint=true,
                          smoothing_length=ref_system.smoothing_length,
                          calculate_other_system_density=false)
    points_coords = [start +
                     (end_ - start) *
                     (endpoint ? t / (no_points - 1) : (t + 1) / (no_points + 1))
                     for t in 0:(no_points - 1)]

    results = []
    for point in points_coords
        result = interpolate_point(point, semi, ref_system, sol,
                                   smoothing_length=smoothing_length,
                                   calculate_other_system_density=calculate_other_system_density)
        push!(results, result)
    end

    return results
end

@doc raw"""
    interpolate_point(points_coords::Array{Array{Float64,1},1}, semi, ref_system, sol; smoothing_length=ref_system.smoothing_length)

    interpolate_point(point_coords, semi, ref_system, sol; smoothing_length=ref_system.smoothing_length)

Performs interpolation of properties at specified points or an array of points in an SPH (Smoothed Particle Hydrodynamics) simulation.
This function can handle either a single point or multiple points.

When given an array of points (`points_coords`), it iterates over each point and applies interpolation individually.
For a single point (`point_coords`), it performs the interpolation at that specific location.
The interpolation is based on the SPH method, utilizing a kernel function to weigh contributions from nearby particles.

### Parameters:
- `points_coords`: An array of points (each being an Array{Float64,1}) for which to interpolate properties.
- `point_coords`: The coordinates of a single point (Array{Float64,1}) for interpolation.
- `semi`: The semidiscretization used in the SPH simulation.
- `ref_system`: The reference system defining the properties of the SPH particles.
- `sol`: The current solution state from which properties are interpolated.
- `smoothing_length`: Optional. The smoothing length used in the kernel function. Defaults to `ref_system.smoothing_length`.

### Returns:
- For multiple points: An array of results, each containing the interpolated property (e.g., density), the neighbor count, and the coordinates of the point.
- For a single point: A tuple containing the interpolated property, the neighbor count, and the coordinates of the point.

### Usage:
```julia
# For a single point
result = interpolate_point([1.0, 0.5], semi, ref_system, sol)

# For multiple points
points = [[1.0, 0.5], [1.0, 0.6], [1.0, 0.7]]
results = interpolate_point(points, semi, ref_system, sol)
```
!!! note
- The accuracy of interpolation depends on the local particle density and the smoothing length.
- This function is particularly useful for extracting physical properties at specific locations within the SPH simulation domain.
"""
function interpolate_point(points_coords::Array{Array{Float64, 1}, 1}, semi, ref_system,
                           sol; smoothing_length=ref_system.smoothing_length,
                           calculate_other_system_density=false)
    results = []

    for point in points_coords
        result = interpolate_point(point, semi, ref_system, sol,
                                   smoothing_length=smoothing_length,
                                   calculate_other_system_density=calculate_other_system_density)
        push!(results, result)
    end

    return results
end

function interpolate_point(point_coords, semi, ref_system, sol;
                           smoothing_length=ref_system.smoothing_length,
                           calculate_other_system_density=false)
    interpolated_density = 0.0
    interpolated_velocity = zeros(size(point_coords))
    interpolated_pressure = 0.0

    shepard_coefficient = 0.0
    ref_id = system_indices(ref_system, semi)
    neighbor_count = 0
    ref_density = 0.0
    other_density = 0.0
    ref_smoothing_kernel = ref_system.smoothing_kernel
    search_radius = compact_support(ref_smoothing_kernel, smoothing_length)
    search_radius2 = search_radius^2
    min_dist_ref = 10 * search_radius
    min_dist_other = 10 * search_radius

    if calculate_other_system_density
        systems = semi
    else
        systems = (ref_system,)
    end

    foreach_system(systems) do system
        system_id = system_indices(system, semi)
        v = wrap_v(sol[end].x[1], system, semi)
        u = wrap_u(sol[end].x[2], system, semi)

        system_coords = current_coordinates(u, system)
        nhs = get_neighborhood_search(system, semi)
        if system isa FluidSystem
            nhs = create_neighborhood_search(u, system, nhs, search_radius)
        else
            nhs = create_neighborhood_search(initial_coordinates(system), system, nhs,
                                             search_radius)
        end

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
            kernel_value = kernel(ref_smoothing_kernel, distance, smoothing_length)
            m_W = mass * kernel_value

            if system_id == ref_id
                interpolated_density += m_W

                volume = mass / particle_density(v, system, particle)
                particle_velocity = current_velocity(v, system, particle)
                interpolated_velocity .+= particle_velocity * (volume * kernel_value)

                particle_pressure = pressure(system, particle)
                interpolated_pressure += particle_pressure * (volume * kernel_value)
                shepard_coefficient += volume * kernel_value
            end

            if system_id === ref_id
                ref_density += m_W
                if min_dist_ref > distance
                    min_dist_ref = distance
                end
            else
                other_density += m_W
                if min_dist_other > distance
                    min_dist_other = distance
                end
            end

            neighbor_count += 1
        end
    end

    # point is not within the ref_system
    if (other_density > ref_density && min_dist_other < min_dist_ref) ||
       shepard_coefficient < eps()
        return (density=0.0, neighbor_count=0, coord=point_coords,
                velocity=zeros(size(point_coords)), pressure=0.0)
    end

    return (density=interpolated_density / shepard_coefficient,
            neighbor_count=neighbor_count,
            coord=point_coords, velocity=interpolated_velocity / shepard_coefficient,
            pressure=interpolated_pressure / shepard_coefficient)
end
