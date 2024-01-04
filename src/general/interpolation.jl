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

### Keyword Arguments
- `cut_off_bnd`: Cut-off at the boundary.

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
                          cut_off_bnd=true)
    points_coords = [start +
                     (end_ - start) *
                     (endpoint ? t / (no_points - 1) : (t + 1) / (no_points + 1))
                     for t in 0:(no_points - 1)]

    return interpolate_point(points_coords, semi, ref_system, sol,
                             smoothing_length=smoothing_length,
                             cut_off_bnd=cut_off_bnd)
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

### Keyword Arguments
- `cut_off_bnd`: Cut-off at the boundary.

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
                           cut_off_bnd=true)
    densities = []
    neighbor_counts = []
    coords = []
    velocities = []
    pressures = []

    for point in points_coords
        result = interpolate_point(point, semi, ref_system, sol,
                                   smoothing_length=smoothing_length,
                                   cut_off_bnd=cut_off_bnd)
        push!(densities, result.density)
        push!(neighbor_counts, result.neighbor_count)
        push!(coords, result.coord)
        push!(velocities, result.velocity)
        push!(pressures, result.pressure)
    end

    return (density=densities, neighbor_count=neighbor_counts, coord=coords,
            velocity=velocities, pressure=pressures)
end

function interpolate_point(point_coords, semi, ref_system, sol;
                           smoothing_length=ref_system.smoothing_length,
                           cut_off_bnd=true)
    neighborhood_searches = process_neighborhood_searches(semi, sol, ref_system,
                                                          smoothing_length)

    return interpolate_point(SVector{ndims(ref_system)}(point_coords), semi, ref_system,
                             sol, neighborhood_searches, smoothing_length=smoothing_length,
                             cut_off_bnd=cut_off_bnd)
end

function process_neighborhood_searches(semi, sol, ref_system, smoothing_length)
    if isapprox(smoothing_length, ref_system.smoothing_length)
        # Update existing NHS
        update_nhs(sol.u[end].x[2], semi)
        neighborhood_searches = semi.neighborhood_searches[system_indices(ref_system, semi)]
    else
        ref_smoothing_kernel = ref_system.smoothing_kernel
        search_radius = compact_support(ref_smoothing_kernel, smoothing_length)
        neighborhood_searches = map(semi.systems) do system
            u = wrap_u(sol[end].x[2], system, semi)
            system_coords = current_coordinates(u, system)
            old_nhs = get_neighborhood_search(ref_system, system, semi)
            nhs = create_neighborhood_search(system_coords, system, old_nhs, search_radius)
            return nhs
        end
    end

    return neighborhood_searches
end

@inline function interpolate_point(point_coords, semi, ref_system, sol,
                                   neighborhood_searches;
                                   smoothing_length=ref_system.smoothing_length,
                                   cut_off_bnd=true)
    interpolated_density = 0.0
    interpolated_velocity = zero(SVector{ndims(ref_system)})
    interpolated_pressure = 0.0

    shepard_coefficient = 0.0
    ref_id = system_indices(ref_system, semi)
    neighbor_count = 0
    ref_density = 0.0
    other_density = 0.0
    ref_smoothing_kernel = ref_system.smoothing_kernel

    if cut_off_bnd
        systems = semi
    else
        systems = (ref_system,)
    end

    foreach_system(systems) do system
        system_id = system_indices(system, semi)
        nhs = neighborhood_searches[system_id]
        (; search_radius, periodic_box) = nhs

        v = wrap_v(sol.u[end].x[1], system, semi)
        u = wrap_u(sol.u[end].x[2], system, semi)

        system_coords = current_coordinates(u, system)

        for particle in eachneighbor(point_coords, nhs)
            coords = extract_svector(system_coords, Val(ndims(system)), particle)

            pos_diff = point_coords - coords
            distance2 = dot(pos_diff, pos_diff)
            pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2,
                                                            search_radius, periodic_box)
            if distance2 > search_radius^2
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
                interpolated_velocity += particle_velocity * (volume * kernel_value)

                particle_pressure = pressure(system, particle)
                interpolated_pressure += particle_pressure * (volume * kernel_value)
                shepard_coefficient += volume * kernel_value
            end

            if system_id === ref_id
                ref_density += m_W
            else
                other_density += m_W
            end

            neighbor_count += 1
        end
    end

    # point is not within the ref_system
    if other_density > ref_density || shepard_coefficient < eps()
        return (density=0.0, neighbor_count=0, coord=point_coords,
                velocity=zeros(size(point_coords)), pressure=0.0)
    end

    return (density=interpolated_density / shepard_coefficient,
            neighbor_count=neighbor_count,
            coord=point_coords, velocity=interpolated_velocity / shepard_coefficient,
            pressure=interpolated_pressure / shepard_coefficient)
end
