@doc raw"""
    interpolate_line(start, end_, no_points, semi, ref_system, sol; endpoint=true,
                     smoothing_length=ref_system.smoothing_length)

Interpolates properties along a line in an TrixiParticles simulation.
The line interpolation is accomplished by generating a series of
evenly spaced points between `start` and `end_`.
If `endpoint` is `false`, the line is interpolated between the start and end points,
but does not include these points.

# Arguments
- `start`: The starting point of the line.
- `end_`: The ending point of the line.
- `no_points`: The number of points to interpolate along the line.
- `semi`: The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`: The solution state from which the properties are interpolated.

# Keywords
- `cut_off_bnd`: Cut-off at the boundary.
- `endpoint`: Optional. A boolean to include (`true`) or exclude (`false`) the end point in the interpolation. Default is `true`.
- `smoothing_length`: Optional. The smoothing length used in the interpolation. Default is `ref_system.smoothing_length`.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point along the line.

!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - When `cut_off_bnd` is used a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```julia
# Interpolating along a line from [1.0, 0.0] to [1.0, 1.0] with 5 points
results = interpolate_line([1.0, 0.0], [1.0, 1.0], 5, semi, ref_system, sol)
```
"""
function interpolate_line(start, end_, no_points, semi, ref_system, sol; endpoint=true,
                          smoothing_length=ref_system.smoothing_length,
                          cut_off_bnd=true)
    start_svector = SVector{ndims(ref_system)}(start)
    end_svector = SVector{ndims(ref_system)}(end_)
    points_coords = range(start_svector, end_svector, length=no_points)

    if !endpoint
        points_coords = points_coords[2:(end-1)]
    end

    return interpolate_point(points_coords, semi, ref_system, sol,
                             smoothing_length=smoothing_length,
                             cut_off_bnd=cut_off_bnd)
end

@doc raw"""
    interpolate_point(points_coords::Array{Array{Float64,1},1}, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length)

    interpolate_point(point_coords, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length)

Performs interpolation of properties at specified points or an array of points in a TrixiParticles simulation.

When given an array of points (`points_coords`), it iterates over each point and applies interpolation individually.
For a single point (`point_coords`), it performs the interpolation at that specific location.
The interpolation utilizes the same kernel function of the SPH simulation to weigh contributions from nearby particles.

# Arguments
- `points_coords`: An array of point coordinates, for which to interpolate properties.
- `point_coords`: The coordinates of a single point for interpolation.
- `semi`: The semidiscretization used in the SPH simulation.
- `ref_system`: The reference system defining the properties of the SPH particles.
- `sol`: The current solution state from which properties are interpolated.

# Keywords
- `cut_off_bnd`: Cut-off at the boundary.
- `smoothing_length`: Optional. The smoothing length used in the kernel function. Defaults to `ref_system.smoothing_length`.

# Returns:
- For multiple points:  A `NamedTuple` of arrays containing interpolated properties at each point.
- For a single point: A `NamedTuple` of containing interpolated properties at the point.

# Examples:
```julia
# For a single point
result = interpolate_point([1.0, 0.5], semi, ref_system, sol)

# For multiple points
points = [[1.0, 0.5], [1.0, 0.6], [1.0, 0.7]]
results = interpolate_point(points, semi, ref_system, sol)
```
!!! note
- This function is particularly useful for analyzing gradients or creating visualizations along a specified line in the SPH simulation domain.
- The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
- When `cut_off_bnd` is used a density-based estimation of the surface is used which is not as
  accurate as a real surface reconstruction.
"""
function interpolate_point(points_coords::AbstractArray{<:AbstractArray}, semi, ref_system,
                           sol; smoothing_length=ref_system.smoothing_length,
                           cut_off_bnd=true)
    num_points = length(points_coords)
    coords = similar(points_coords)
    velocities = similar(points_coords)
    densities = Vector{Float64}(undef, num_points)
    pressures = Vector{Float64}(undef, num_points)
    neighbor_counts = Vector{Int}(undef, num_points)

    neighborhood_searches = process_neighborhood_searches(semi, sol, ref_system,
    smoothing_length)

    for (i, point) in enumerate(points_coords)
        result = interpolate_point(SVector{ndims(ref_system)}(point), semi, ref_system, sol,
                                   neighborhood_searches, smoothing_length=smoothing_length,
                                   cut_off_bnd=cut_off_bnd)
        densities[i] = result.density
        neighbor_counts[i] = result.neighbor_count
        coords[i] = result.coord
        velocities[i] = result.velocity
        pressures[i] = result.pressure
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
            u = wrap_u(sol.u[end].x[2], system, semi)
            system_coords = current_coordinates(u, system)
            old_nhs = get_neighborhood_search(ref_system, system, semi)
            nhs = copy_neighborhood_search(old_nhs, search_radius, system_coords)
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
            else
                other_density += m_W
            end

            neighbor_count += 1
        end
    end

    # Point is not within the ref_system
    if other_density > interpolated_density || shepard_coefficient < eps()
        return (density=0.0, neighbor_count=0, coord=point_coords,
                velocity=zero(SVector{ndims(ref_system)}), pressure=0.0)
    end

    return (density=interpolated_density / shepard_coefficient,
            neighbor_count=neighbor_count,
            coord=point_coords, velocity=interpolated_velocity / shepard_coefficient,
            pressure=interpolated_pressure / shepard_coefficient)
end
