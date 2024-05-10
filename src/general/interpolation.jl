using LinearAlgebra

@doc raw"""
    interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system, sol;
                         smoothing_length=ref_system.smoothing_length, cut_off_bnd=true,
                         clip_negative_pressure=false)

Interpolates properties along a plane in a TrixiParticles simulation.
The region for interpolation is defined by its lower left and top right corners,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points within the defined region,
spaced uniformly according to the given resolution.

See also: [`interpolate_plane_2d_vtk`](@ref), [`interpolate_plane_3d`](@ref),
          [`interpolate_line`](@ref), [`interpolate_point`](@ref).

# Arguments
- `min_corner`: The lower left corner of the interpolation region.
- `max_corner`: The top right corner of the interpolation region.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `smoothing_length=ref_system.smoothing_length`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point within the plane.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used, which is not as
      accurate as a real surface reconstruction.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing); ref_system = fluid_system)
# Interpolating across a plane from [0.0, 0.0] to [1.0, 1.0] with a resolution of 0.2
results = interpolate_plane_2d([0.0, 0.0], [1.0, 1.0], 0.2, semi, ref_system, sol)

# output
(density = ...)
```
"""
function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system,
                              sol::ODESolution;
                              smoothing_length=ref_system.smoothing_length,
                              cut_off_bnd=true, clip_negative_pressure=false)
    # Filter out particles without neighbors
    filter_no_neighbors = true
    v_ode, u_ode = sol.u[end].x

    results, _, _ = interpolate_plane_2d(min_corner, max_corner, resolution,
                                         semi, ref_system, v_ode, u_ode,
                                         filter_no_neighbors, smoothing_length, cut_off_bnd,
                                         clip_negative_pressure)

    return results
end

function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system,
                              v_ode, u_ode;
                              smoothing_length=ref_system.smoothing_length,
                              cut_off_bnd=true, clip_negative_pressure=false)
    # Filter out particles without neighbors
    filter_no_neighbors = true

    results, _, _ = interpolate_plane_2d(min_corner, max_corner, resolution,
                                         semi, ref_system, v_ode, u_ode,
                                         filter_no_neighbors, smoothing_length, cut_off_bnd,
                                         clip_negative_pressure)

    return results
end

@doc raw"""
    interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system, sol;
                             smoothing_length=ref_system.smoothing_length, cut_off_bnd=true,
                             clip_negative_pressure=false, output_directory="out", filename="plane")

Interpolates properties along a plane in a TrixiParticles simulation and exports the result
as a VTI file.
The region for interpolation is defined by its lower left and top right corners,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points within the defined region,
spaced uniformly according to the given resolution.

See also: [`interpolate_plane_2d`](@ref), [`interpolate_plane_3d`](@ref),
          [`interpolate_line`](@ref), [`interpolate_point`](@ref).

# Arguments
- `min_corner`: The lower left corner of the interpolation region.
- `max_corner`: The top right corner of the interpolation region.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `smoothing_length=ref_system.smoothing_length`: The smoothing length used in the interpolation.
- `output_directory="out"`: Directory to save the VTI file.
- `filename="plane"`:       Name of the VTI file.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used, which is not as
      accurate as a real surface reconstruction.

# Examples
```julia
# Interpolating across a plane from [0.0, 0.0] to [1.0, 1.0] with a resolution of 0.2
results = interpolate_plane_2d([0.0, 0.0], [1.0, 1.0], 0.2, semi, ref_system, sol)
```
"""
function interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system,
                                  sol::ODESolution; clip_negative_pressure=false,
                                  smoothing_length=ref_system.smoothing_length,
                                  cut_off_bnd=true,
                                  output_directory="out", filename="plane")
    v_ode, u_ode = sol.u[end].x

    interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system,
                             v_ode, u_ode; clip_negative_pressure,
                             smoothing_length, cut_off_bnd, output_directory, filename)
end

function interpolate_plane_2d_vtk(min_corner, max_corner, resolution, semi, ref_system,
                                  v_ode, u_ode;
                                  smoothing_length=ref_system.smoothing_length,
                                  cut_off_bnd=true, clip_negative_pressure=false,
                                  output_directory="out", filename="plane")
    # Don't filter out particles without neighbors to keep 2D grid structure
    filter_no_neighbors = false
    results, x_range, y_range = interpolate_plane_2d(min_corner, max_corner, resolution,
                                                     semi, ref_system, v_ode, u_ode,
                                                     filter_no_neighbors,
                                                     smoothing_length, cut_off_bnd,
                                                     clip_negative_pressure)

    density = reshape(results.density, length(x_range), length(y_range))
    velocity = reshape(results.velocity, length(x_range), length(y_range))
    pressure = reshape(results.pressure, length(x_range), length(y_range))

    vtk_grid(joinpath(output_directory, filename), x_range, y_range) do vtk
        vtk["density"] = density
        vtk["velocity"] = velocity
        vtk["pressure"] = pressure
    end
end

function interpolate_plane_2d(min_corner, max_corner, resolution, semi, ref_system,
                              v_ode, u_ode, filter_no_neighbors, smoothing_length,
                              cut_off_bnd, clip_negative_pressure)
    dims = length(min_corner)
    if dims != 2 || length(max_corner) != 2
        throw(ArgumentError("function is intended for 2D coordinates only"))
    end

    if any(min_corner .> max_corner)
        throw(ArgumentError("`min_corner` should be smaller than `max_corner` in every dimension"))
    end

    # Calculate the number of points in each dimension based on the resolution
    no_points_x = ceil(Int, (max_corner[1] - min_corner[1]) / resolution) + 1
    no_points_y = ceil(Int, (max_corner[2] - min_corner[2]) / resolution) + 1

    x_range = range(min_corner[1], max_corner[1], length=no_points_x)
    y_range = range(min_corner[2], max_corner[2], length=no_points_y)

    # Generate points within the plane
    points_coords = [SVector(x, y) for x in x_range, y in y_range]

    results = interpolate_point(points_coords, semi, ref_system, v_ode, u_ode,
                                smoothing_length=smoothing_length,
                                cut_off_bnd=cut_off_bnd,
                                clip_negative_pressure=clip_negative_pressure)

    if filter_no_neighbors
        # Find indices where neighbor_count > 0
        indices = findall(x -> x > 0, results.neighbor_count)

        # Filter all arrays in the named tuple using these indices
        results = map(x -> x[indices], results)
    end

    return results, x_range, y_range
end

@doc raw"""
    interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system, sol;
                         smoothing_length=ref_system.smoothing_length, cut_off_bnd=true,
                         clip_negative_pressure=false)

Interpolates properties along a plane in a 3D space in a TrixiParticles simulation.
The plane for interpolation is defined by three points in 3D space,
with a specified resolution determining the density of the interpolation points.

The function generates a grid of points on a parallelogram within the plane defined by the
three points, spaced uniformly according to the given resolution.

See also: [`interpolate_plane_2d`](@ref), [`interpolate_plane_2d_vtk`](@ref),
          [`interpolate_line`](@ref), [`interpolate_point`](@ref).

# Arguments
- `point1`:     The first point defining the plane.
- `point2`:     The second point defining the plane.
- `point3`:     The third point defining the plane. The points must not be collinear.
- `resolution`: The distance between adjacent interpolation points in the grid.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `smoothing_length=ref_system.smoothing_length`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point within the plane.

!!! note
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_3d.jl"), tspan=(0.0, 0.01)); ref_system = fluid_system)
# Interpolating across a plane defined by points [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], and [0.0, 1.0, 0.0]
# with a resolution of 0.1
results = interpolate_plane_3d([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.1, semi, ref_system, sol)

# output
(density = ...)
```
"""
function interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system,
                              sol::ODESolution;
                              smoothing_length=ref_system.smoothing_length,
                              cut_off_bnd=true, clip_negative_pressure=false)
    v_ode, u_ode = sol.u[end].x

    interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system,
                         v_ode, u_ode; smoothing_length, cut_off_bnd,
                         clip_negative_pressure)
end

function interpolate_plane_3d(point1, point2, point3, resolution, semi, ref_system,
                              v_ode, u_ode; smoothing_length=ref_system.smoothing_length,
                              cut_off_bnd=true, clip_negative_pressure=false)
    if ndims(ref_system) != 3
        throw(ArgumentError("`interpolate_plane_3d` requires a 3D simulation"))
    end

    coords, resolution_ = sample_plane((point1, point2, point3), resolution)

    if !isapprox(resolution, resolution_, rtol=5e-2)
        @info "The desired plane size is not a multiple of the resolution $resolution." *
              "\nNew resolution is set to $resolution_."
    end

    points_coords = reinterpret(reshape, SVector{3, Float64}, coords)

    # Interpolate using the generated points
    results = interpolate_point(points_coords, semi, ref_system, v_ode, u_ode,
                                smoothing_length=smoothing_length,
                                cut_off_bnd=cut_off_bnd,
                                clip_negative_pressure=clip_negative_pressure)

    # Filter results
    indices = findall(x -> x > 0, results.neighbor_count)
    filtered_results = map(x -> x[indices], results)

    return filtered_results
end

@doc raw"""
    interpolate_line(start, end_, n_points, semi, ref_system, sol; endpoint=true,
                     smoothing_length=ref_system.smoothing_length, cut_off_bnd=true,
                     clip_negative_pressure=false)

Interpolates properties along a line in a TrixiParticles simulation.
The line interpolation is accomplished by generating a series of
evenly spaced points between `start` and `end_`.
If `endpoint` is `false`, the line is interpolated between the start and end points,
but does not include these points.

See also: [`interpolate_point`](@ref), [`interpolate_plane_2d`](@ref),
          [`interpolate_plane_2d_vtk`](@ref), [`interpolate_plane_3d`](@ref).

# Arguments
- `start`:      The starting point of the line.
- `end_`:       The ending point of the line.
- `n_points`:   The number of points to interpolate along the line.
- `semi`:       The semidiscretization used for the simulation.
- `ref_system`: The reference system for the interpolation.
- `sol`:        The solution state from which the properties are interpolated.

# Keywords
- `endpoint=true`: A boolean to include (`true`) or exclude (`false`) the end point in the interpolation.
- `smoothing_length=ref_system.smoothing_length`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.

# Returns
- A `NamedTuple` of arrays containing interpolated properties at each point along the line.

!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
      accurate as a real surface reconstruction.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing); ref_system = fluid_system)
# Interpolating along a line from [1.0, 0.0] to [1.0, 1.0] with 5 points
results = interpolate_line([1.0, 0.0], [1.0, 1.0], 5, semi, ref_system, sol)

# output
(density = ...)
```
"""
function interpolate_line(start, end_, n_points, semi, ref_system, sol::ODESolution;
                          endpoint=true, smoothing_length=ref_system.smoothing_length,
                          cut_off_bnd=true, clip_negative_pressure=false)
    v_ode, u_ode = sol.u[end].x

    interpolate_line(start, end_, n_points, semi, ref_system, v_ode, u_ode;
                     endpoint, smoothing_length, cut_off_bnd, clip_negative_pressure)
end
function interpolate_line(start, end_, n_points, semi, ref_system, v_ode, u_ode;
                          endpoint=true, smoothing_length=ref_system.smoothing_length,
                          cut_off_bnd=true, clip_negative_pressure=false)
    start_svector = SVector{ndims(ref_system)}(start)
    end_svector = SVector{ndims(ref_system)}(end_)
    points_coords = range(start_svector, end_svector, length=n_points)

    if !endpoint
        points_coords = points_coords[2:(end - 1)]
    end

    return interpolate_point(points_coords, semi, ref_system, v_ode, u_ode;
                             smoothing_length=smoothing_length,
                             cut_off_bnd=cut_off_bnd, clip_negative_pressure)
end

@doc raw"""
    interpolate_point(points_coords::Array{Array{Float64,1},1}, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length, cut_off_bnd=true,
                      clip_negative_pressure=false)

    interpolate_point(point_coords, semi, ref_system, sol;
                      smoothing_length=ref_system.smoothing_length, cut_off_bnd=true,
                      clip_negative_pressure=false)

Performs interpolation of properties at specified points or an array of points in a TrixiParticles simulation.

When given an array of points (`points_coords`), it iterates over each point and applies interpolation individually.
For a single point (`point_coords`), it performs the interpolation at that specific location.
The interpolation utilizes the same kernel function of the SPH simulation to weigh contributions from nearby particles.

See also: [`interpolate_line`](@ref), [`interpolate_plane_2d`](@ref),
          [`interpolate_plane_2d_vtk`](@ref), [`interpolate_plane_3d`](@ref), .

# Arguments
- `points_coords`:  An array of point coordinates, for which to interpolate properties.
- `point_coords`:   The coordinates of a single point for interpolation.
- `semi`:           The semidiscretization used in the SPH simulation.
- `ref_system`:     The reference system defining the properties of the SPH particles.
- `sol`:            The current solution state from which properties are interpolated.

# Keywords
- `smoothing_length=ref_system.smoothing_length`: The smoothing length used in the interpolation.
- `cut_off_bnd=true`: Boolean to indicate if quantities should be set to `NaN` when the point
                      is "closer" to the boundary than to the fluid in a kernel-weighted sense.
                      Or, in more detail, when the boundary has more influence than the fluid
                      on the density summation in this point, i.e., when the boundary particles
                      add more kernel-weighted mass than the fluid particles.
- `clip_negative_pressure=false`: One common approach in SPH models is to clip negative pressure
                                  values, but this is unphysical. Instead we clip here during
                                  interpolation thus only impacting the local interpolated value.

# Returns
- For multiple points:  A `NamedTuple` of arrays containing interpolated properties at each point.
- For a single point: A `NamedTuple` of interpolated properties at the point.

# Examples
```jldoctest; output = false, filter = r"density = .*", setup = :(trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hydrostatic_water_column_2d.jl"), tspan=(0.0, 0.01), callbacks=nothing); ref_system = fluid_system)
# For a single point
result = interpolate_point([1.0, 0.5], semi, ref_system, sol)

# For multiple points
points = [[1.0, 0.5], [1.0, 0.6], [1.0, 0.7]]
results = interpolate_point(points, semi, ref_system, sol)

# output
(density = ...)
```
!!! note
    - This function is particularly useful for analyzing gradients or creating visualizations
      along a specified line in the SPH simulation domain.
    - The interpolation accuracy is subject to the density of particles and the chosen smoothing length.
    - With `cut_off_bnd`, a density-based estimation of the surface is used which is not as
    accurate as a real surface reconstruction.
"""
@inline function interpolate_point(point_coords, semi, ref_system, sol::ODESolution;
                                   smoothing_length=ref_system.smoothing_length,
                                   cut_off_bnd=true, clip_negative_pressure=false)
    v_ode, u_ode = sol.u[end].x

    interpolate_point(point_coords, semi, ref_system, v_ode, u_ode;
                      smoothing_length, cut_off_bnd, clip_negative_pressure)
end

@inline function interpolate_point(points_coords::AbstractArray{<:AbstractArray}, semi,
                                   ref_system, v_ode, u_ode;
                                   smoothing_length=ref_system.smoothing_length,
                                   cut_off_bnd=true, clip_negative_pressure=false)
    num_points = length(points_coords)
    coords = similar(points_coords)
    velocities = similar(points_coords)
    densities = Vector{Float64}(undef, num_points)
    pressures = Vector{Float64}(undef, num_points)
    neighbor_counts = Vector{Int}(undef, num_points)

    neighborhood_searches = process_neighborhood_searches(semi, u_ode, ref_system,
                                                          smoothing_length)

    for (i, point) in enumerate(points_coords)
        result = interpolate_point(SVector{ndims(ref_system)}(point), semi, ref_system,
                                   v_ode, u_ode, neighborhood_searches;
                                   smoothing_length, cut_off_bnd, clip_negative_pressure)
        densities[i] = result.density
        neighbor_counts[i] = result.neighbor_count
        coords[i] = result.coord
        velocities[i] = result.velocity
        pressures[i] = result.pressure
    end

    return (density=densities, neighbor_count=neighbor_counts, coord=coords,
            velocity=velocities, pressure=pressures)
end

function interpolate_point(point_coords, semi, ref_system, v_ode, u_ode;
                           smoothing_length=ref_system.smoothing_length,
                           cut_off_bnd=true, clip_negative_pressure=false)
    neighborhood_searches = process_neighborhood_searches(semi, u_ode, ref_system,
                                                          smoothing_length)

    return interpolate_point(SVector{ndims(ref_system)}(point_coords), semi, ref_system,
                             v_ode, u_ode, neighborhood_searches;
                             smoothing_length, cut_off_bnd, clip_negative_pressure)
end

function process_neighborhood_searches(semi, u_ode, ref_system, smoothing_length)
    if isapprox(smoothing_length, ref_system.smoothing_length)
        # Update existing NHS
        update_nhs!(semi, u_ode)
        neighborhood_searches = semi.neighborhood_searches[system_indices(ref_system, semi)]
    else
        ref_smoothing_kernel = ref_system.smoothing_kernel
        search_radius = compact_support(ref_smoothing_kernel, smoothing_length)
        neighborhood_searches = map(semi.systems) do system
            u = wrap_u(u_ode, system, semi)
            system_coords = current_coordinates(u, system)
            old_nhs = get_neighborhood_search(ref_system, system, semi)
            nhs = PointNeighbors.copy_neighborhood_search(old_nhs, search_radius,
                                                                   system_coords)
            return nhs
        end
    end

    return neighborhood_searches
end

@inline function interpolate_point(point_coords, semi, ref_system, v_ode, u_ode,
                                   neighborhood_searches;
                                   smoothing_length=ref_system.smoothing_length,
                                   cut_off_bnd=true, clip_negative_pressure=false)
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
        # Don't loop over other systems
        systems = (ref_system,)
    end

    foreach_system(systems) do system
        system_id = system_indices(system, semi)
        nhs = neighborhood_searches[system_id]
        (; search_radius, periodic_box) = nhs

        v = wrap_v(v_ode, system, semi)
        u = wrap_u(u_ode, system, semi)

        system_coords = current_coordinates(u, system)

        # This is basically `for_particle_neighbor` unrolled
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

                pressure = particle_pressure(v, system, particle)
                if clip_negative_pressure
                    pressure = max(0.0, pressure)
                end

                interpolated_pressure += pressure * (volume * kernel_value)
                shepard_coefficient += volume * kernel_value
            else
                other_density += m_W
            end

            neighbor_count += 1
        end
    end

    # Point is not within the ref_system
    if other_density > interpolated_density || shepard_coefficient < eps()
        # Return NaN values that can be filtered out in ParaView
        return (density=NaN, neighbor_count=0, coord=point_coords,
                velocity=fill(NaN, SVector{ndims(ref_system)}), pressure=NaN)
    end

    return (density=interpolated_density / shepard_coefficient,
            neighbor_count=neighbor_count,
            coord=point_coords, velocity=interpolated_velocity / shepard_coefficient,
            pressure=interpolated_pressure / shepard_coefficient)
end
